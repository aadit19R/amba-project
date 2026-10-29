"""Train-fitted BigMart cleaning and feature engineering for the EDA and later CV."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


TARGET = "Item_Outlet_Sales"
ID_COLUMNS = ["Item_Identifier", "Outlet_Identifier"]
RAW_FEATURES = [
    "Item_Identifier", "Item_Weight", "Item_Fat_Content", "Item_Visibility",
    "Item_Type", "Item_MRP", "Outlet_Identifier", "Outlet_Establishment_Year",
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
]


def _mode(values):
    modes = values.dropna().mode()
    return modes.iloc[0] if len(modes) else np.nan


class BigMartPreprocessor(TransformerMixin, BaseEstimator):
    """Learn all statistics on fit rows; transform never learns from new rows.

    ``clean`` exposes imputed, unencoded features for the notebook's EDA.
    ``engineer`` adds the documented feature variants to a cleaned frame.
    ``transform_frame`` retains identifiers for CSV export.
    ``transform`` returns only numeric predictors, suitable for sklearn Pipeline.
    Unknown nominal categories have an all-zero block, using the fitted schema.
    ``visibility_policy='keep_zero'`` preserves observed zeros for model comparisons;
    the default retains the EDA's existing positive-mean imputation behaviour.
    """

    def __init__(self, reference_year=2013, visibility_policy="impute"):
        self.reference_year = reference_year
        self.visibility_policy = visibility_policy

    def _base_frame(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Supply a pandas DataFrame with the original BigMart columns.")
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        # Selecting explicitly prevents targets and their transforms entering X.
        frame = X.loc[:, RAW_FEATURES].copy()
        string_cols = [c for c in RAW_FEATURES if c not in [
            "Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"
        ]]
        for col in string_cols:
            frame[col] = frame[col].astype("string").str.strip().replace("", pd.NA)
        frame["Item_Fat_Content"] = frame["Item_Fat_Content"].replace({
            "LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"
        })
        for col in ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            if not np.isfinite(frame[col].dropna().to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite values in {col}")
        if frame["Item_MRP"].isna().any() or frame["Item_MRP"].le(0).any():
            raise ValueError("Item_MRP must be observed and positive.")
        years = frame["Outlet_Establishment_Year"]
        if years.isna().any() or years.le(0).any() or years.gt(self.reference_year).any():
            raise ValueError("Outlet establishment years must precede the reference year.")
        required = ["Item_Identifier", "Outlet_Identifier", "Item_Type", "Outlet_Type",
                    "Item_Fat_Content", "Outlet_Location_Type"]
        if frame[required].isna().any().any():
            raise ValueError("Missing required identifiers or categorical features.")
        frame["Item_Weight"] = frame["Item_Weight"].where(frame["Item_Weight"].gt(0))
        return frame

    def fit(self, X, y=None):
        if self.visibility_policy not in {"impute", "keep_zero"}:
            raise ValueError("visibility_policy must be 'impute' or 'keep_zero'.")
        frame = self._base_frame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.weight_by_item_ = frame.groupby("Item_Identifier")["Item_Weight"].mean()
        self.weight_by_type_ = frame.groupby("Item_Type")["Item_Weight"].median()
        self.weight_global_ = frame["Item_Weight"].median()
        self.size_by_outlet_ = frame.groupby("Outlet_Identifier")["Outlet_Size"].agg(_mode)
        self.size_by_type_ = frame.groupby("Outlet_Type")["Outlet_Size"].agg(_mode)
        self.size_global_ = _mode(frame["Outlet_Size"])
        positive = frame.loc[frame["Item_Visibility"].gt(0)]
        self.visibility_by_item_ = positive.groupby("Item_Identifier")["Item_Visibility"].mean()
        self.visibility_by_type_ = positive.groupby("Outlet_Type")["Item_Visibility"].median()
        self.visibility_global_ = positive["Item_Visibility"].median()
        if (pd.isna(self.weight_global_) or pd.isna(self.size_global_)
                or pd.isna(self.visibility_global_)):
            raise ValueError("Fit rows must contain observed weight, size and positive visibility.")
        cleaned = self.clean(frame)
        q1, q3 = cleaned["Item_Visibility"].quantile([0.25, 0.75])
        self.visibility_lower_ = max(0.0, q1 - 1.5 * (q3 - q1))
        self.visibility_upper_ = q3 + 1.5 * (q3 - q1)
        self.visibility_reference_ = cleaned.groupby("Item_Identifier")["Item_Visibility"].mean()
        engineered = self.engineer(cleaned)
        self.nominal_columns_ = ["Item_Fat_Content", "Item_Category", "Item_Type", "Outlet_Type", "Outlet"]
        self.nominal_categories_ = {
            col: sorted(engineered[col].dropna().unique().tolist()) for col in self.nominal_columns_
        }
        encoded = self._encode(engineered)
        self.output_columns_ = encoded.columns.tolist()
        self.predictor_columns_ = [c for c in self.output_columns_ if c not in ID_COLUMNS]
        return self

    def clean(self, X):
        check_is_fitted(self, "weight_by_item_")
        frame = self._base_frame(X)
        frame["Item_Weight_Missing"] = frame["Item_Weight"].isna().astype(int)
        frame["Outlet_Size_Missing"] = frame["Outlet_Size"].isna().astype(int)
        invalid_visibility = frame["Item_Visibility"].isna() | frame["Item_Visibility"].le(0)
        frame["Item_Visibility_ZeroOrMissing"] = invalid_visibility.astype(int)
        frame["Item_Weight"] = (
            frame["Item_Weight"]
            .fillna(frame["Item_Identifier"].map(self.weight_by_item_))
            .fillna(frame["Item_Type"].map(self.weight_by_type_))
            .fillna(self.weight_global_)
        )
        frame["Outlet_Size"] = (
            frame["Outlet_Size"]
            .fillna(frame["Outlet_Identifier"].map(self.size_by_outlet_))
            .fillna(frame["Outlet_Type"].map(self.size_by_type_))
            .fillna(self.size_global_)
        )
        fill_visibility = invalid_visibility if self.visibility_policy == "impute" else (
            frame["Item_Visibility"].isna() | frame["Item_Visibility"].lt(0)
        )
        frame["Item_Visibility"] = (
            frame["Item_Visibility"].mask(fill_visibility)
            .fillna(frame["Item_Identifier"].map(self.visibility_by_item_))
            .fillna(frame["Outlet_Type"].map(self.visibility_by_type_))
            .fillna(self.visibility_global_)
        )
        return frame

    def engineer(self, cleaned):
        check_is_fitted(self, "visibility_reference_")
        frame = cleaned.copy()
        frame["Item_Category"] = frame["Item_Identifier"].str[:2].map({
            "FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"
        }).fillna("Unknown")
        frame.loc[frame["Item_Category"].eq("Non-Consumable"), "Item_Fat_Content"] = "Non-Edible"
        frame["Outlet_Years"] = self.reference_year - frame["Outlet_Establishment_Year"]
        frame["Item_Visibility_iqr"] = frame["Item_Visibility"].clip(
            self.visibility_lower_, self.visibility_upper_
        )
        reference = (frame["Item_Identifier"].map(self.visibility_reference_)
                     .where(lambda s: s.gt(0))
                     .fillna(frame["Outlet_Type"].map(self.visibility_by_type_))
                     .fillna(self.visibility_global_))
        frame["Item_Visibility_MeanRatio"] = frame["Item_Visibility"] / reference
        frame["MRP_Band"] = pd.cut(frame["Item_MRP"], [0, 69, 136, 203, np.inf],
                                   labels=["Low", "Medium", "High", "Very High"])
        frame["Is_Grocery"] = frame["Outlet_Type"].eq("Grocery Store").astype(int)
        frame["Item_Visibility_sqrt"] = np.sqrt(frame["Item_Visibility"])
        frame["Item_Visibility_log"] = np.log1p(frame["Item_Visibility"])
        frame["Outlet"] = frame["Outlet_Identifier"]
        return frame

    def _encode(self, frame):
        frame = frame.copy()
        for source, dest, mapping in [
            ("Outlet_Size", "Outlet_Size_code", {"Small": 0, "Medium": 1, "High": 2}),
            ("Outlet_Location_Type", "Outlet_Location_code", {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}),
        ]:
            frame[dest] = frame[source].map(mapping)
            if frame[dest].isna().any():
                raise ValueError(f"Unknown ordered category in {source}")
            frame[dest] = frame[dest].astype(int)
        frame["MRP_Band_code"] = frame["MRP_Band"].cat.codes.astype(int)
        for col, categories in self.nominal_categories_.items():
            known_values = frame[col].where(frame[col].isin(categories))
            frame[col] = pd.Categorical(known_values, categories=categories)
        frame = pd.get_dummies(frame, columns=self.nominal_columns_, dtype=int)
        return frame.drop(columns=["Outlet_Size", "Outlet_Location_Type", "MRP_Band",
                                   "Outlet_Establishment_Year"])

    def transform_frame(self, X):
        check_is_fitted(self, "output_columns_")
        return self._encode(self.engineer(self.clean(X))).reindex(columns=self.output_columns_)

    def transform(self, X):
        return self.transform_frame(X).loc[:, self.predictor_columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "predictor_columns_")
        return np.asarray(self.predictor_columns_, dtype=object)
