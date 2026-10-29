#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BigMart Sales Prediction: standalone Python copy of the complete notebook.

Includes all notebook code, explanatory Markdown as comments, and the original
preprocessing/model-selection helpers. No local helper modules, notebook file,
existing fitted models or existing result caches are needed.

Install the project's dependencies, then run:
    python3 BigMart_Sales_Prediction.py --data-dir .

Required input files:
    train_v9rqX0R.csv
    test_AbJTz2l.csv
    sample_submission_8RXa3c6 (1).csv

Outputs default to ./python_export_outputs/ to keep the notebook's existing
artifacts separate. All figures are saved as PNGs. Use --show-plots to also open
interactive plot windows, or --output-dir PATH to choose another output folder.
The first run performs the full CV searches; matching results are reused later.

Upload this .py as the code file and submission.csv as the solution file.
"""

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

SOURCE_NOTEBOOK_SHA256 = "30ebf09485a9e7d75cc09517040abeea558655f94940155d74b7cdbad1753a67"
NOTEBOOK_CELL_COUNT = 120
NOTEBOOK_CODE_CELL_COUNT = 69

# =============================================================================
# Bundled preprocessing.py (same cleaning and engineering as the notebook)
# =============================================================================
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

# =============================================================================
# Bundled modeling.py (same models, searches, scoring and evaluation policy)
# =============================================================================
"""Reproducible, fold-fitted BigMart model selection and final inference.

Run ``python3 -m modeling`` from the project folder for the complete experiment.
The notebook calls the same functions. Content-addressed caches avoid rerunning
unchanged searches; changing inputs, code, versions or search settings invalidates
them. The reserved modelling check is used only after a winner has been frozen.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_config
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, cross_val_predict,
    cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits



SEED = 42
MODEL_ORDER = ["LinearRegression", "KNN", "SVR", "RandomForest"]
PROFILES = ["compact", "descriptive_ordered", "descriptive_onehot"]
SEARCH_BUDGETS = {"KNN": 16, "SVR": 20, "RandomForest": 24}
VISIBILITY_OPTIONS = [
    ("impute", "raw"), ("impute", "sqrt"),
    ("impute", "capped"), ("keep_zero", "raw"),
]
TARGET_MODES = ["raw", "sqrt", "log1p"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def frame_hash(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def package_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__}




@contextmanager
def computation_context():
    # Threads work without local worker sockets in restricted environments.
    # Each estimator and numerical library stays single-threaded inside a fold.
    with threadpool_limits(limits=1), parallel_config(backend="threading"):
        yield


class ModelFeatures(TransformerMixin, BaseEstimator):
    """Select a documented profile using only the fitted input feature schema."""

    def __init__(self, profile="compact", visibility="raw", include_ratio=False):
        self.profile = profile
        self.visibility = visibility
        self.include_ratio = include_ratio

    @staticmethod
    def _block(frame, prefix):
        # The first training-observed level is the reference category.
        return sorted(c for c in frame if c.startswith(prefix))[1:]

    def fit(self, X, y=None):
        if self.profile not in PROFILES + ["price_interactions"]:
            raise ValueError(f"Unknown feature profile: {self.profile}")
        if self.visibility not in {"raw", "sqrt", "capped"}:
            raise ValueError(f"Unknown visibility representation: {self.visibility}")
        self.item_columns_ = self._block(X, "Item_Type_")
        prefix = "Outlet_OUT" if self.profile in {"compact", "price_interactions"} else "Outlet_Type_"
        self.outlet_columns_ = self._block(X, prefix)
        self.interaction_columns_ = self._block(X, "Outlet_Type_")
        self.metadata_levels_ = {
            col: sorted(X[col].unique().tolist())
            for col in ["Outlet_Size_code", "Outlet_Location_code"]
        }
        self.output_columns_ = self._assemble(X).columns.tolist()
        return self

    def _assemble(self, X):
        visibility_col = {"raw": "Item_Visibility", "sqrt": "Item_Visibility_sqrt",
                          "capped": "Item_Visibility_iqr"}[self.visibility]
        out = X[["Item_MRP", "Item_Weight", "Item_Visibility_ZeroOrMissing"]].copy()
        out["Visibility"] = X[visibility_col]
        # With item-type columns, the non-edible indicator is redundant.
        out["Item_Fat_Content_Regular"] = X.get("Item_Fat_Content_Regular", 0)
        for col in self.item_columns_ + self.outlet_columns_:
            out[col] = X[col]
        if self.profile.startswith("descriptive"):
            for col in ["Outlet_Years", "MRP_Band_code", "Item_Weight_Missing", "Outlet_Size_Missing"]:
                out[col] = X[col]
            for col, levels in self.metadata_levels_.items():
                if self.profile == "descriptive_ordered":
                    out[col] = X[col]
                else:
                    for level in levels[1:]:
                        out[f"{col}_level_{level}"] = X[col].eq(level).astype(int)
        if self.profile == "price_interactions":
            for col in self.interaction_columns_:
                out[f"MRP_x_{col}"] = X["Item_MRP"] * X[col]
        if self.include_ratio:
            out["Item_Visibility_MeanRatio"] = X["Item_Visibility_MeanRatio"]
        return out

    def transform(self, X):
        check_is_fitted(self, "output_columns_")
        return self._assemble(X).loc[:, self.output_columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


class FeatureScaler(TransformerMixin, BaseEstimator):
    """Scale numerical magnitudes while leaving binary indicators unchanged."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()
        binary_prefixes = ("Item_Type_", "Outlet_OUT", "Outlet_Type_")
        binary_names = {"Item_Fat_Content_Regular", "Item_Visibility_ZeroOrMissing",
                        "Item_Weight_Missing", "Outlet_Size_Missing"}
        self.numeric_columns_ = [
            c for c in X if not (c.startswith(binary_prefixes) or c in binary_names or "_level_" in c)
        ]
        if self.enabled:
            self.scaler_ = StandardScaler().fit(X[self.numeric_columns_])
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        out = X.loc[:, self.columns_].copy()
        if self.enabled:
            scaled = self.scaler_.transform(out[self.numeric_columns_])
            for i, col in enumerate(self.numeric_columns_):
                out[col] = scaled[:, i]
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")
        return np.asarray(self.columns_, dtype=object)


def transform_target(y, mode):
    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Sales targets must be finite and nonnegative.")
    if mode == "raw":
        return values
    if mode == "sqrt":
        return np.sqrt(values)
    if mode == "log1p":
        return np.log1p(values)
    raise ValueError(f"Unknown target mode: {mode}")


def inverse_target(values, mode):
    values = np.asarray(values, dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        if mode == "raw":
            result = values
        elif mode == "sqrt":
            result = np.maximum(values, 0) ** 2
        elif mode == "log1p":
            result = np.expm1(values)
        else:
            raise ValueError(f"Unknown target mode: {mode}")
    if not np.isfinite(result).all():
        raise ValueError("Estimator produced non-finite sales predictions.")
    return np.maximum(result, 0)


class SalesRegressor(RegressorMixin, BaseEstimator):
    """Clone a raw-input pipeline, transform y, and predict original-scale sales."""

    def __init__(self, pipeline, target_mode="raw"):
        self.pipeline = pipeline
        self.target_mode = target_mode

    def fit(self, X, y):
        self.pipeline_ = clone(self.pipeline).fit(X, transform_target(y, self.target_mode))
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        check_is_fitted(self, "pipeline_")
        return inverse_target(self.pipeline_.predict(X), self.target_mode)


def build_model(name):
    if name == "MeanBaseline":
        return SalesRegressor(Pipeline([("model", DummyRegressor(strategy="mean"))]))
    models = {
        "LinearRegression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=31, weights="uniform", p=2, n_jobs=1),
        "SVR": SVR(kernel="rbf", C=1000, gamma="scale", epsilon=100),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=100,
            max_features=1.0, random_state=SEED, n_jobs=1,
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return SalesRegressor(Pipeline([
        ("preprocess", BigMartPreprocessor()),
        ("features", ModelFeatures()),
        ("scale", FeatureScaler(enabled=name != "RandomForest")),
        ("model", models[name]),
    ]))


def development_split(train):
    dev, check = train_test_split(
        np.arange(len(train)), test_size=.2, random_state=SEED,
        stratify=train["Outlet_Identifier"],
    )
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(dev))
    assignments = pd.DataFrame({"row_index": np.arange(len(train)), "partition": "reserved_check",
                                "cv_fold": pd.array([pd.NA] * len(train), dtype="Int64")})
    assignments.loc[dev, "partition"] = "development"
    for number, (_, valid_idx) in enumerate(folds, 1):
        assignments.loc[dev[valid_idx], "cv_fold"] = number
    return dev, check, folds, assignments


def search_space(name):
    feature_key = "pipeline__features__profile"
    if name == "LinearRegression":
        return {feature_key: PROFILES + ["price_interactions"], "target_mode": TARGET_MODES}
    parameters = {
        "KNN": {
            "n_neighbors": [5, 11, 21, 31, 51, 81],
            "weights": ["uniform", "distance"], "p": [1, 2],
        },
        "SVR": {
            "C": [10, 100, 1000, 10000], "gamma": ["scale", .001, .01, .1],
            "epsilon": [.01, .1, 1, 10, 100],
        },
        "RandomForest": {
            "n_estimators": [200, 400, 600], "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [1, 5, 20, 50, 100], "max_features": [.5, .75, 1.0],
        },
    }[name]
    shared = {f"pipeline__model__{k}": v for k, v in parameters.items()}
    shared.update({feature_key: PROFILES, "target_mode": TARGET_MODES})
    if name == "RandomForest":
        shared["pipeline__features__include_ratio"] = [False, True]
    return [dict(shared, **{
        "pipeline__preprocess__visibility_policy": [policy],
        "pipeline__features__visibility": [representation],
    }) for policy, representation in VISIBILITY_OPTIONS]


def _candidate_record(name, stage, parameters, test_scores, train_scores, fit_times, score_times):
    model = build_model(name).set_params(**parameters)
    settings = model.pipeline.named_steps
    return {
        "model": name, "stage": stage, "parameters": _jsonable(parameters),
        "profile": settings["features"].profile if "features" in settings else "none",
        "visibility": settings["features"].visibility if "features" in settings else "none",
        "visibility_policy": settings["preprocess"].visibility_policy if "preprocess" in settings else "none",
        "include_ratio": settings["features"].include_ratio if "features" in settings else False,
        "target_mode": model.target_mode,
        "cv_rmse": float(np.mean(test_scores)), "cv_std": float(np.std(test_scores, ddof=0)),
        "train_rmse": float(np.mean(train_scores)), "fit_seconds": float(np.mean(fit_times)),
        "predict_seconds": float(np.mean(score_times)),
        "fold_rmse": [float(v) for v in test_scores],
    }


def comparison_table(selection):
    rows = []
    for row in selection["finalists"]:
        rows.append({k: v for k, v in row.items() if k not in {"parameters", "fold_rmse"}})
    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


def run_selection(train, output_dir="model_outputs", n_jobs=4, reuse_results=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fingerprint = experiment_fingerprint(train)
    cache = output_dir / "selection_results.joblib"
    if reuse_results and cache.exists():
        result = joblib.load(cache)
        if result["fingerprint"] == fingerprint:
            print("Loaded matching development-CV results; no search repeated.", flush=True)
            return result

    def progress(message):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with (output_dir / "progress.log").open("a") as stream:
            stream.write(line + "\n")

    dev, check, folds, assignments = development_split(train)
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    X_dev = train.iloc[dev][RAW_FEATURES]
    y_dev = train.iloc[dev][TARGET]
    baselines, tuned, search_results = [], [], {}
    progress(f"Development={len(dev)}, reserved check={len(check)}; five shared folds.")
    with computation_context():
        for name in ["MeanBaseline"] + MODEL_ORDER:
            progress(f"Baseline: {name}")
            scores = cross_validate(
                build_model(name), X_dev, y_dev, cv=folds, n_jobs=n_jobs,
                scoring="neg_root_mean_squared_error", return_train_score=True,
                error_score="raise",
            )
            row = _candidate_record(name, "baseline", {}, -scores["test_score"],
                                    -scores["train_score"], scores["fit_time"], scores["score_time"])
            baselines.append(row)
            progress(f"{name}: CV RMSE={row['cv_rmse']:.3f} ± {row['cv_std']:.3f}")
        for name in MODEL_ORDER:
            budget = 12 if name == "LinearRegression" else SEARCH_BUDGETS[name]
            progress(f"Search: {name}, {budget} candidates × five folds")
            kwargs = dict(estimator=build_model(name), scoring="neg_root_mean_squared_error",
                          cv=folds, n_jobs=n_jobs, refit=False,
                          return_train_score=True, error_score="raise")
            if name == "LinearRegression":
                search = GridSearchCV(param_grid=search_space(name), **kwargs)
            else:
                search = RandomizedSearchCV(param_distributions=search_space(name),
                                            n_iter=budget, random_state=SEED, **kwargs)
            search.fit(X_dev, y_dev)
            results = pd.DataFrame(search.cv_results_)
            results["params"] = results["params"].map(lambda p: json.dumps(_jsonable(p), sort_keys=True))
            results.to_csv(output_dir / f"search_{name}.csv", index=False)
            search_results[name] = results
            idx = search.best_index_
            raw = search.cv_results_
            row = _candidate_record(
                name, "tuned", search.best_params_,
                [-raw[f"split{i}_test_score"][idx] for i in range(5)],
                [-raw[f"split{i}_train_score"][idx] for i in range(5)],
                [raw["mean_fit_time"][idx]], [raw["mean_score_time"][idx]],
            )
            tuned.append(row)
            progress(f"{name}: best searched CV RMSE={row['cv_rmse']:.3f} ± {row['cv_std']:.3f}")

    def rank(row):
        return (row["cv_rmse"], row["predict_seconds"],
                MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else -1)

    finalists = [baselines[0]]
    for name in MODEL_ORDER:
        finalists.append(min([r for r in baselines + tuned if r["model"] == name], key=rank))
    winner = min(finalists[1:], key=rank)
    if winner["cv_rmse"] >= baselines[0]["cv_rmse"]:
        raise RuntimeError("No approved model improved on the development mean baseline.")
    result = {"fingerprint": fingerprint, "seed": SEED, "dev_indices": dev,
              "check_indices": check, "folds": folds, "baselines": baselines,
              "tuned": tuned, "finalists": finalists, "winner": winner,
              "search_results": search_results, "versions": package_versions()}
    joblib.dump(result, cache)
    write_json(output_dir / "frozen_selection.json", {
        "fingerprint": fingerprint, "winner": winner,
        "selection_basis": "Development CV only; reserved-check metrics not yet computed.",
    })
    comparison_table(result).to_csv(output_dir / "model_comparison.csv", index=False)
    write_json(output_dir / "baseline_results.json", baselines)
    progress(f"FROZEN WINNER: {winner['model']} ({winner['stage']}), CV RMSE={winner['cv_rmse']:.3f}")
    return result


def selected_estimator(selection):
    winner = selection["winner"]
    return build_model(winner["model"]).set_params(**winner["parameters"])


def regression_metrics(actual, predicted):
    return {"rmse": float(root_mean_squared_error(actual, predicted)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted))}


def prediction_frame(raw, predicted, partition, tail_cutoff):
    out = raw[ID_COLUMNS + ["Outlet_Type", "Item_MRP", TARGET]].copy()
    out.insert(0, "row_index", raw.index)
    out["partition"] = partition
    out["prediction"] = predicted
    out["residual"] = out[TARGET] - predicted
    out["absolute_error"] = out["residual"].abs()
    out["MRP_Band"] = pd.cut(out.Item_MRP, [0, 69, 136, 203, np.inf],
                             labels=["Low", "Medium", "High", "Very High"])
    out["Sales_segment"] = np.where(out[TARGET].gt(tail_cutoff), "Above development p90", "At/below development p90")
    return out.sort_index()


def group_errors(predictions, column):
    rows = []
    for level, group in predictions.groupby(column, observed=True):
        rows.append({column: str(level), "rows": len(group),
                     **regression_metrics(group[TARGET], group.prediction),
                     "mean_residual": float(group.residual.mean())})
    return pd.DataFrame(rows)


def evaluate_selected(train, selection, output_dir="model_outputs", n_jobs=4, reuse_results=True):
    output_dir = Path(output_dir)
    if experiment_fingerprint(train) != selection["fingerprint"]:
        raise ValueError("Selection cache does not match the training data/code.")
    frozen = json.loads((output_dir / "frozen_selection.json").read_text())
    if frozen["fingerprint"] != selection["fingerprint"] or frozen["winner"] != _jsonable(selection["winner"]):
        raise ValueError("Frozen selection is missing or inconsistent.")
    cache = output_dir / "evaluation_results.joblib"
    if reuse_results and cache.exists():
        result = joblib.load(cache)
        if result["fingerprint"] == selection["fingerprint"]:
            print("Loaded the frozen winner's saved OOF/check evaluation.", flush=True)
            return result
    dev, check = selection["dev_indices"], selection["check_indices"]
    X_dev, y_dev = train.iloc[dev][RAW_FEATURES], train.iloc[dev][TARGET]
    X_check, y_check = train.iloc[check][RAW_FEATURES], train.iloc[check][TARGET]
    print("Winner already frozen. Generating development OOF predictions...", flush=True)
    with computation_context():
        oof = cross_val_predict(selected_estimator(selection), X_dev, y_dev,
                                cv=selection["folds"], n_jobs=n_jobs, method="predict")
        estimator = selected_estimator(selection).fit(X_dev, y_dev)
        check_predictions = estimator.predict(X_check)
        baseline_predictions = np.full(len(y_check), y_dev.mean())
        metrics = {
            "development_oof": regression_metrics(y_dev, oof),
            "reserved_check": regression_metrics(y_check, check_predictions),
            "reserved_mean_baseline": regression_metrics(y_check, baseline_predictions),
        }
        print(f"Reserved-check RMSE: {metrics['reserved_check']['rmse']:.3f}", flush=True)
        print("Permutation importance: frozen model, reserved check, ten repeats.", flush=True)
        importance = permutation_importance(
            estimator, X_check, y_check, scoring="neg_root_mean_squared_error",
            n_repeats=10, random_state=SEED, n_jobs=n_jobs,
        )
    tail_cutoff = y_dev.quantile(.9)
    oof_frame = prediction_frame(train.iloc[dev], oof, "development_oof", tail_cutoff)
    check_frame = prediction_frame(train.iloc[check], check_predictions, "reserved_check", tail_cutoff)
    importance_frame = pd.DataFrame({"feature": RAW_FEATURES,
        "rmse_increase_mean": importance.importances_mean,
        "rmse_increase_std": importance.importances_std}).sort_values("rmse_increase_mean", ascending=False)
    oof_frame.to_csv(output_dir / "development_oof_predictions.csv", index=False)
    check_frame.to_csv(output_dir / "reserved_check_predictions.csv", index=False)
    importance_frame.to_csv(output_dir / "permutation_importance.csv", index=False)
    for partition, frame in [("development_oof", oof_frame), ("reserved_check", check_frame)]:
        for col in ["Outlet_Identifier", "Outlet_Type", "MRP_Band", "Sales_segment"]:
            group_errors(frame, col).to_csv(output_dir / f"errors_{partition}_{col}.csv", index=False)
    write_json(output_dir / "evaluation_metrics.json", metrics)
    joblib.dump(estimator, output_dir / "development_model.joblib")
    result = {"fingerprint": selection["fingerprint"], "metrics": metrics,
              "oof_predictions": oof_frame, "check_predictions": check_frame,
              "importance": importance_frame, "development_tail_cutoff": float(tail_cutoff)}
    joblib.dump(result, cache)
    return result


def validate_submission(submission, test, sample):
    if list(submission.columns) != ID_COLUMNS + [TARGET]:
        raise ValueError("Submission columns do not match the required order.")
    if len(submission) != len(test) or len(sample) != len(test):
        raise ValueError("Submission/test/template row counts differ.")
    pd.testing.assert_frame_equal(submission[ID_COLUMNS].reset_index(drop=True),
                                  test[ID_COLUMNS].reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(sample[ID_COLUMNS].reset_index(drop=True),
                                  test[ID_COLUMNS].reset_index(drop=True), check_dtype=False)
    values = submission[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Submission sales must be finite and nonnegative.")


def build_final(train, test, sample, selection, evaluation, output_dir="model_outputs",
                submission_path="submission.csv"):
    output_dir = Path(output_dir)
    if experiment_fingerprint(train) != selection["fingerprint"] or evaluation["fingerprint"] != selection["fingerprint"]:
        raise ValueError("Selection/evaluation fingerprints do not match the current run.")
    print("Refitting the frozen configuration on all training rows...", flush=True)
    with computation_context():
        estimator = selected_estimator(selection).fit(train[RAW_FEATURES], train[TARGET])
        predictions = estimator.predict(test[RAW_FEATURES])
        model_path = output_dir / "final_model.joblib"
        joblib.dump(estimator, model_path)
        reloaded = joblib.load(model_path)
        np.testing.assert_allclose(reloaded.predict(test[RAW_FEATURES]), predictions, rtol=1e-12, atol=1e-12)
    submission = test[ID_COLUMNS].copy()
    submission[TARGET] = predictions
    validate_submission(submission, test, sample)
    submission.to_csv(submission_path, index=False)
    validate_submission(pd.read_csv(submission_path), test, sample)
    selected_names = estimator.pipeline_.named_steps["features"].get_feature_names_out().tolist()
    model_settings = estimator.pipeline_.named_steps["model"].get_params(deep=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": selection["fingerprint"], "versions": package_versions(),
        "train_frame_sha256": frame_hash(train), "test_frame_sha256": frame_hash(test),
        "raw_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            DATA_DIR / "train_v9rqX0R.csv", DATA_DIR / "test_AbJTz2l.csv",
        ]},
        "winner": selection["winner"], "estimator_parameters": model_settings,
        "selected_features": selected_names, "selected_feature_count": len(selected_names),
        "metrics": evaluation["metrics"], "development_rows": len(selection["dev_indices"]),
        "reserved_check_rows": len(selection["check_indices"]), "final_fit_rows": len(train),
        "test_rows": len(test), "seed": SEED, "submission": str(submission_path),
        "selection_policy": "Frozen from development CV; no retuning on reserved-check results.",
        "limitations": ["Full-dataset EDA preceded the modelling split.",
                         "Only ten existing outlets; not an assessment of unseen-store generalisation.",
                         "Local RMSE is not a public leaderboard score."],
    }
    write_json(output_dir / "run_metadata.json", metadata)
    print(f"Saved {model_path}; validated {len(submission)} submission rows.", flush=True)
    return metadata






def experiment_fingerprint(train):
    """Fingerprint the bundled source itself; no sibling Python files required."""
    digest = hashlib.sha256(frame_hash(train).encode())
    digest.update(Path(__file__).read_bytes())
    digest.update(json.dumps(package_versions(), sort_keys=True).encode())
    return digest.hexdigest()


# Keep fitted estimators importable when a model saved by this script is reloaded.
# Importing this file defines helpers without running any notebook workflow.
_EXPORT_MODULE = "BigMart_Sales_Prediction"
sys.modules.setdefault(_EXPORT_MODULE, sys.modules[__name__])
for _exported_class in (BigMartPreprocessor, ModelFeatures, FeatureScaler, SalesRegressor):
    _exported_class.__module__ = _EXPORT_MODULE


class Markdown:
    """Plain-Python representation of a notebook Markdown display."""

    def __init__(self, data):
        self.data = data


def display(*values):
    """Print notebook table/text displays without requiring an IPython kernel."""
    for value in values:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            print(value.to_string())
        elif isinstance(value, Markdown):
            print(value.data)
        else:
            print(value)


def _display_result(value):
    if value is not None:
        display(value)


def run_notebook(data_dir=None, output_dir="python_export_outputs", show_plots=False):
    """Execute all original notebook cells in order with bundled helpers."""
    global DATA_DIR
    DATA_DIR = Path(data_dir).resolve() if data_dir is not None else Path(__file__).resolve().parent
    OUTPUT_DIR = Path(output_dir).resolve()
    required = ["train_v9rqX0R.csv", "test_AbJTz2l.csv", "sample_submission_8RXa3c6 (1).csv"]
    missing = [name for name in required if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input CSVs in {DATA_DIR}: {missing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = OUTPUT_DIR / "notebook_figures"
    figure_dir.mkdir(exist_ok=True)

    import matplotlib
    if not show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_counter = 0

    def _save_and_show_figures():
        nonlocal figure_counter
        for number in plt.get_fignums():
            figure_counter += 1
            figure = plt.figure(number)
            figure.savefig(figure_dir / f"figure_{figure_counter:02d}.png", dpi=130, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close("all")

    print(f"Inputs: {DATA_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")

    # %% [markdown] — Notebook cell 1
    # # Big Mart Sales Prediction — EDA, Cleaning & Modelling
    #
    # **Problem:** Predict `Item_Outlet_Sales` for products across 10 Big Mart stores (2013 data). This is a **regression** problem, scored by **RMSE** in the original sales units.
    #
    # This notebook follows the professor's six EDA steps in `LAB EDA  5.ipynb` and the Phase 1 plan in `README.md`. It adds data-quality checks, missingness patterns, train/test comparisons and analysis within outlet types. The approved Phase 2 modelling plan and Phase 3 conclusions follow the EDA, with original-scale RMSE throughout.
    #
    # **Hypotheses to examine**
    # - Higher-priced items (`Item_MRP`) have higher sales.
    # - Supermarkets have higher sales than grocery stores.
    # - Larger stores and Tier 1 locations have higher sales.
    # - More visible items have higher sales.
    #
    # These are associations to investigate, not causal claims. Product rows repeat across outlets; only 10 outlets are represented.
    #
    # **Reproducibility:** Run all cells from the project folder. Raw CSVs remain unchanged. Imputation values, visibility references, clipping limits and encoding categories are fitted on training features only. The reusable `BigMartPreprocessor` in `preprocessing.py` will also allow refitting inside each future cross-validation fold.

    # %% [markdown] — Notebook cell 2
    # ## Load & Inspect

    # %% — Notebook cell 3 (In[1])
    import json
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", palette="colorblind")
    _display_result(pd.set_option("display.max_columns", 20))

    # %% — Notebook cell 4 (In[2])
    df = pd.read_csv(DATA_DIR / "train_v9rqX0R.csv")
    test = pd.read_csv(DATA_DIR / "test_AbJTz2l.csv")
    # Keep both raw frames intact throughout the notebook.
    raw_train = df.copy(deep=True)
    raw_test = test.copy(deep=True)

    # %% — Notebook cell 5 (In[3])
    _display_result(df.head())

    # %% — Notebook cell 6 (In[4])
    _display_result(df.info())

    # %% — Notebook cell 7 (In[5])
    _display_result(df.describe())

    # %% — Notebook cell 8 (In[6])
    y = df["Item_Outlet_Sales"]
    X = df.drop(columns=["Item_Outlet_Sales"])

    print("Independent variables:", X.shape[1])
    print("Dependent variables:", 1)
    _display_result(print("Train shape:", df.shape, "| Test shape:", test.shape))

    # %% — Notebook cell 9 (In[7])
    _display_result(df.isnull().sum())

    # %% — Notebook cell 10 (In[8])
    _display_result(test.isnull().sum())

    # %% — Notebook cell 11 (In[9])
    _display_result(df.nunique())

    # %% [markdown] — Notebook cell 12
    # ### Additional audit — keys, valid ranges and consistency
    #
    # One observation should represent one item in one outlet. Count duplicates before deciding whether anything should be dropped. Inspect stable item/outlet metadata; do not silently deduplicate or overwrite contradictory values.

    # %% — Notebook cell 13 (In[10])
    assert list(df.drop(columns=TARGET).columns) == list(test.columns)
    assert TARGET not in test.columns
    key_cols = ["Item_Identifier", "Outlet_Identifier"]
    audit_rows = []
    for split, frame in [("train", df), ("test", test)]:
        audit_rows.append({
            "split": split, "rows": len(frame), "items": frame["Item_Identifier"].nunique(),
            "outlets": frame["Outlet_Identifier"].nunique(),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_item_outlet_keys": int(frame.duplicated(key_cols).sum()),
            "missing_keys": int(frame[key_cols].isna().any(axis=1).sum()),
            "nonpositive_observed_weight": int(frame["Item_Weight"].dropna().le(0).sum()),
            "nonpositive_mrp": int(frame["Item_MRP"].le(0).sum()),
            "visibility_outside_0_1": int((frame["Item_Visibility"].lt(0) | frame["Item_Visibility"].gt(1)).sum()),
            "year_outside_1_2013": int((frame["Outlet_Establishment_Year"].le(0) | frame["Outlet_Establishment_Year"].gt(2013)).sum()),
            "nonfinite_observed_numeric": int((~np.isfinite(frame.select_dtypes(include="number").stack().dropna().to_numpy())).sum()),
        })
    display(pd.DataFrame(audit_rows).set_index("split").T)
    print("Shared item/outlet keys across train and test:",
          len(pd.MultiIndex.from_frame(df[key_cols]).intersection(pd.MultiIndex.from_frame(test[key_cols]))))
    print("Nonpositive or missing sales:", int((df[TARGET].le(0) | df[TARGET].isna()).sum()))

    outlet_metadata = ["Outlet_Type", "Outlet_Size", "Outlet_Location_Type", "Outlet_Establishment_Year"]
    outlet_consistency = df.groupby("Outlet_Identifier")[outlet_metadata].nunique(dropna=True)
    print("Distinct observed metadata values per outlet (0 = all missing):")
    display(outlet_consistency)
    item_weight_counts = df.groupby("Item_Identifier")["Item_Weight"].nunique(dropna=True)
    print("Items with conflicting observed training weights:", int(item_weight_counts.gt(1).sum()))
    print("Items with no observed training weight:", int(item_weight_counts.eq(0).sum()))
    normalised_item_metadata = df.assign(Item_Fat_Content=df["Item_Fat_Content"].replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}))
    for col in ["Item_Type", "Item_Fat_Content"]:
        conflicts = normalised_item_metadata.groupby("Item_Identifier")[col].nunique().gt(1).sum()
        print(f"Items with conflicting {col} after label normalisation:", int(conflicts))
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission_8RXa3c6 (1).csv")
    assert list(sample_submission.columns) == key_cols + [TARGET]
    pd.testing.assert_frame_equal(sample_submission[key_cols], test[key_cols])
    _display_result(print("Sample submission identifiers match raw test order."))

    # %% [markdown] — Notebook cell 14
    # ### Additional EDA — train/test coverage and distribution comparison
    #
    # These are feature-only diagnostics. Numeric differences use the training standard deviation as a common scale; category total variation distance (TVD) is half the summed absolute proportion difference (0 = identical, 1 = disjoint). Neither is a formal test of future model accuracy. Missing category values are included explicitly.

    # %% — Notebook cell 15 (In[11])
    coverage = []
    for col in ["Item_Identifier", "Outlet_Identifier", "Item_Type", "Outlet_Type", "Item_Fat_Content"]:
        unseen = sorted(set(test[col].dropna()) - set(df[col].dropna()))
        coverage.append({"feature": col, "train_levels": df[col].nunique(),
                         "test_levels": test[col].nunique(), "unseen_test_levels": len(unseen),
                         "unseen_test_rows": int(test[col].isin(unseen).sum())})
    display(pd.DataFrame(coverage).set_index("feature"))
    shift_rows = []
    for col in ["Item_Weight", "Item_Visibility", "Item_MRP"]:
        a, b = df[col], test[col]
        shift_rows.append({"feature": col, "train_mean": a.mean(), "test_mean": b.mean(),
            "mean_shift_in_train_sd": (b.mean() - a.mean()) / a.std(),
            "train_median": a.median(), "test_median": b.median(),
            "train_p95": a.quantile(.95), "test_p95": b.quantile(.95),
            "train_missing_pct": a.isna().mean()*100, "test_missing_pct": b.isna().mean()*100})
    display(pd.DataFrame(shift_rows).set_index("feature").round(4))
    category_shift = []
    for col in ["Outlet_Identifier", "Outlet_Type", "Outlet_Size", "Outlet_Location_Type", "Item_Type", "Item_Fat_Content"]:
        proportions = pd.concat([
            df[col].fillna("Missing").value_counts(normalize=True).rename("train"),
            test[col].fillna("Missing").value_counts(normalize=True).rename("test")], axis=1).fillna(0)
        category_shift.append({"feature": col, "TVD": .5*(proportions["train"]-proportions["test"]).abs().sum(),
                               "max_gap_percentage_points": 100*(proportions["train"]-proportions["test"]).abs().max()})
    display(pd.DataFrame(category_shift).set_index("feature").sort_values("TVD", ascending=False).round(4))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ["Item_Weight", "Item_Visibility", "Item_MRP"]):
        bins = np.linspace(min(df[col].min(), test[col].min()), max(df[col].max(), test[col].max()), 31)
        for split, frame in [("train", df), ("test", test)]:
            sns.histplot(frame[col].dropna(), bins=bins, stat="density", element="step", fill=False, label=split, ax=ax)
        ax.set_title(col)
        ax.legend()
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 16
    # ### Additional EDA — missingness by outlet
    #
    # Missing values need not be random. Examine their source before choosing a fill strategy. Percentages below use each outlet's own row count; the zero-visibility percentage is separate from actual NaNs.

    # %% — Notebook cell 17 (In[12])
    missingness = df.assign(
        Weight_missing=df["Item_Weight"].isna(),
        Size_missing=df["Outlet_Size"].isna(),
        Visibility_zero=df["Item_Visibility"].eq(0),
    )
    missingness_summary = missingness.groupby("Outlet_Identifier").agg(
        Outlet_Type=("Outlet_Type", "first"), rows=(TARGET, "size"),
        Weight_missing_pct=("Weight_missing", "mean"),
        Size_missing_pct=("Size_missing", "mean"),
        Visibility_zero_pct=("Visibility_zero", "mean"),
    )
    rate_cols = ["Weight_missing_pct", "Size_missing_pct", "Visibility_zero_pct"]
    missingness_summary[rate_cols] *= 100
    display(missingness_summary.round(2))
    plt.figure(figsize=(8, 5))
    sns.heatmap(missingness_summary[rate_cols], annot=True, fmt=".1f", vmin=0, vmax=100,
                cmap="YlOrRd", cbar_kws={"label": "% of outlet rows"})
    plt.title("Training missingness and zero visibility by outlet")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 18
    # **Audit interpretation:** no exact duplicates or repeated item/outlet keys occur within either split. All test items and outlets are represented in training, but the item/outlet combinations are different. Missing weights are concentrated entirely in OUT019 and OUT027; missing sizes in OUT010, OUT017 and OUT045. This is structural missingness by outlet, so a global mean/mode alone would hide useful context. Distribution comparisons describe this particular split; they cannot rule out future shifts.

    # %% [markdown] — Notebook cell 19
    # # Step 1 — Univariate Analysis

    # %% [markdown] — Notebook cell 20
    # ### Categorical variables

    # %% — Notebook cell 21 (In[13])
    categorical_features = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size",
                            "Outlet_Location_Type", "Outlet_Type", "Outlet_Establishment_Year"]

    for col in categorical_features:
        print(f"----- {col} -----")
        print(df[col].value_counts(normalize=True, dropna=False).mul(100).round(2))
        print()

    # %% — Notebook cell 22 (In[14])
    cols = 2
    rows = (len(categorical_features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4.5), squeeze=False)

    for ax, col in zip(axes.flat, categorical_features):
        values = df[col].fillna("Missing").astype(str)
        sns.countplot(y=values, order=values.value_counts().index, ax=ax, color="skyblue")
        ax.set_title(f"Countplot - {col}")

    for ax in list(axes.flat)[len(categorical_features):]:
        ax.set_visible(False)

    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 23
    # ### Numerical variables

    # %% — Notebook cell 24 (In[15])
    numerical_features = ["Item_Weight", "Item_Visibility", "Item_MRP", "Item_Outlet_Sales"]

    df[numerical_features].hist(figsize=(12, 8), bins=30)
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% — Notebook cell 25 (In[16])
    sns.histplot(df["Item_Weight"], kde=True)
    plt.title("Distribution of Item_Weight")
    _save_and_show_figures()

    _display_result(print("Skewness:", round(df["Item_Weight"].skew(), 3)))

    # %% — Notebook cell 26 (In[17])
    sns.histplot(df["Item_Visibility"], kde=True)
    plt.title("Distribution of Item_Visibility")
    _save_and_show_figures()

    _display_result(print("Skewness:", round(df["Item_Visibility"].skew(), 3)))

    # %% — Notebook cell 27 (In[18])
    sns.histplot(df["Item_MRP"], kde=True)
    plt.title("Distribution of Item_MRP")
    _save_and_show_figures()

    _display_result(print("Skewness:", round(df["Item_MRP"].skew(), 3)))

    # %% — Notebook cell 28 (In[19])
    sns.histplot(df["Item_Outlet_Sales"], kde=True)
    plt.title("Distribution of Item_Outlet_Sales")
    _save_and_show_figures()

    _display_result(print("Skewness:", round(df["Item_Outlet_Sales"].skew(), 3)))

    # %% — Notebook cell 29 (In[20])
    plt.figure(figsize=(15, 4))
    for i, col in enumerate(numerical_features, start=1):
        plt.subplot(1, 4, i)
        sns.boxplot(y=df[col], color="skyblue",
                    flierprops={"marker": "o", "markerfacecolor": "red"})
        plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 30
    # **Observations — Univariate**
    #
    # - `Item_Fat_Content` has 5 labels for 2 values (including LF, low fat and reg), which are standardised during cleaning.
    # - `Outlet_Size` is missing in about 28% of rows; `Item_Weight` in about 17%. The outlet-level analysis below checks whether this is systematic.
    # - Weight (skew ≈0.08) and MRP (≈0.13) are roughly symmetric. MRP has **4 visible price groups**.
    # - Visibility (skew ≈1.17) and sales (≈1.18) are right-skewed. Zero visibility could be a missing/sentinel value; the data dictionary alone does not establish that zero is impossible. We retain an indicator and compare imputed/capped variants later.
    # - Supermarket Type1 contributes about 65% of rows. OUT027 has the most rows, but row count is not the same as sales volume.

    # %% [markdown] — Notebook cell 31
    # # Step 2 — Bivariate Analysis

    # %% [markdown] — Notebook cell 32
    # ### Numerical vs target

    # %% — Notebook cell 33 (In[21])
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="Item_MRP", y="Item_Outlet_Sales", alpha=0.4)
    plt.title("Item_MRP vs Item_Outlet_Sales")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% — Notebook cell 34 (In[22])
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="Item_Visibility", y="Item_Outlet_Sales", alpha=0.4)
    plt.title("Item_Visibility vs Item_Outlet_Sales")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% — Notebook cell 35 (In[23])
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="Item_Weight", y="Item_Outlet_Sales", alpha=0.4)
    plt.title("Item_Weight vs Item_Outlet_Sales")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% — Notebook cell 36 (In[24])
    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation coefficient"}
    )
    plt.title("Correlation Matrix Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 37
    # ### Categorical vs target

    # %% — Notebook cell 38 (In[25])
    for col in categorical_features:
        print(f"----- Mean Item_Outlet_Sales by {col} -----")
        print(df.groupby(col)["Item_Outlet_Sales"].mean().sort_values(ascending=False).round(1))
        print()

    # %% — Notebook cell 39 (In[26])
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4.5), squeeze=False)

    for ax, col in zip(axes.flat, categorical_features):
        values = df[col].fillna("Missing").astype(str)
        order = df.groupby(values)["Item_Outlet_Sales"].median().sort_values(ascending=False).index
        sns.boxplot(y=values, x=df["Item_Outlet_Sales"], order=order, ax=ax, color="skyblue")
        ax.set_title(f"Item_Outlet_Sales by {col}")

    for ax in list(axes.flat)[len(categorical_features):]:
        ax.set_visible(False)

    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 40
    # **Observations — Bivariate**
    #
    # - **MRP has the strongest numerical association** with sales (Pearson r≈0.57). This is partly expected because the target is a sales amount, not a count of units.
    # - Grocery stores average about 340 in sales, against about 2,000–3,700 for supermarket types. OUT027 (the only Supermarket Type3 outlet) averages about 3,694. Its store effect cannot be separated from its type using this sample.
    # - Visibility has a weak negative overall association (r≈−0.13). The analysis within outlet types below checks how much of this comes from mixing store types.
    # - Weight has almost no *marginal linear* association (r≈0.01); item-type means span about 1,900–2,400. This does not rule out interactions or predictive value after adjustment.
    # - Tier 1 and High-size stores do not have the highest overall means. Store mix and missing store sizes limit that comparison.
    # - Establishment year is also tied to store identity; 1998 represents the grocery outlet OUT010.

    # %% [markdown] — Notebook cell 41
    # ### Additional EDA — price and visibility within outlet types
    #
    # Compare correlations within each outlet type, and inspect both counts and mean sales across price bands. This checks whether overall relationships are driven by mixing outlet types. These are descriptive strata, not a full adjustment for all confounders.

    # %% — Notebook cell 42 (In[27])
    within_rows = []
    for label, group in [("All outlets", df), *list(df.groupby("Outlet_Type"))]:
        within_rows.append({"Outlet_Type": label, "n": len(group),
            "MRP_Pearson_r": group["Item_MRP"].corr(group[TARGET]),
            "Visibility_Pearson_r": group["Item_Visibility"].corr(group[TARGET]),
            "Visibility_Spearman_r": group["Item_Visibility"].corr(group[TARGET], method="spearman"),
            "Median_sales": group[TARGET].median(), "Mean_sales": group[TARGET].mean()})
    display(pd.DataFrame(within_rows).set_index("Outlet_Type").round(3))
    price_view = df.assign(MRP_Band=pd.cut(df["Item_MRP"], [0, 69, 136, 203, np.inf],
                                          labels=["Low", "Medium", "High", "Very High"]))
    price_means = price_view.pivot_table(index="Outlet_Type", columns="MRP_Band", values=TARGET, aggfunc="mean", observed=True)
    price_counts = price_view.pivot_table(index="Outlet_Type", columns="MRP_Band", values=TARGET, aggfunc="size", observed=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(price_means, annot=True, fmt=".0f", cmap="Blues", ax=axes[0], cbar_kws={"label": "Mean sales"})
    sns.heatmap(price_counts, annot=True, fmt="d", cmap="Greys", ax=axes[1], cbar_kws={"label": "Rows"})
    axes[0].set_title("Mean sales: outlet type × MRP band")
    axes[1].set_title("Sample size behind each mean")
    plt.tight_layout()
    _save_and_show_figures()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, feature in zip(axes, ["Item_MRP", "Item_Visibility"]):
        sns.scatterplot(data=df, x=feature, y=TARGET, hue="Outlet_Type", alpha=.25, s=12, ax=ax)
        ax.set_title(f"{feature} and sales by outlet type")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 43
    # **Within-type interpretation:** the raw visibility correlation changes from about −0.13 overall to between −0.013 and 0.017 within outlet types. MRP correlations remain positive (about 0.59–0.74). The overall negative visibility relationship is therefore largely explained by store-type composition, rather than a consistent relationship within types.
    #
    # ### Additional EDA — store mix behind location and size comparisons
    #
    # Display outlet counts as well as row-weighted sales. Thousands of product rows do not represent thousands of independent stores. Type2 and Type3 each have only one outlet, so their effects are inseparable from those outlet identities here.

    # %% — Notebook cell 44 (In[28])
    outlet_summary = df.groupby("Outlet_Identifier").agg(
        Outlet_Type=("Outlet_Type", "first"), Location=("Outlet_Location_Type", "first"),
        Size=("Outlet_Size", "first"), rows=(TARGET, "size"),
        mean_sales=(TARGET, "mean"), median_sales=(TARGET, "median"),
        p90_sales=(TARGET, lambda s: s.quantile(.9)),
    ).sort_values("mean_sales", ascending=False)
    display(outlet_summary.round(2))
    print("Number of distinct outlets by type:")
    display(df.groupby("Outlet_Type")["Outlet_Identifier"].nunique())
    print("Location × outlet-type mix (% of product rows in each location):")
    display(pd.crosstab(df["Outlet_Location_Type"], df["Outlet_Type"], normalize="index").mul(100).round(2))
    print("Observed-size × outlet-type mix (%; Missing is retained):")
    _display_result(display(pd.crosstab(df["Outlet_Size"].fillna("Missing"), df["Outlet_Type"], normalize="index").mul(100).round(2)))

    # %% [markdown] — Notebook cell 45
    # # Step 3 — Missing Value Treatment
    #
    # **Fit on training features; apply unchanged to train and test.** We do not learn imputation values from test rows. A combined frame is created only for displaying the already-cleaned splits.
    #
    # `preprocessing.py` contains the same cleaning and engineering used by the exports. Its sklearn-compatible transformer can be placed inside a pipeline in Phase 2. Full-training cleaned CSVs are convenient for inspection and final fitting, but must **not** be precomputed inputs to cross-validation: each fold must fit its own preprocessing on its training rows.
    #
    # Missingness flags preserve the original weight/size omissions and zero/missing visibility. They are candidate features; some are redundant with outlet identity.

    # %% — Notebook cell 46 (In[29])
    preprocessor = BigMartPreprocessor(reference_year=2013).fit(X)
    train_imputed = preprocessor.clean(X)
    test_imputed = preprocessor.clean(test)

    data = pd.concat([
        train_imputed.assign(Item_Outlet_Sales=y.to_numpy(), source="train"),
        test_imputed.assign(Item_Outlet_Sales=np.nan, source="test"),
    ], ignore_index=True)
    print("Display frame:", data.shape)
    _display_result(print("Fitted on:", len(X), "training rows only"))

    # %% [markdown] — Notebook cell 47
    # ### Item_Weight — same-item mean → item-type median → global median
    #
    # Observed training weights are constant within each item. Four training items have no observed training weight, so item-type and global fallbacks are needed even though the original combined-data treatment filled everything from item means. These fallbacks also cover unseen items during validation.

    # %% — Notebook cell 48 (In[30])
    weight_audit = []
    for split, raw, cleaned in [("train", X, train_imputed), ("test", test, test_imputed)]:
        missing = raw["Item_Weight"].isna()
        has_item_weight = raw["Item_Identifier"].map(preprocessor.weight_by_item_).notna()
        has_type_weight = raw["Item_Type"].map(preprocessor.weight_by_type_).notna()
        weight_audit.append({
            "split": split, "missing_before": int(missing.sum()),
            "item_mean_fills": int((missing & has_item_weight).sum()),
            "type_median_fills": int((missing & ~has_item_weight & has_type_weight).sum()),
            "global_median_fills": int((missing & ~has_item_weight & ~has_type_weight).sum()),
            "missing_after": int(cleaned["Item_Weight"].isna().sum()),
        })
    _display_result(display(pd.DataFrame(weight_audit).set_index("split")))

    # %% [markdown] — Notebook cell 49
    # ### Outlet_Size — known outlet size → mode by outlet type → global mode
    #
    # Use the training outlet's known size where available. For outlets with no observed size, the training type mode is a pragmatic assumption, not a recovered ground truth. In this dataset it fills OUT010, OUT017 and OUT045 with Small. Keep `Outlet_Size_Missing` to distinguish those guesses.

    # %% — Notebook cell 50 (In[31])
    _display_result(display(pd.crosstab(df["Outlet_Type"], df["Outlet_Size"].fillna("Missing"))))

    # %% — Notebook cell 51 (In[32])
    print("Training outlet-type modes:")
    display(preprocessor.size_by_type_.rename("Imputed_size"))
    for split, raw, cleaned in [("train", X, train_imputed), ("test", test, test_imputed)]:
        print(split, "missing size before:", raw["Outlet_Size"].isna().sum(),
              "| after:", cleaned["Outlet_Size"].isna().sum())

    # %% [markdown] — Notebook cell 52
    # ### Item_Visibility — flag zeros and use positive training observations
    #
    # Treat zero visibility as a missing/sentinel value **for this candidate treatment**. Fill using the item's positive training mean, then outlet-type positive median, then the global positive median. Keep the flag so Phase 2 can compare the assumption against keeping zeros. Positive observed visibility is unchanged.

    # %% — Notebook cell 53 (In[33])
    for split, raw, cleaned in [("train", X, train_imputed), ("test", test, test_imputed)]:
        print(split, "zeros before:", raw["Item_Visibility"].eq(0).sum(),
              "| nonpositive after:", cleaned["Item_Visibility"].le(0).sum(),
              "| missing after:", cleaned["Item_Visibility"].isna().sum())
        observed = raw["Item_Visibility"].gt(0)
        np.testing.assert_allclose(raw.loc[observed, "Item_Visibility"],
                                   cleaned.loc[observed, "Item_Visibility"])

    # %% — Notebook cell 54 (In[34])
    _display_result(display(pd.DataFrame({
        "train_missing_after": train_imputed.isna().sum(),
        "test_missing_after": test_imputed.isna().sum(),
    })))

    # %% [markdown] — Notebook cell 55
    # # Step 4 — Outlier Treatment
    #
    # Three detection methods, as in the lab: mean ± 3σ, IQR (1.5 × IQR) and the 99th percentile.

    # %% — Notebook cell 56 (In[35])
    outlier_cols = ["Item_Weight", "Item_Visibility", "Item_MRP", "Item_Outlet_Sales"]
    train_part = data[data["source"] == "train"]

    for col in outlier_cols:
        s = train_part[col]
        m, sd = np.mean(s), np.std(s)
        lower_3s, upper_3s = m - 3 * sd, m + 3 * sd
        Q1, Q3 = s.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_iqr, upper_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        p99 = np.percentile(s, 99)
        print(f"{col}:")
        print(f"   3σ bounds  = [{lower_3s:.3f}, {upper_3s:.3f}], outliers = {((s < lower_3s) | (s > upper_3s)).sum()}")
        print(f"   IQR bounds = [{lower_iqr:.3f}, {upper_iqr:.3f}], outliers = {((s < lower_iqr) | (s > upper_iqr)).sum()}")
        print(f"   99th pct   = {p99:.3f}, values above = {(s > p99).sum()}")

    # %% — Notebook cell 57 (In[36])
    plt.figure(figsize=(15, 4))
    for i, col in enumerate(outlier_cols, start=1):
        plt.subplot(1, 4, i)
        sns.boxplot(y=train_part[col], color="skyblue",
                    flierprops={"marker": "o", "markerfacecolor": "red"})
        plt.title(f"Box Plot - {col}")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 58
    # ### Item_Visibility — keep an additional capped variant using training IQR limits
    #
    # The imputed visibility column is retained. `Item_Visibility_iqr` is a candidate feature rather than a proven improvement. High shelf share may be legitimate in grocery stores; inspect which outlet types are affected before choosing the feature for a model.

    # %% — Notebook cell 59 (In[37])
    lower, upper = preprocessor.visibility_lower_, preprocessor.visibility_upper_
    data["Item_Visibility_iqr"] = data["Item_Visibility"].clip(lower, upper)
    print("Training-fitted visibility limits:", round(lower, 4), round(upper, 4))
    cap_audit = data.assign(Capped=data["Item_Visibility"].ne(data["Item_Visibility_iqr"]))
    _display_result(display(cap_audit.groupby(["source", "Outlet_Type"], observed=True)["Capped"]
            .agg(rows="size", capped="sum", fraction="mean").round(4)))

    # %% — Notebook cell 60 (In[38])
    train_part = data.loc[data["source"].eq("train")].copy()
    plt.figure(figsize=(7, 4))
    sns.histplot(data=train_part, x="Item_Visibility_iqr", kde=True)
    plt.title("Training visibility: IQR-capped candidate")
    plt.tight_layout()
    _save_and_show_figures()
    print("Training skewness before:", round(train_part["Item_Visibility"].skew(), 3))
    _display_result(print("Training skewness after :", round(train_part["Item_Visibility_iqr"].skew(), 3)))

    # %% — Notebook cell 61 (In[39])
    plt.figure(figsize=(7, 3))
    sns.boxplot(data=train_part, x="Item_Visibility_iqr")
    plt.title("Training visibility after IQR capping")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 62
    # ### Item_Outlet_Sales — flagged extremes are retained unchanged
    #
    # An outlier rule is a diagnostic, not evidence of a recording error. The high sales are positive and occur alongside plausible price/outlet values. With no evidence that they are erroneous, retain all rows and original sales values. Their squared errors matter directly to RMSE.
    #
    # Weight and MRP have no observations outside their 3σ/IQR bounds in this dataset; being above a 99th percentile alone does not justify changing a value. Target transformations below are exploratory candidates, not mandatory cleaning.
    #
    # **Visibility-cap review:** 135 of 144 capped training observations are grocery rows. Most flagged visibility values therefore belong to a distinct store segment, which is another reason to retain the uncapped candidate and test the choice with CV.

    # %% — Notebook cell 63 (In[40])
    sales_q1, sales_q3 = y.quantile([.25, .75])
    sales_iqr_upper = sales_q3 + 1.5 * (sales_q3-sales_q1)
    sales_tail = df.assign(High_sales_IQR=y.gt(sales_iqr_upper))
    display(sales_tail.groupby("Outlet_Type")["High_sales_IQR"]
            .agg(rows="size", flagged="sum", flagged_fraction="mean").round(4))
    print("Largest training sales rows (retained):")
    display(df.nlargest(10, TARGET)[ID_COLUMNS + ["Outlet_Type", "Item_MRP", TARGET]])
    _display_result(print("Share of total training sales in the top 1% of rows:",
          f"{y[y > y.quantile(.99)].sum()/y.sum():.2%}"))

    # %% [markdown] — Notebook cell 64
    # # Step 5 — Variable Creation
    #
    # Generate candidate features with the fitted helper so the EDA, CSV exports and later pipelines use the same definitions. The following cells inspect each feature. Predictive usefulness must still be checked with validation.

    # %% [markdown] — Notebook cell 65
    # ### Item_Fat_Content — merge inconsistent labels and mark non-consumables as Non-Edible

    # %% — Notebook cell 66 (In[41])
    print("Raw fat labels:", df["Item_Fat_Content"].unique())
    train_engineered = preprocessor.engineer(train_imputed)
    test_engineered = preprocessor.engineer(test_imputed)
    data = pd.concat([
        train_engineered.assign(Item_Outlet_Sales=y.to_numpy(), source="train"),
        test_engineered.assign(Item_Outlet_Sales=np.nan, source="test"),
    ], ignore_index=True)
    train_part = data.loc[data["source"].eq("train")].copy()
    print("Cleaned labels:", train_part["Item_Fat_Content"].unique())
    _display_result(display(train_part.groupby("Item_Fat_Content")[TARGET].agg(["count", "mean"]).round(2)))

    # %% [markdown] — Notebook cell 67
    # ### Item_Category — first two letters of Item_Identifier (FD / DR / NC)

    # %% — Notebook cell 68 (In[42])
    print("Training category percentages:")
    display(train_part["Item_Category"].value_counts(normalize=True).mul(100).round(2))
    _display_result(display(train_part.groupby("Item_Category")[TARGET].agg(["count", "mean"]).round(2)))

    # %% [markdown] — Notebook cell 69
    # ### Outlet_Years — store age in 2013

    # %% — Notebook cell 70 (In[43])
    display(train_part.groupby("Outlet_Years")[TARGET].agg(["count", "mean"]).round(2))
    _display_result(print("Age is measured in 2013, not in the current calendar year."))

    # %% [markdown] — Notebook cell 71
    # ### Item_Visibility_MeanRatio — visibility relative to a training-only item reference
    #
    # The denominator is the item's mean **imputed training visibility**, fixed during fit. Unknown items use the training outlet-type positive median, then the global median. Test rows cannot change this reference, even when transformed in a different batch.

    # %% — Notebook cell 72 (In[44])
    display(train_part["Item_Visibility_MeanRatio"].describe())
    ratio_correlations = []
    for label, group in [("All", train_part), *list(train_part.groupby("Outlet_Type"))]:
        ratio_correlations.append({"group": label, "n": len(group),
            "visibility_ratio_sales_r": group["Item_Visibility_MeanRatio"].corr(group[TARGET])})
    _display_result(display(pd.DataFrame(ratio_correlations).set_index("group").round(3)))

    # %% [markdown] — Notebook cell 73
    # ### MRP_Band — the MRP histogram shows 4 clear price groups

    # %% — Notebook cell 74 (In[45])
    display(train_part["MRP_Band"].value_counts(normalize=True).mul(100).round(2))
    _display_result(display(train_part.groupby("MRP_Band", observed=True)[TARGET].agg(["count", "mean"]).round(2)))

    # %% — Notebook cell 75 (In[46])
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Item_MRP"], bins=80)
    for b in [69, 136, 203]:
        plt.axvline(b, color="red", linestyle="--")
    plt.title("Training MRP with descriptive band cut-offs")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 76
    # ### Is_Grocery — flag for grocery stores (like `zn1` in the lab)

    # %% — Notebook cell 77 (In[47])
    _display_result(display(train_part.groupby("Is_Grocery")[TARGET].agg(["count", "mean"]).round(2)))

    # %% [markdown] — Notebook cell 78
    # **Observations — Variable Creation**
    #
    # - MRP bands separate training mean sales: about 736 → 1,616 → 2,631 → 3,681. Cut-offs 69/136/203 are descriptive candidates; retain continuous MRP too until feature selection.
    # - Grocery stores average about 340 versus 2,449 elsewhere. `Is_Grocery` duplicates the grocery one-hot column; keep one of them for the final feature set.
    # - The visibility ratio's overall association is much stronger than its associations within supermarket types. It largely captures store mix; it is not evidence that reducing shelf visibility raises sales.
    # - Age, location, size and the new missingness flags overlap with outlet identity. The final redundancy check makes this explicit before modelling.

    # %% [markdown] — Notebook cell 79
    # # Step 6 — Variable Transformation

    # %% [markdown] — Notebook cell 80
    # ### Item_Outlet_Sales — compare raw, log and square root without exporting target-derived predictors
    #
    # A regression target does not need a normal marginal distribution. A more symmetric histogram is not proof of better original-scale RMSE. In Phase 2, any target transformation must be inverted before scoring (e.g. with `TransformedTargetRegressor`), with any bias correction learned within training folds. Select it using validation, not skewness alone.

    # %% — Notebook cell 81 (In[48])
    target_views = pd.DataFrame({
        TARGET: y,
        "Item_Outlet_Sales_log": np.log1p(y),
        "Item_Outlet_Sales_sqrt": np.sqrt(y),
    })
    display(target_views.head())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, target_views):
        sns.histplot(target_views[col], kde=True, ax=ax)
        ax.set_title(col.replace("Item_Outlet_Sales", "Sales"))
    plt.tight_layout()
    _save_and_show_figures()
    print("Training target skewness:")
    _display_result(display(target_views.skew().round(3)))

    # %% [markdown] — Notebook cell 82
    # ### Item_Visibility — square-root and log transforms

    # %% — Notebook cell 83 (In[49])
    visibility_views = train_part[["Item_Visibility", "Item_Visibility_iqr",
                                   "Item_Visibility_sqrt", "Item_Visibility_log"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Item_Visibility", "Item_Visibility_sqrt", "Item_Visibility_log"]):
        sns.histplot(visibility_views[col], kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    _save_and_show_figures()
    _display_result(display(visibility_views.skew().rename("Training_skewness").round(3)))

    # %% [markdown] — Notebook cell 84
    # ### Encoding categorical variables
    #
    # Size and MRP bands have ordered codes. Location tiers use the existing ordered coding as a candidate; for Linear Regression, KNN and SVR, compare one-hot location/size because codes impose equal spacing. Nominal features use training-fitted one-hot categories; unseen nominal levels produce an all-zero block.
    #
    # All one-hot levels and related features are retained in the EDA export for comparison. Select redundant columns inside the future modelling pipeline. Scaling for KNN/SVR must also be fitted within each CV fold.

    # %% — Notebook cell 85 (In[50])
    train_encoded = preprocessor.transform_frame(X)
    test_encoded = preprocessor.transform_frame(test)
    feature_cols = preprocessor.get_feature_names_out().tolist()
    print("Numeric predictors:", len(feature_cols))
    print("Identifiers retained for export:", ID_COLUMNS)
    _display_result(display(train_encoded.head()))

    # %% [markdown] — Notebook cell 86
    # ### Validate and save the cleaned splits
    #
    # Exports retain both identifiers in original row order. `train_clean.csv` contains the original target **only**; log/sqrt target views remain separate above. `feature_columns.json` lists exactly the numeric predictors, excluding identifiers and every target-derived column.

    # %% — Notebook cell 87 (In[51])
    train_clean = train_encoded.copy()
    train_clean[TARGET] = y.to_numpy()
    test_clean = test_encoded.copy()

    assert len(train_clean) == len(df) and len(test_clean) == len(test)
    assert list(train_encoded.columns) == list(test_encoded.columns)
    assert train_clean.isna().sum().sum() == 0 and test_clean.isna().sum().sum() == 0
    assert np.isfinite(train_clean[feature_cols].to_numpy(dtype=float)).all()
    assert np.isfinite(test_clean[feature_cols].to_numpy(dtype=float)).all()
    assert not any("Item_Outlet_Sales" in col for col in feature_cols)
    assert set(feature_cols).isdisjoint(ID_COLUMNS)
    assert set(train_clean.columns) - set(test_clean.columns) == {TARGET}
    assert train_clean["Item_Visibility"].gt(0).all() and test_clean["Item_Visibility"].gt(0).all()
    pd.testing.assert_frame_equal(train_clean[ID_COLUMNS], df[ID_COLUMNS], check_dtype=False)
    pd.testing.assert_frame_equal(test_clean[ID_COLUMNS], test[ID_COLUMNS], check_dtype=False)
    np.testing.assert_array_equal(train_clean[TARGET].to_numpy(), y.to_numpy())
    pd.testing.assert_frame_equal(df, raw_train)
    pd.testing.assert_frame_equal(test, raw_test)
    # Both notebook walk-through and pipeline use the same feature definitions.
    pd.testing.assert_frame_equal(preprocessor.transform(X), train_clean[feature_cols])

    train_clean.to_csv(OUTPUT_DIR / "train_clean.csv", index=False)
    test_clean.to_csv(OUTPUT_DIR / "test_clean.csv", index=False)
    (OUTPUT_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2) + "\n")
    print("All export checks passed.")
    print("Train:", train_clean.shape, "| Test:", test_clean.shape)
    print("Training target unchanged; no target transforms in predictors.")
    _display_result(print("Original train/test order and both raw input frames preserved."))

    # %% [markdown] — Notebook cell 88
    # ### Final correlation check — engineered numeric features and sales
    #
    # This heatmap is descriptive. Correlation with the target is not used to fit cleaning values or encode categories.

    # %% — Notebook cell 89 (In[52])
    check_cols = ["Item_MRP", "MRP_Band_code", "Is_Grocery", "Outlet_Years", "Outlet_Size_code",
                  "Outlet_Location_code", "Item_Weight", "Item_Visibility",
                  "Item_Visibility_MeanRatio", TARGET]
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_clean[check_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
    plt.title("Training correlations after cleaning & feature creation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 90
    # ### Additional EDA — redundant candidate features
    #
    # Strong correlations and exact duplicate columns are flags for model-specific feature selection. Full one-hot blocks, an intercept, numeric variants and outlet metadata can be redundant together. Linear Regression can fit collinear inputs, but coefficients become difficult to interpret; repeated representations also change distances for KNN and SVR.

    # %% — Notebook cell 91 (In[53])
    predictor_corr = train_clean[feature_cols].corr().abs()
    upper_triangle = predictor_corr.where(np.triu(np.ones(predictor_corr.shape), k=1).astype(bool))
    strong_pairs = upper_triangle.stack().sort_values(ascending=False)
    display(strong_pairs[strong_pairs.ge(.85)].head(20).rename("Absolute Pearson r").to_frame())
    exact_duplicates = []
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i+1:]:
            if np.array_equal(train_clean[a].to_numpy(), train_clean[b].to_numpy()):
                exact_duplicates.append((a, b))
    print("Exact duplicate predictor columns:", exact_duplicates)
    print("Candidate selection: keep one visibility representation for distance-based models;")
    _display_result(print("compare continuous MRP versus additional bands; avoid duplicate grocery flags."))

    # %% [markdown] — Notebook cell 92
    # ## Summary of EDA & Cleaning — Phase 1 checkpoint
    #
    # | Step | What was done | Interpretation / next decision |
    # |---|---|---|
    # | Quality audit | Schema, duplicate rows and item/outlet keys, ranges, metadata consistency and submission order | Raw rows retained; zero visibility handled as an explicit assumption |
    # | 1. Univariate | Frequencies, histograms, skewness and boxplots | Visibility/sales are right-skewed; fat labels need merging; MRP has 4 price groups |
    # | 2. Bivariate | Scatter, Pearson/Spearman correlations, group counts/means/medians, price × outlet and within-type comparisons | Price and store mix dominate descriptive associations; avoid causal conclusions |
    # | Train/test check | Missingness, numeric mean/quantile comparisons, category proportions and unseen items/levels | Diagnose covariate differences without using test sales or fitting statistics on test |
    # | 3. Missing values | Training item mean/type-median fallback for weight; outlet/type mode for size; positive item visibility mean with fallbacks; 3 flags | Complete numeric features; structural outlet missingness remains visible |
    # | 4. Outliers | 3σ, IQR and 99th percentile diagnostics; training-fitted capped visibility variant | Preserve original sales and positive observed visibility; choose capping with CV |
    # | 5. New variables | Normalised fat labels, category, age in 2013, fixed-reference visibility ratio, MRP band and grocery flag | Candidate signals overlap; prune redundant representations in model-specific pipelines |
    # | 6. Transform/encode | Compare log/sqrt; training-fitted category schema | sqrt sales has skew ≈0.24, but original-scale RMSE determines transformation choice |
    # | Export | Rebuilt train/test CSVs and explicit predictor list | No target-derived predictors; row order, target values, schema and finite values asserted |
    #
    # **Modelling transition:** the approved model-selection and building phase is implemented below. Start from raw features with fold-fitted preprocessing. Compare the four course-approved regressors, freeze the winner using development CV, then inspect the reserved modelling check. The full-training EDA exports are not cross-validation inputs.
    #
    # **Limitations:** only 10 outlets and one historical cross-section; outlet type, age, size and location are confounded with store identity. Training EDA has informed candidate features, so an untouched final holdout or nested CV would provide a stronger final performance estimate. Associations are not effects of interventions.
    #
    # Saved: `train_clean.csv`, `test_clean.csv`, `feature_columns.json`. Target transformations are held only in `target_views`; identifiers are retained in CSVs for traceability and excluded by `feature_cols` / `preprocessor.transform()`.

    # %% [markdown] — Notebook cell 93
    # # Phase 2 — Model Selection & Building
    #
    # **Approved approach:** extend this notebook; use an 80% development set with the same five CV folds for every candidate, a reserved 20% modelling check, and balanced, bounded searches. Compare Linear Regression, scaled KNN, scaled RBF SVR and Random Forest. A training-mean predictor is the reference baseline.
    #
    # All model choices are made using development CV. The reserved check is used only after the selected configuration is frozen. Prior EDA used all training rows, so this is a **reserved modelling check**, not a completely untouched final assessment. Prediction concerns item/outlet combinations in existing stores; it does not establish generalisation to new stores.
    #
    # Reusable code is in `modeling.py`. Saved results use a fingerprint of raw training data, both Python modules and package versions, so unchanged searches can be inspected without retraining. Changed inputs/code invalidate the cache.

    # %% — Notebook cell 94 (In[54])
    import joblib

    model_train = pd.read_csv(DATA_DIR / "train_v9rqX0R.csv")
    model_test = pd.read_csv(DATA_DIR / "test_AbJTz2l.csv")
    model_sample = pd.read_csv(DATA_DIR / "sample_submission_8RXa3c6 (1).csv")
    model_output_dir = OUTPUT_DIR / "model_outputs"
    model_output_dir.mkdir(exist_ok=True)
    (model_output_dir / "figures").mkdir(exist_ok=True)
    _display_result(print("Modelling uses raw inputs:", model_train.shape, model_test.shape))

    # %% [markdown] — Notebook cell 95
    # ## 1. Fixed development / reserved-check split
    #
    # Use seed 42 and stratify the 80/20 split by outlet identity to represent all ten stores on both sides. CV is shuffled, seeded five-fold KFold on development rows only. Every development row appears in exactly one validation fold; reserved-check rows appear in none. Split assignments are saved with original row positions.

    # %% — Notebook cell 96 (In[55])
    dev_idx, check_idx, model_folds, split_assignments = development_split(model_train)
    split_assignments.to_csv(model_output_dir / "split_assignments.csv", index=False)
    display(split_assignments.groupby("partition").size().rename("rows"))
    display(split_assignments.loc[split_assignments.partition.eq("development")]
            .groupby("cv_fold").size().rename("development_validation_rows"))
    assert not set(dev_idx).intersection(check_idx)
    assert len(dev_idx) == 6818 and len(check_idx) == 1705
    assert split_assignments.loc[check_idx, "cv_fold"].isna().all()
    _display_result(print("All candidate models reuse these exact folds."))

    # %% [markdown] — Notebook cell 97
    # ## 2. Fold-fitted pipelines and feature candidates
    #
    # Each estimator receives raw inputs. Its pipeline fits **BigMartPreprocessor → ModelFeatures → numerical scaling (except Random Forest) → regressor** inside each fold. Binary indicators remain 0/1. A serialisable `SalesRegressor` wrapper transforms training targets, inverts predictions and floors sales at zero consistently in CV and final inference.
    #
    # | Profile | Inputs |
    # |---|---|
    # | Compact | MRP, weight, one visibility representation, zero/missing-visibility flag, Regular-fat flag, item-type and outlet-identity one-hot columns |
    # | Descriptive ordered | Same item features; replace outlet identity with outlet type, age, ordered size/location, their missingness flags and MRP-band code |
    # | Descriptive one-hot | Descriptive profile with one-hot size/location |
    # | Price interactions (Linear Regression only) | Compact plus MRP × non-reference outlet-type interactions, allowing different price slopes by type |
    #
    # One reference level is dropped from selected one-hot blocks. Raw IDs, target-derived columns, duplicate fat/category columns and duplicate grocery flags are excluded. Outlet identity remains available as **encoded categories** in the compact profile. Correlated store metadata in descriptive profiles can still limit coefficient interpretation.
    #
    # KNN/SVR/Random Forest searches compare imputed raw, imputed square-root, imputed capped and zero-preserving raw visibility. Only Random Forest additionally tests the visibility ratio. Ratios use positive training-only references, including for zero-only or unseen items. All families compare raw, square-root and log1p targets; no bias correction is assumed from a histogram alone.

    # %% — Notebook cell 98 (In[56])
    # Preview feature definitions on development FEATURES only; CV fits fresh copies.
    preview_preprocessor = BigMartPreprocessor().fit(model_train.iloc[dev_idx][RAW_FEATURES])
    preview_encoded = preview_preprocessor.transform(model_train.iloc[dev_idx][RAW_FEATURES])
    profile_preview = []
    for profile in PROFILES + ["price_interactions"]:
        feature_builder = ModelFeatures(profile=profile).fit(preview_encoded)
        names = feature_builder.get_feature_names_out().tolist()
        profile_preview.append({"profile": profile, "predictors": len(names),
                                "raw_IDs_or_target": bool(set(ID_COLUMNS + [TARGET]).intersection(names))})
    _display_result(display(pd.DataFrame(profile_preview)))

    # %% [markdown] — Notebook cell 99
    # ## 3. Baselines and bounded hyperparameter searches
    #
    # First compare the training-mean predictor and all four approved models using compact features, imputed raw visibility and raw sales. Starting settings: KNN k=31/uniform/Euclidean; SVR C=1000/gamma=scale/epsilon=100; Random Forest 400 trees/depth=6/minimum leaf=100/max_features=1.0.
    #
    # Then tune on the same folds. Retain an original baseline if it beats that family's searched candidates. Use four concurrent search workers with single-threaded estimators/numerical libraries.
    #
    # | Model | Search |
    # |---|---|
    # | Linear Regression | 12 configurations: four feature profiles × three target modes |
    # | KNN | 16 sampled candidates; k∈{5,11,21,31,51,81}, uniform/distance, Manhattan/Euclidean |
    # | RBF SVR | 20 sampled candidates; C∈{10,100,1000,10000}, gamma∈{scale,.001,.01,.1}, epsilon∈{.01,.1,1,10,100} |
    # | Random Forest | 24 sampled candidates; trees∈{200,400,600}, depth∈{4,6,8,10,None}, leaf∈{1,5,20,50,100}, max_features∈{.5,.75,1} |
    #
    # The three non-interaction profiles, four linked visibility treatments and three target modes are sampled jointly with KNN/SVR/forest hyperparameters. These bounded searches are not exhaustive. `error_score="raise"` makes failed candidates visible. RMSE is always computed after target inversion and the zero floor.

    # %% — Notebook cell 100 (In[57])
    selection = run_selection(model_train, output_dir=model_output_dir, n_jobs=4, reuse_results=True)
    baseline_table = pd.DataFrame(selection["baselines"])[
        ["model", "cv_rmse", "cv_std", "train_rmse", "fit_seconds", "predict_seconds"]]
    _display_result(display(baseline_table.round(3)))

    # %% — Notebook cell 101 (In[58])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(baseline_table.model, baseline_table.cv_rmse, yerr=baseline_table.cv_std,
           capsize=4, color="skyblue")
    ax.set_ylabel("Mean development CV RMSE ± fold SD")
    ax.set_title("Starting model comparison: original sales scale")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(model_output_dir / "figures" / "baseline_comparison.png", dpi=160, bbox_inches="tight")
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 102
    # ## 4. Model selection using development CV only
    #
    # The following table contains each family's best eligible baseline/search configuration. Lower RMSE is better; fold SD describes variability across these folds, not a confidence interval. Training-versus-validation error helps identify overfitting. Mean score time includes prediction and metric computation.
    #
    # Search results are saved per family, including every fold score and parameter set. Selection uses mean CV RMSE; exact ties use mean prediction/scoring time, then the model order above. Search-best CV scores can be optimistic after tuning, which is why the reserved check is evaluated after selection is frozen.

    # %% — Notebook cell 103 (In[59])
    model_comparison = comparison_table(selection)
    display(model_comparison.round(3))
    search_summary = pd.DataFrame([
        {"model": name, "searched_candidates": len(results),
         "best_searched_cv_rmse": -results.mean_test_score.max()}
        for name, results in selection["search_results"].items()
    ])
    display(search_summary.round(3))
    for result in selection["finalists"]:
        if result["model"] != "MeanBaseline":
            print(result["model"], "eligible configuration:")
            print(json.dumps(result["parameters"], indent=2))

    # %% — Notebook cell 104 (In[60])
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(MODEL_ORDER))
    starting = baseline_table.set_index("model").loc[MODEL_ORDER]
    best = model_comparison.set_index("model").loc[MODEL_ORDER]
    ax.bar(positions-.18, starting.cv_rmse, .36, yerr=starting.cv_std, capsize=3, label="Starting baseline")
    ax.bar(positions+.18, best.cv_rmse, .36, yerr=best.cv_std, capsize=3, label="Best eligible configuration")
    ax.set_xticks(positions, MODEL_ORDER)
    ax.set_ylabel("Mean development CV RMSE ± fold SD")
    ax.set_title("Effect of bounded tuning and EDA-driven feature choices")
    ax.legend()
    plt.tight_layout()
    fig.savefig(model_output_dir / "figures" / "tuned_comparison.png", dpi=160, bbox_inches="tight")
    _display_result(_save_and_show_figures())

    # %% — Notebook cell 105 (In[61])
    frozen_winner = selection["winner"]
    print("Frozen winner:", frozen_winner["model"], "|", frozen_winner["stage"])
    print("Mean CV RMSE:", round(frozen_winner["cv_rmse"], 3))
    print("Profile:", frozen_winner["profile"], "| target:", frozen_winner["target_mode"])
    print("Visibility:", frozen_winner["visibility_policy"], frozen_winner["visibility"])
    print("Parameters:", json.dumps(frozen_winner["parameters"], indent=2))
    assert frozen_winner["cv_rmse"] < selection["baselines"][0]["cv_rmse"]
    assert (model_output_dir / "frozen_selection.json").exists()
    _display_result(print("Selection was frozen before reserved-check metrics were computed."))

    # %% [markdown] — Notebook cell 106
    # ## 5. Frozen winner: OOF predictions and reserved modelling check
    #
    # Generate development out-of-fold predictions using the frozen configuration. Separately fit that configuration on all development rows and assess the reserved check once, alongside a development-fitted mean predictor. Do not change the winner or hyperparameters using these check results.
    #
    # Pooled OOF RMSE and mean fold RMSE need not be identical: one squares errors across all OOF rows, while the other averages five separately computed RMSE values. OOF results also reuse the folds used for tuning, so they are development diagnostics rather than an independent final estimate.

    # %% — Notebook cell 107 (In[62])
    evaluation = evaluate_selected(model_train, selection, output_dir=model_output_dir,
                                   n_jobs=4, reuse_results=True)
    evaluation_table = pd.DataFrame(evaluation["metrics"]).T
    print("All metrics use original sales units (R² is dimensionless).")
    _display_result(display(evaluation_table.round(3)))

    # %% [markdown] — Notebook cell 108
    # ## 6. Residual diagnostics
    #
    # Residual = actual sales − predicted sales. Positive values indicate underprediction; negative values indicate overprediction. Inspect accuracy, bias, changing error spread and the reserved-check gap against the mean baseline. These diagnostics explain limitations; they are not used to retune the frozen model.

    # %% — Notebook cell 109 (In[63])
    check_predictions = evaluation["check_predictions"]
    oof_predictions = evaluation["oof_predictions"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.scatter(check_predictions[TARGET], check_predictions.prediction, s=14, alpha=.35)
    line_max = max(check_predictions[TARGET].max(), check_predictions.prediction.max())
    ax.plot([0, line_max], [0, line_max], "r--")
    ax.set(xlabel="Actual sales", ylabel="Predicted sales", title="Reserved check: actual vs predicted")
    ax = axes[0, 1]
    ax.scatter(check_predictions.prediction, check_predictions.residual, s=14, alpha=.35)
    ax.axhline(0, color="red", linestyle="--")
    ax.set(xlabel="Predicted sales", ylabel="Actual − predicted", title="Reserved check: residuals")
    ax = axes[1, 0]
    residual_bins = np.linspace(min(oof_predictions.residual.min(), check_predictions.residual.min()),
                                max(oof_predictions.residual.max(), check_predictions.residual.max()), 45)
    for label, frame in [("Development OOF", oof_predictions), ("Reserved check", check_predictions)]:
        ax.hist(frame.residual, bins=residual_bins, density=True, alpha=.4, label=label)
    ax.axvline(0, color="red", linestyle="--")
    ax.set(xlabel="Residual", ylabel="Density", title="Residual distributions")
    ax.legend()
    ax = axes[1, 1]
    labels = ["Development OOF", "Reserved check", "Reserved mean baseline"]
    values = [evaluation["metrics"][k]["rmse"] for k in
              ["development_oof", "reserved_check", "reserved_mean_baseline"]]
    ax.bar(labels, values, color=["steelblue", "seagreen", "grey"])
    ax.set(ylabel="RMSE", title="Development and reserved-check error")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(model_output_dir / "figures" / "residual_diagnostics.png", dpi=160, bbox_inches="tight")
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 110
    # ### Errors by outlet, type, price and high-sales segment
    #
    # Group metrics show where average performance hides difficult cases. The high-sales cut-off is the **development target's 90th percentile**, also applied to the check set. Group R² can be low or negative despite useful global predictions; small group counts require caution. Both OOF and check summaries are saved, with check tables displayed here.

    # %% — Notebook cell 111 (In[64])
    check_group_tables = {}
    for group_column in ["Outlet_Identifier", "Outlet_Type", "MRP_Band", "Sales_segment"]:
        print("Reserved check by", group_column)
        check_group_tables[group_column] = group_errors(check_predictions, group_column)
        display(check_group_tables[group_column].round(3))
    _display_result(print("Development p90 sales threshold:", round(evaluation["development_tail_cutoff"], 3)))

    # %% — Notebook cell 112 (In[65])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key in zip(axes, ["Outlet_Identifier", "Outlet_Type"]):
        group_table = check_group_tables[key].sort_values("rmse")
        sns.barplot(data=group_table, x="rmse", y=key, color="skyblue", ax=ax)
        ax.set_title("Reserved-check RMSE by " + key)
        ax.set_xlabel("RMSE in original sales units")
    plt.tight_layout()
    fig.savefig(model_output_dir / "figures" / "outlet_errors.png", dpi=160, bbox_inches="tight")
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 113
    # ## 7. Permutation importance after model selection
    #
    # Shuffle one raw feature at a time on the reserved check and measure the increase in RMSE, using ten repeats and seed 42. This is a diagnostic for the frozen development-fitted model, not another feature-selection stage. Positive values mean prediction became worse after shuffling; near-zero or negative values indicate little marginal benefit in this particular model/sample.
    #
    # Store identity, type, size, age and location are correlated. Permutation can create unusual combinations of these fields, and correlated features can share importance. Interpret these scores as predictive diagnostics, not causal effects or independent business levers. Raw item identifiers can affect cleaning/category derivation even though they are excluded as direct numeric predictors.

    # %% — Notebook cell 114 (In[66])
    importance_table = evaluation["importance"]
    display(importance_table.round(3))
    importance_plot = importance_table.sort_values("rmse_increase_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_plot.feature, importance_plot.rmse_increase_mean,
            xerr=importance_plot.rmse_increase_std, capsize=3, color="skyblue")
    ax.axvline(0, color="grey", linewidth=1)
    ax.set(xlabel="Increase in original-scale RMSE after permutation (mean ± SD)",
           title="Frozen model: reserved-check permutation importance")
    plt.tight_layout()
    fig.savefig(model_output_dir / "figures" / "permutation_importance.png", dpi=160, bbox_inches="tight")
    _display_result(_save_and_show_figures())

    # %% [markdown] — Notebook cell 115
    # ## 8. Final refit and validated submission
    #
    # Keep the selected configuration unchanged. Fit it on all 8,523 labelled rows, then predict the 5,681 raw test rows. The full fitted pipeline includes cleaning, category discovery, selected features, scaling, target inversion and the zero floor.
    #
    # Save the complete estimator, verify a save/load prediction round-trip, and validate the submission against the sample's identifiers and column order. Record model settings, selected features, package versions, input hashes, CV/check metrics and run sizes. The local scores are not a public leaderboard result. Upload remains a manual step.

    # %% — Notebook cell 116 (In[67])
    run_metadata = build_final(model_train, model_test, model_sample, selection, evaluation,
                               output_dir=model_output_dir, submission_path=OUTPUT_DIR / "submission.csv")
    print("Final estimator:", model_output_dir / "final_model.joblib")
    print("Selected predictors:", run_metadata["selected_feature_count"])
    print("Final training rows:", run_metadata["final_fit_rows"])
    _display_result(print("Submission rows:", run_metadata["test_rows"]))

    # %% — Notebook cell 117 (In[68])
    submission = pd.read_csv(OUTPUT_DIR / "submission.csv")
    validate_submission(submission, model_test, model_sample)
    assert submission.shape == (5681, 3)
    display(submission.head())
    display(submission[TARGET].describe())
    print("Submission checks passed: exact columns, original test identifiers/order, finite nonnegative sales.")
    print("Saved metadata:", model_output_dir / "run_metadata.json")
    _display_result(print("Final predictor names:", run_metadata["selected_features"]))

    # %% [markdown] — Notebook cell 118
    # # Phase 3 — Conclusions, Business Interpretation & Limitations
    #
    # The following report is generated from the saved experiment results, rather than assuming Random Forest must win. It distinguishes development selection, the reserved modelling check and the final all-training refit. All observations about drivers are associations in this historical sample.

    # %% — Notebook cell 119 (In[69])
    winner = selection["winner"]
    check_metrics = evaluation["metrics"]["reserved_check"]
    mean_check_rmse = evaluation["metrics"]["reserved_mean_baseline"]["rmse"]
    relative_improvement = 100 * (1-check_metrics["rmse"]/mean_check_rmse)
    ranked_models = model_comparison.loc[model_comparison.model.ne("MeanBaseline")]
    ranking_text = "; ".join(f"{r.model}: {r.cv_rmse:,.2f}" for r in ranked_models.itertuples())
    main_features = importance_table.head(3).feature.tolist()
    tail_table = check_group_tables["Sales_segment"].set_index("Sales_segment")
    tail_rmse = tail_table.loc["Above development p90", "rmse"]
    tail_bias = tail_table.loc["Above development p90", "mean_residual"]
    bias_description = "underprediction" if tail_bias > 0 else "overprediction"
    report = f"""
    **Selected model:** {winner['model']} ({winner['stage']}), using `{winner['profile']}` features,
    `{winner['visibility_policy']}` / `{winner['visibility']}` visibility and a `{winner['target_mode']}` target.

    **Development comparison — mean five-fold RMSE:** {ranking_text}.
    The frozen winner's mean CV RMSE is **{winner['cv_rmse']:,.2f} ± {winner['cv_std']:,.2f}** (fold SD).
    Its pooled development OOF RMSE is **{evaluation['metrics']['development_oof']['rmse']:,.2f}**.

    **Reserved modelling check:** RMSE **{check_metrics['rmse']:,.2f}**, MAE **{check_metrics['mae']:,.2f}**,
    R² **{check_metrics['r2']:.3f}**. The reserved mean-baseline RMSE is **{mean_check_rmse:,.2f}**,
    so the selected model changes RMSE by **{relative_improvement:.1f}%** relative to that baseline.
    The model was not changed after inspecting these results.

    **Difficult cases:** above-development-p90 sales have reserved-check RMSE **{tail_rmse:,.2f}**,
    with mean residual **{tail_bias:,.2f}**, indicating average {bias_description} in that segment.
    A global RMSE alone therefore does not describe all operating conditions.

    **Predictive drivers:** the largest three mean permutation scores are for
    **{', '.join(main_features)}**. Interpret them alongside the EDA and correlated-store limitations;
    permutation rankings are not causal explanations.

    **Business use:** use store-specific and price-segment forecasts to support inventory planning,
    with extra review for high-sales items where errors are larger. Compare forecast errors within stores
    before reallocating inventory. The analysis does not show that changing price, visibility, location
    or store size would itself cause a sales increase.

    **Limits and next evaluation:** only ten stores and one 2013 cross-section are available; product and
    store identities overlap across development/test. Full-data EDA preceded the modelling split, and
    model selection reused development CV folds. A new labelled dataset, nested evaluation or a separate
    new-store study would strengthen generalisation claims. There is no time series here for validating
    future years. The approximate README leaderboard benchmark has not been verified by an upload.

    **Delivered:** all-training fitted estimator, reproducible split/search results, OOF/check diagnostics,
    run metadata and a validated **5,681-row `submission.csv`**. Public leaderboard evaluation remains manual.
    """
    _display_result(display(Markdown(report)))

    # %% [markdown] — Notebook cell 120
    # ### Reproduce or inspect this phase
    #
    # - Run all notebook cells from the project folder, or run `python3 -m modeling` for the modelling artifacts.
    # - Existing results are reused only when their raw-data/code/version fingerprint matches. For a deliberate new experiment, call `run_selection(..., reuse_results=False)`; do not repeatedly tune against the reserved check.
    # - Run `python3 -m unittest -v test_preprocessing.py test_modeling.py` for the preprocessing, split, pipeline, target-transform, serialization and submission checks.
    # - `model_outputs/` contains per-family search tables, split assignments, the frozen selection, model comparison, predictions, grouped errors, permutation importance, model artifacts and run metadata. `submission.csv` is in the project root.

    print(f"Completed {NOTEBOOK_CODE_CELL_COUNT} notebook code cells.")
    print(f"Saved {figure_counter} notebook plots to {figure_dir}")
    print(f"Submission: {OUTPUT_DIR / 'submission.csv'}")
    return run_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder containing the three original input CSVs (default: script folder).")
    parser.add_argument("--output-dir", type=Path, default=Path("python_export_outputs"),
                        help="Output folder (default: ./python_export_outputs).")
    parser.add_argument("--show-plots", action="store_true", help="Also open interactive plot windows.")
    args = parser.parse_args()
    run_notebook(data_dir=args.data_dir, output_dir=args.output_dir, show_plots=args.show_plots)


if __name__ == "__main__":
    main()
