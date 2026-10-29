"""Meaningful checks for model selection, fold-local transforms and inference."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, ParameterSampler, cross_val_score

from modeling import (
    FeatureScaler, ModelFeatures, MODEL_ORDER, PROFILES, RAW_FEATURES,
    SEARCH_BUDGETS, SEED, SalesRegressor, build_model, computation_context,
    development_split, inverse_target, search_space, transform_target,
    validate_submission,
)
from preprocessing import BigMartPreprocessor, ID_COLUMNS, TARGET


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(Path(__file__).with_name("train_v9rqX0R.csv"))
        cls.raw = cls.train.head(180)[RAW_FEATURES]
        cls.y = cls.train.head(180)[TARGET]

    def test_development_check_split_is_disjoint_and_reproducible(self):
        dev, check, folds, assignments = development_split(self.train)
        self.assertEqual((len(dev), len(check)), (6818, 1705))
        self.assertFalse(set(dev) & set(check))
        self.assertEqual(set(dev) | set(check), set(range(len(self.train))))
        self.assertEqual(set(self.train.iloc[dev].Outlet_Identifier), set(self.train.Outlet_Identifier))
        self.assertEqual(set(self.train.iloc[check].Outlet_Identifier), set(self.train.Outlet_Identifier))
        self.assertTrue(assignments.loc[check, "cv_fold"].isna().all())
        covered = []
        for fit, valid in folds:
            self.assertFalse(set(fit) & set(valid))
            covered.extend(valid)
        self.assertEqual(sorted(covered), list(range(len(dev))))
        dev_again, check_again, _, _ = development_split(self.train)
        np.testing.assert_array_equal(dev, dev_again)
        np.testing.assert_array_equal(check, check_again)

    def test_visibility_policy_preserves_defaults_and_protects_zero_ratios(self):
        default = BigMartPreprocessor().fit(self.raw)
        explicit = BigMartPreprocessor(visibility_policy="impute").fit(self.raw)
        pd.testing.assert_frame_equal(default.transform(self.raw), explicit.transform(self.raw))
        raw = self.raw.copy()
        raw.loc[raw.index[:2], "Item_Identifier"] = "FD_ZERO_TEST"
        raw.loc[raw.index[:2], "Item_Visibility"] = 0.0
        preserve = BigMartPreprocessor(visibility_policy="keep_zero").fit(raw)
        cleaned = preserve.clean(raw)
        zeros = raw.Item_Visibility.eq(0)
        self.assertTrue(cleaned.loc[zeros, "Item_Visibility"].eq(0).all())
        result = preserve.transform(raw)
        self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())
        self.assertTrue(result.loc[raw.Item_Identifier.eq("FD_ZERO_TEST"),
                                   "Item_Visibility_MeanRatio"].eq(0).all())

    def test_profiles_have_no_identifiers_or_target_derived_features(self):
        encoded = BigMartPreprocessor().fit_transform(self.raw)
        for profile in PROFILES + ["price_interactions"]:
            for visibility in ["raw", "sqrt", "capped"]:
                transform = ModelFeatures(profile, visibility, include_ratio=True).fit(encoded)
                result = transform.transform(encoded)
                self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())
                self.assertTrue(set(ID_COLUMNS).isdisjoint(result.columns))
                self.assertFalse(any(TARGET in c for c in result))
                self.assertFalse(any(c.startswith("Item_Category_") for c in result))
                self.assertEqual(result.columns.tolist(), transform.get_feature_names_out().tolist())
                self.assertEqual(result.filter(like="MRP_x_").shape[1],
                                 3 if profile == "price_interactions" else 0)
                self.assertEqual(bool(result.filter(like="Outlet_OUT").shape[1]),
                                 profile in {"compact", "price_interactions"})

    def test_unknown_categories_keep_the_fitted_schema(self):
        processor = BigMartPreprocessor().fit(self.raw)
        features = ModelFeatures("descriptive_onehot").fit(processor.transform(self.raw))
        unseen = self.raw.iloc[[0]].copy()
        unseen["Item_Type"] = "Brand new type"
        unseen["Outlet_Type"] = "Brand new outlet type"
        unseen["Outlet_Identifier"] = "OUT_NEW"
        encoded = features.transform(processor.transform(unseen))
        self.assertEqual(encoded.columns.tolist(), features.get_feature_names_out().tolist())
        self.assertTrue(np.isfinite(encoded.to_numpy(dtype=float)).all())

    def test_scaling_uses_only_fit_rows_and_keeps_binary_values(self):
        frame = pd.DataFrame({"Item_MRP": [10., 20., 30.], "Item_Weight": [1., 2., 3.],
                              "Outlet_OUT010": [0, 1, 0], "Item_Visibility_ZeroOrMissing": [1, 0, 0]})
        scaler = FeatureScaler().fit(frame.iloc[:2])
        np.testing.assert_allclose(scaler.scaler_.mean_, [15, 1.5])
        transformed = scaler.transform(frame)
        self.assertEqual(transformed.Item_MRP.iloc[2], 3)
        pd.testing.assert_frame_equal(transformed[["Outlet_OUT010", "Item_Visibility_ZeroOrMissing"]],
                                      frame[["Outlet_OUT010", "Item_Visibility_ZeroOrMissing"]])

    def test_target_roundtrips_and_nonnegative_inverse(self):
        values = np.array([0., 1., 100., 10000.])
        for mode in ["raw", "sqrt", "log1p"]:
            np.testing.assert_allclose(inverse_target(transform_target(values, mode), mode), values)
            self.assertTrue((inverse_target([-10, -1, 0], mode) >= 0).all())
        with self.assertRaises(ValueError):
            transform_target([-1], "sqrt")

    def test_all_models_fit_and_predict_original_scale(self):
        with computation_context():
            for name in MODEL_ORDER:
                estimator = build_model(name)
                if name == "RandomForest":
                    estimator.set_params(pipeline__model__n_estimators=10,
                                         pipeline__model__min_samples_leaf=2)
                for mode in ["raw", "sqrt", "log1p"]:
                    fitted = clone(estimator).set_params(target_mode=mode).fit(self.raw, self.y)
                    predictions = fitted.predict(self.raw.iloc[:10])
                    self.assertEqual(predictions.shape, (10,))
                    self.assertTrue(np.isfinite(predictions).all())
                    self.assertTrue((predictions >= 0).all())
                    np.testing.assert_allclose(predictions, inverse_target(
                        fitted.pipeline_.predict(self.raw.iloc[:10]), mode))

    def test_cv_and_grid_search_clone_and_refit_the_whole_pipeline(self):
        raw = self.raw.iloc[:100]
        target = self.y.iloc[:100]
        with computation_context():
            scores = cross_val_score(build_model("LinearRegression"), raw, target, cv=2,
                                     scoring="neg_root_mean_squared_error", error_score="raise")
            self.assertTrue(np.isfinite(scores).all())
            search = GridSearchCV(build_model("LinearRegression"),
                                  {"target_mode": ["raw", "sqrt"]}, cv=2,
                                  scoring="neg_root_mean_squared_error", error_score="raise").fit(raw, target)
        fitted = search.best_estimator_.pipeline_.named_steps["preprocess"]
        expected = raw.assign(Item_Identifier=raw.Item_Identifier.astype("string")) \
            .groupby("Item_Identifier").Item_Weight.mean()
        pd.testing.assert_series_equal(fitted.weight_by_item_, expected)

    def test_search_budgets_and_linked_visibility_choices(self):
        self.assertEqual(len(list(ParameterGrid(search_space("LinearRegression")))), 12)
        for name in ["KNN", "SVR", "RandomForest"]:
            samples = list(ParameterSampler(search_space(name), n_iter=SEARCH_BUDGETS[name], random_state=SEED))
            self.assertEqual(len(samples), SEARCH_BUDGETS[name])
            self.assertEqual({p["target_mode"] for p in samples}, {"raw", "sqrt", "log1p"})
            for parameters in samples:
                build_model(name).set_params(**parameters)
                if parameters["pipeline__preprocess__visibility_policy"] == "keep_zero":
                    self.assertEqual(parameters["pipeline__features__visibility"], "raw")

    def test_model_serialization_roundtrip(self):
        with computation_context():
            estimator = build_model("LinearRegression").set_params(target_mode="sqrt").fit(self.raw, self.y)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.joblib"
                joblib.dump(estimator, path)
                loaded = joblib.load(path)
                np.testing.assert_allclose(loaded.predict(self.raw), estimator.predict(self.raw))

    def test_submission_rejects_reordered_or_invalid_predictions(self):
        test = self.raw.iloc[:3].reset_index(drop=True)
        sample = test[ID_COLUMNS].assign(Item_Outlet_Sales=0.)
        valid = sample.copy()
        validate_submission(valid, test, sample)
        for invalid in [valid.iloc[::-1].reset_index(drop=True),
                        valid.assign(Item_Outlet_Sales=np.nan),
                        valid.assign(Item_Outlet_Sales=-1.)]:
            with self.assertRaises((ValueError, AssertionError)):
                validate_submission(invalid, test, sample)


if __name__ == "__main__":
    unittest.main()
