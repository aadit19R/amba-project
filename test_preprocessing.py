"""Checks for train-only statistics, validation fallbacks and export integrity."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from preprocessing import BigMartPreprocessor, ID_COLUMNS, TARGET


class BigMartPreprocessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        cls.train = pd.read_csv(root / "train_v9rqX0R.csv")
        cls.test = pd.read_csv(root / "test_AbJTz2l.csv")

    def test_real_data_exports_preserve_rows_and_exclude_targets(self):
        processor = BigMartPreprocessor().fit(self.train.drop(columns=TARGET))
        train_before, test_before = self.train.copy(), self.test.copy()
        train = processor.transform_frame(self.train)
        test = processor.transform_frame(self.test)
        self.assertEqual(list(train), list(test))
        pd.testing.assert_frame_equal(train[ID_COLUMNS], self.train[ID_COLUMNS], check_dtype=False)
        pd.testing.assert_frame_equal(test[ID_COLUMNS], self.test[ID_COLUMNS], check_dtype=False)
        for frame in [train, test]:
            numeric = frame.drop(columns=ID_COLUMNS)
            self.assertTrue(np.isfinite(numeric.to_numpy(dtype=float)).all())
            self.assertFalse(any(TARGET in c for c in frame))
            self.assertTrue(frame.Item_Visibility.gt(0).all())
        pd.testing.assert_frame_equal(self.train, train_before)
        pd.testing.assert_frame_equal(self.test, test_before)

    def test_transform_is_batch_independent_and_does_not_refit(self):
        processor = BigMartPreprocessor().fit(self.train.drop(columns=TARGET))
        observed_stats = processor.weight_by_item_.copy()
        reference_stats = processor.visibility_reference_.copy()
        upper = processor.visibility_upper_
        new = self.test.iloc[:2].copy()
        new["Item_Identifier"] = "FD_UNSEEN"
        new.loc[new.index[0], ["Item_Weight", "Item_Visibility"]] = [np.nan, 0]
        new.loc[new.index[1], ["Item_Weight", "Item_Visibility"]] = [999, .99]
        together = processor.transform(new)
        separately = pd.concat([processor.transform(new.iloc[[i]]) for i in range(len(new))])
        pd.testing.assert_frame_equal(together, separately)
        pd.testing.assert_series_equal(processor.weight_by_item_, observed_stats)
        pd.testing.assert_series_equal(processor.visibility_reference_, reference_stats)
        self.assertEqual(processor.visibility_upper_, upper)

    def test_unseen_item_and_type_use_training_global_fallbacks(self):
        processor = BigMartPreprocessor().fit(self.train.drop(columns=TARGET))
        new = self.test.iloc[[0]].copy()
        new["Item_Identifier"] = "XX_UNSEEN"
        new["Outlet_Identifier"] = "OUT_NEW"
        new["Item_Type"] = "New item type"
        new["Outlet_Type"] = "New outlet type"
        new["Item_Weight"] = np.nan
        new["Outlet_Size"] = pd.NA
        new["Item_Visibility"] = 0.0
        cleaned = processor.clean(new).iloc[0]
        self.assertEqual(cleaned.Item_Weight, processor.weight_global_)
        self.assertEqual(cleaned.Outlet_Size, processor.size_global_)
        self.assertEqual(cleaned.Item_Visibility, processor.visibility_global_)
        transformed = processor.transform(new)
        self.assertTrue(np.isfinite(transformed.to_numpy(dtype=float)).all())
        self.assertEqual(transformed.filter(like="Item_Type_").sum(axis=1).iloc[0], 0)
        self.assertEqual(list(transformed), processor.get_feature_names_out().tolist())

    def test_observed_values_and_nondefault_row_order_survive(self):
        processor = BigMartPreprocessor().fit(self.train.drop(columns=TARGET))
        raw = self.test.iloc[[50, 3, 19]].copy()
        cleaned = processor.clean(raw)
        pd.testing.assert_index_equal(cleaned.index, raw.index)
        for col in ["Item_Weight", "Item_Visibility"]:
            observed = raw[col].notna() & raw[col].gt(0)
            np.testing.assert_array_equal(cleaned.loc[observed, col], raw.loc[observed, col])
        pd.testing.assert_frame_equal(processor.transform_frame(raw)[ID_COLUMNS],
                                      raw[ID_COLUMNS], check_dtype=False)

    def test_extra_target_columns_never_become_predictors(self):
        raw = self.train.head(200).copy()
        raw[TARGET + "_log"] = np.log1p(raw[TARGET])
        with_target = BigMartPreprocessor().fit_transform(raw)
        without_target = BigMartPreprocessor().fit_transform(raw.drop(columns=[TARGET, TARGET + "_log"]))
        pd.testing.assert_frame_equal(with_target, without_target)

    def test_clone_refits_statistics_on_each_fold(self):
        raw = self.train.head(300).drop(columns=TARGET)
        base = BigMartPreprocessor()
        for train_idx, valid_idx in KFold(3, shuffle=True, random_state=42).split(raw):
            fit_rows = raw.iloc[train_idx]
            processor = clone(base).fit(fit_rows)
            expected = (
                fit_rows.assign(Item_Identifier=fit_rows.Item_Identifier.astype("string"))
                .groupby("Item_Identifier").Item_Weight.mean()
            )
            pd.testing.assert_series_equal(processor.weight_by_item_, expected)
            valid = processor.transform(raw.iloc[valid_idx])
            self.assertTrue(np.isfinite(valid.to_numpy(dtype=float)).all())
        self.assertFalse(hasattr(base, "weight_by_item_"))

    def test_bad_inputs_raise_actionable_errors(self):
        raw = self.train.head(200).drop(columns=TARGET)
        with self.assertRaisesRegex(ValueError, "Missing required features"):
            BigMartPreprocessor().fit(raw.drop(columns="Item_MRP"))
        with self.assertRaisesRegex(ValueError, "observed weight"):
            BigMartPreprocessor().fit(raw.assign(Item_Weight=np.nan))
        with self.assertRaisesRegex(ValueError, "positive"):
            BigMartPreprocessor().fit(raw.assign(Item_MRP=-1))


if __name__ == "__main__":
    unittest.main()
