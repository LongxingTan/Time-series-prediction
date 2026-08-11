import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.data.auto_preprocessor import AutoPreprocessor
from tfts.data.processor import DataProcessor, _looks_like_time


class DataProcessorTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=20, freq="D"),
                "value": np.arange(20, dtype=float),
            }
        )

    def test_standard_inverse_transform(self):
        processor = DataProcessor(lookback=4, horizon=2, normalize="standard", validation_split=0)
        processor.prepare(self.df, target_col="value", time_col="date")
        normalized = (self.df["value"].to_numpy() - self.df["value"].mean()) / self.df["value"].std()
        np.testing.assert_allclose(processor.inverse_transform(normalized), self.df["value"], rtol=1e-6)

    def test_inference_reuses_training_scaler_and_latest_window(self):
        processor = DataProcessor(lookback=4, horizon=2, normalize="minmax", validation_split=0)
        processor.prepare(self.df, target_col="value", time_col="date")
        scaler = dict(processor._scaler_params)

        inference_df = self.df.tail(4).copy()
        inference_ds = processor.prepare_for_inference(inference_df, target_col="value", time_col="date")
        batches = list(inference_ds.as_numpy_iterator())

        self.assertEqual(processor._scaler_params, scaler)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape, (1, 4, 1))

    def test_validation_split_is_chronological(self):
        processor = DataProcessor(
            lookback=3,
            horizon=1,
            normalize=None,
            validation_split=0.25,
            shuffle=False,
            batch_size=64,
        )
        train_ds, valid_ds = processor.prepare(self.df, target_col="value", time_col="date")
        train_x = next(iter(train_ds))[0].numpy()
        valid_x = next(iter(valid_ds))[0].numpy()
        self.assertLess(train_x[-1, -1, 0], valid_x[0, -1, 0])

    def test_scaler_is_fitted_without_validation_leakage(self):
        processor = DataProcessor(lookback=3, horizon=1, normalize="minmax", validation_split=0.25)
        processor.prepare(self.df, target_col="value", time_col="date")
        self.assertEqual(processor._scaler_params["max"], 14.0)

    def test_prepare_without_validation_and_standard_inverse(self):
        processor = DataProcessor(lookback=4, horizon=2, normalize="standard", validation_split=0, shuffle=False)
        dataset = processor.prepare(self.df, target_col="value", time_col="date")
        self.assertIsInstance(dataset, tf.data.Dataset)
        np.testing.assert_allclose(processor.inverse_transform(np.array([0.0])), np.array([self.df["value"].mean()]))

        minmax = DataProcessor(lookback=4, horizon=2, normalize="minmax", validation_split=0)
        minmax.prepare(self.df, target_col="value", time_col="date")
        np.testing.assert_allclose(minmax.inverse_transform(np.array([0.0, 1.0])), np.array([0.0, 19.0]))
        unfitted = DataProcessor(normalize="minmax")
        np.testing.assert_array_equal(unfitted.inverse_transform(np.array([1.0])), np.array([1.0]))

    def test_inference_requires_fitted_normalizer(self):
        processor = DataProcessor(lookback=4, horizon=2, normalize="minmax")
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            processor.prepare_for_inference(self.df.tail(4), target_col="value", time_col="date")

    def test_validation_and_target_time_inference_helpers(self):
        for kwargs in [
            {"lookback": 0},
            {"horizon": 0},
            {"batch_size": 0},
            {"stride": 0},
            {"validation_split": -0.1},
            {"validation_split": 1.0},
            {"normalize": "bad"},
        ]:
            with self.assertRaises(ValueError):
                DataProcessor(**kwargs)

        numeric = pd.DataFrame({"index": [1, 2, 3], "value": [4.0, 5.0, 6.0]})
        self.assertEqual(DataProcessor._infer_target(numeric), "value")
        self.assertEqual(DataProcessor._infer_target(pd.DataFrame({"first": [1, 2, 3]})), "first")
        self.assertEqual(DataProcessor._infer_time(self.df), "date")
        self.assertEqual(DataProcessor._infer_time(pd.DataFrame({"date": [1, 2]})), "date")
        self.assertEqual(DataProcessor._infer_time(pd.DataFrame({"first": [1, 2]})), "first")
        self.assertTrue(_looks_like_time(pd.Series(pd.date_range("2024-01-01", periods=2))))
        with self.assertRaises(ValueError):
            DataProcessor._infer_target(pd.DataFrame({"text": ["a", "b"]}))
        with self.assertRaises(ValueError):
            DataProcessor(lookback=2, horizon=2, validation_split=0.5)._split_dataset(
                tf.data.Dataset.from_tensor_slices((np.zeros((0, 2, 1)), np.zeros((0, 2, 1))))
            )

    def test_group_normalization_fit_frame_and_time_like_heuristic(self):
        grouped = self.df.assign(group=["a"] * 10 + ["b"] * 10)
        processor = DataProcessor(group_col="group", validation_split=0.25)
        fit_frame = processor._normalization_fit_frame(grouped)
        self.assertEqual(len(fit_frame), 14)
        self.assertTrue(
            DataProcessor._infer_time(pd.DataFrame(index=pd.date_range("2024-01-01", periods=2))).endswith("idx")
        )
        self.assertTrue(_looks_like_time(pd.Series([1, 2, 3])))
        self.assertFalse(_looks_like_time(pd.Series([1, 3, 2])))


class AutoPreprocessorTest(unittest.TestCase):
    def test_forward_fill_preserves_unprocessed_columns(self):
        df = pd.DataFrame({"time": [1, 2, 3], "value": [1.0, np.nan, 3.0]})
        result = AutoPreprocessor(handle_missing="ffill", columns=["value"]).fit_transform(df)
        self.assertEqual(list(result.columns), ["time", "value"])
        self.assertEqual(result.loc[1, "value"], 1.0)

    def test_interpolate_drop_and_no_missing_strategies(self):
        df = pd.DataFrame({"value": [np.nan, 2.0, np.nan, 6.0, np.nan], "label": [1, 2, 3, 4, 5]})
        interpolated = AutoPreprocessor(handle_missing="interpolate", columns=["value"]).fit_transform(df)
        np.testing.assert_allclose(interpolated["value"], [2.0, 2.0, 4.0, 6.0, 6.0])

        dropped = AutoPreprocessor(handle_missing="drop", columns=["value"]).fit_transform(df)
        self.assertEqual(len(dropped), 2)
        untouched = AutoPreprocessor(handle_missing=None, columns=["value"]).fit_transform(df)
        self.assertTrue(untouched["value"].isna().any())

    def test_clip_normalize_and_inverse_transform(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 100.0], "other": [10, 20, 30, 40]})
        preprocessor = AutoPreprocessor(handle_outliers="clip", normalize="minmax", columns=["value"])
        transformed = preprocessor.fit_transform(df)
        self.assertLess(transformed.loc[3, "value"], 1.0)
        restored = preprocessor.inverse_transform(transformed)
        self.assertAlmostEqual(restored.loc[0, "value"], 1.0)
        self.assertEqual(preprocessor.get_fitted_columns(), ["value"])
        self.assertIn("fitted", repr(preprocessor))

        standard = AutoPreprocessor(normalize="standard", columns=["value"]).fit(df)
        standard_values = standard.transform(df)
        np.testing.assert_allclose(standard.inverse_transform(standard_values)["value"], df["value"])

    def test_validation_and_unfitted_errors(self):
        with self.assertRaises(ValueError):
            AutoPreprocessor(handle_missing="invalid")
        with self.assertRaises(ValueError):
            AutoPreprocessor(handle_outliers="invalid")
        with self.assertRaises(ValueError):
            AutoPreprocessor(normalize="invalid")

        preprocessor = AutoPreprocessor()
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            preprocessor.transform(pd.DataFrame({"value": [1.0]}))
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            preprocessor.inverse_transform(pd.DataFrame({"value": [1.0]}))

    def test_missing_requested_columns_are_ignored(self):
        preprocessor = AutoPreprocessor(
            handle_missing=None, handle_outliers="clip", normalize="standard", columns=["missing", "value"]
        )
        result = preprocessor.fit_transform(pd.DataFrame({"value": [1.0, 2.0, 3.0]}))
        self.assertEqual(list(result.columns), ["value"])
        self.assertIn("value", preprocessor.inverse_transform(result))


if __name__ == "__main__":
    unittest.main()
