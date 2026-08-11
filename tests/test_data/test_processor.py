import unittest

import numpy as np
import pandas as pd

from tfts.data.auto_preprocessor import AutoPreprocessor
from tfts.data.processor import DataProcessor


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


class AutoPreprocessorTest(unittest.TestCase):
    def test_forward_fill_preserves_unprocessed_columns(self):
        df = pd.DataFrame({"time": [1, 2, 3], "value": [1.0, np.nan, 3.0]})
        result = AutoPreprocessor(handle_missing="ffill", columns=["value"]).fit_transform(df)
        self.assertEqual(list(result.columns), ["time", "value"])
        self.assertEqual(result.loc[1, "value"], 1.0)


if __name__ == "__main__":
    unittest.main()
