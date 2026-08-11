import unittest

import numpy as np
import pandas as pd

from tfts.features.auto_feature import AutoFeatureEngineer, _default_datetime_features, _default_fourier_features


class AutoFeatureEngineerTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=12, freq="D"),
                "value": np.arange(12, dtype=float),
            }
        )

    def test_requires_fit_before_transform(self):
        engineer = AutoFeatureEngineer(lags=[1], windows=[2])
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            engineer.transform(self.df)

    def test_fit_transform_adds_datetime_and_fourier_features(self):
        engineer = AutoFeatureEngineer(
            lags=[1],
            windows=[2],
            rolling_functions="all",
            add_datetime=True,
            add_fourier=True,
        )
        result = engineer.fit_transform(self.df, time_col="date", target_col="value")

        self.assertEqual(len(result), len(self.df) - 1)
        self.assertIn("value_lag_1", result.columns)
        self.assertIn("value_roll_2_max", result.columns)
        self.assertIn("date_month", result.columns)
        self.assertIn("date_month_sin", result.columns)
        self.assertEqual(len(engineer.get_feature_names()), len(result.columns) - 2)
        self.assertIn("fitted", repr(engineer))

    def test_rolling_function_variants_and_datetime_defaults(self):
        self.assertEqual(AutoFeatureEngineer(rolling_functions="median")._resolve_rolling_functions(), ["median"])
        self.assertEqual(
            AutoFeatureEngineer(rolling_functions=["min", "max"])._resolve_rolling_functions(), ["min", "max"]
        )
        self.assertEqual(_default_fourier_features(), ["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"])
        self.assertEqual(_default_datetime_features(pd.Series([1, 2, 3])), ["month", "dayofweek"])
        self.assertNotIn("hour", _default_datetime_features(self.df["date"]))

        subdaily = pd.Series(pd.date_range("2024-01-01", periods=3, freq="h"))
        self.assertIn("hour", _default_datetime_features(subdaily))

    def test_grouped_features_preserve_group_boundaries(self):
        df = pd.DataFrame(
            {
                "date": list(pd.date_range("2024-01-01", periods=4, freq="D")) * 2,
                "group": ["a"] * 4 + ["b"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            }
        )
        engineer = AutoFeatureEngineer(lags=[1], windows=[2], group_cols=["group"])
        result = engineer.fit_transform(df, time_col="date", target_col="value")
        self.assertEqual(set(result["group"]), {"a", "b"})
        self.assertEqual(result.loc[result["group"] == "b", "value_lag_1"].iloc[0], 10.0)


if __name__ == "__main__":
    unittest.main()
