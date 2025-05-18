"""Tests for one-order features."""

from datetime import datetime, timedelta
import unittest

import numpy as np
import pandas as pd

from tfts.features.one_order_feature import (
    add_lag_feature,
    add_moving_average_feature,
    add_roll_feature,
    add_transform_feature,
)


class TestOneOrderFeatures(unittest.TestCase):
    """Test cases for one-order features."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.data = pd.DataFrame(
            {"date": dates, "value": np.random.randn(100).cumsum(), "group": ["A"] * 50 + ["B"] * 50}
        )

    def test_lag_feature(self):
        """Test lag feature generation."""
        # Test single lag
        result = add_lag_feature(self.data, "value", 1, fill_method=None)
        self.assertIn("value_lag_1", result.columns)
        self.assertTrue(result["value_lag_1"].isna().iloc[0])
        self.assertEqual(result["value_lag_1"].iloc[1], self.data["value"].iloc[0])

        # Test multiple lags
        result = add_lag_feature(self.data, "value", [1, 2, 3], fill_method=None)
        self.assertIn("value_lag_1", result.columns)
        self.assertIn("value_lag_2", result.columns)
        self.assertIn("value_lag_3", result.columns)

        # Test with group
        result = add_lag_feature(self.data, "value", 1, group_cols=["group"], fill_method=None)
        self.assertIn("value_lag_1", result.columns)
        self.assertTrue(result["value_lag_1"].isna().iloc[0])
        self.assertTrue(result["value_lag_1"].isna().iloc[50])  # First value in group B

        # Test fill methods
        result = add_lag_feature(self.data, "value", 1, fill_method="bfill")
        self.assertFalse(result["value_lag_1"].isna().any())

        # Test invalid inputs
        with self.assertRaises(ValueError):
            add_lag_feature(self.data, "nonexistent", 1)
        with self.assertRaises(ValueError):
            add_lag_feature(self.data, "value", -1)
        with self.assertRaises(ValueError):
            add_lag_feature(self.data, "value", 1, fill_method="invalid")

    def test_roll_feature(self):
        """Test rolling feature generation."""
        # Test single window
        result = add_roll_feature(self.data, "value", 3)
        self.assertIn("value_roll_3_mean", result.columns)
        self.assertIn("value_roll_3_std", result.columns)
        self.assertIn("value_roll_3_min", result.columns)
        self.assertIn("value_roll_3_max", result.columns)

        # Test multiple windows
        result = add_roll_feature(self.data, "value", [3, 5])
        self.assertIn("value_roll_3_mean", result.columns)
        self.assertIn("value_roll_5_mean", result.columns)

        # Test custom functions
        result = add_roll_feature(self.data, "value", 3, functions=["median", "skew"])
        self.assertIn("value_roll_3_median", result.columns)
        self.assertIn("value_roll_3_skew", result.columns)

        # Test with group
        result = add_roll_feature(self.data, "value", 3, group_cols=["group"])
        self.assertIn("value_roll_3_mean", result.columns)

        # Test invalid inputs
        with self.assertRaises(ValueError):
            add_roll_feature(self.data, "nonexistent", 3)
        with self.assertRaises(ValueError):
            add_roll_feature(self.data, "value", -3)
        with self.assertRaises(ValueError):
            add_roll_feature(self.data, "value", 3, functions=["invalid"])

    def test_transform_feature(self):
        """Test transform feature generation."""
        # Test single function
        result = add_transform_feature(self.data, "value", "log1p")
        self.assertIn("value_log1p", result.columns)

        # Test multiple functions
        result = add_transform_feature(self.data, "value", ["sqrt", "square"])
        self.assertIn("value_sqrt", result.columns)
        self.assertIn("value_square", result.columns)

        # Test all functions
        result = add_transform_feature(self.data, "value", ["log1p", "sqrt", "square", "exp", "sin", "cos", "tan"])
        self.assertIn("value_log1p", result.columns)
        self.assertIn("value_sqrt", result.columns)
        self.assertIn("value_square", result.columns)
        self.assertIn("value_exp", result.columns)
        self.assertIn("value_sin", result.columns)
        self.assertIn("value_cos", result.columns)
        self.assertIn("value_tan", result.columns)

        # Test invalid inputs
        with self.assertRaises(ValueError):
            add_transform_feature(self.data, "nonexistent", "log1p")
        with self.assertRaises(ValueError):
            add_transform_feature(self.data, "value", "invalid")

    def test_moving_average_feature(self):
        """Test moving average feature generation."""
        # Test single window
        result = add_moving_average_feature(self.data, "value", 3)
        self.assertIn("value_sma_3", result.columns)
        self.assertIn("value_ema_3", result.columns)
        self.assertIn("value_wma_3", result.columns)
        self.assertIn("value_hma_3", result.columns)

        # Test multiple windows
        result = add_moving_average_feature(self.data, "value", [3, 5])
        self.assertIn("value_sma_3", result.columns)
        self.assertIn("value_sma_5", result.columns)
        self.assertIn("value_ema_3", result.columns)
        self.assertIn("value_ema_5", result.columns)
        self.assertIn("value_wma_3", result.columns)
        self.assertIn("value_wma_5", result.columns)
        self.assertIn("value_hma_3", result.columns)
        self.assertIn("value_hma_5", result.columns)

        # Test with group
        result = add_moving_average_feature(self.data, "value", 3, group_cols=["group"])
        self.assertIn("value_sma_3", result.columns)
        self.assertIn("value_ema_3", result.columns)
        self.assertIn("value_wma_3", result.columns)
        self.assertIn("value_hma_3", result.columns)

        # Test invalid inputs
        with self.assertRaises(ValueError):
            add_moving_average_feature(self.data, "nonexistent", 3)
        with self.assertRaises(ValueError):
            add_moving_average_feature(self.data, "value", -3)


if __name__ == "__main__":
    unittest.main()
