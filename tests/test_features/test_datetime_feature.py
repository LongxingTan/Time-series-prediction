"""Tests for datetime features."""

import unittest

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from tfts.features.datetime_feature import add_datetime_feature


class TestDatetimeFeatures(unittest.TestCase):
    """Test cases for datetime features."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.data = pd.DataFrame({"date": dates, "value": np.random.randn(100).cumsum()})

    def test_basic_datetime_features(self):
        """Test basic datetime feature generation."""
        # Test with default features
        result = add_datetime_feature(self.data, "date")

        # Check basic features
        self.assertIn("date_year", result.columns)
        self.assertIn("date_month", result.columns)
        self.assertIn("date_day", result.columns)
        self.assertIn("date_dayofweek", result.columns)

        # Check boolean features
        self.assertIn("date_is_month_start", result.columns)
        self.assertIn("date_is_month_end", result.columns)
        self.assertIn("date_is_weekend", result.columns)

        # Check cyclical features
        self.assertIn("date_month_sin", result.columns)
        self.assertIn("date_month_cos", result.columns)
        self.assertIn("date_dayofweek_sin", result.columns)
        self.assertIn("date_dayofweek_cos", result.columns)

    def test_custom_features(self):
        """Test custom feature selection."""
        # Test with specific features
        features = ["year", "month", "is_weekend"]
        result = add_datetime_feature(self.data, "date", features=features)

        self.assertIn("date_year", result.columns)
        self.assertIn("date_month", result.columns)
        self.assertIn("date_is_weekend", result.columns)
        self.assertNotIn("date_day", result.columns)
        self.assertNotIn("date_month_sin", result.columns)

    def test_holiday_features(self):
        """Test holiday feature generation."""
        # Test with holiday features
        result = add_datetime_feature(self.data, "date", features=["is_holiday", "is_business_day"])

        self.assertIn("date_is_holiday", result.columns)
        self.assertIn("date_is_business_day", result.columns)

        # Verify holiday detection
        holidays = USFederalHolidayCalendar().holidays(start=self.data["date"].min(), end=self.data["date"].max())
        holiday_dates = set(holidays)
        result_holidays = set(self.data["date"][result["date_is_holiday"] == 1])
        self.assertTrue(holiday_dates.issubset(result_holidays))

    def test_custom_holidays(self):
        """Test custom holiday handling."""
        # Add custom holiday
        custom_holiday = pd.Timestamp("2023-01-15")
        result = add_datetime_feature(self.data, "date", features=["is_holiday"], custom_holidays=[custom_holiday])

        # Verify custom holiday is detected
        self.assertEqual(result.loc[result["date"] == custom_holiday, "date_is_holiday"].iloc[0], 1)

    def test_cyclical_features(self):
        """Test cyclical feature generation."""
        # Test with only cyclical features
        features = ["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]
        result = add_datetime_feature(self.data, "date", features=features)

        # Check cyclical features
        self.assertIn("date_month_sin", result.columns)
        self.assertIn("date_month_cos", result.columns)
        self.assertIn("date_dayofweek_sin", result.columns)
        self.assertIn("date_dayofweek_cos", result.columns)

        # Verify cyclical properties
        month_sin = result["date_month_sin"]
        month_cos = result["date_month_cos"]
        self.assertTrue(np.allclose(month_sin**2 + month_cos**2, 1, atol=1e-10))

    def test_invalid_inputs(self):
        """Test invalid input handling."""
        # Test with non-existent column
        with self.assertRaises(ValueError):
            add_datetime_feature(self.data, "nonexistent")

        # Test with non-datetime column
        data = self.data.copy()
        data["date"] = "invalid_date"  # This string cannot be converted to datetime
        with self.assertRaises(TypeError):
            add_datetime_feature(data, "date")

        # Test with invalid feature name
        with self.assertRaises(KeyError):
            add_datetime_feature(self.data, "date", features=["invalid_feature"])

    def test_feature_values(self):
        """Test feature value correctness."""
        # Create data with known dates
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({"date": dates})

        result = add_datetime_feature(data, "date")

        # Check specific date values
        self.assertEqual(result.loc[0, "date_year"], 2023)
        self.assertEqual(result.loc[0, "date_month"], 1)
        self.assertEqual(result.loc[0, "date_day"], 1)
        self.assertEqual(result.loc[0, "date_dayofweek"], 6)  # Sunday
        self.assertEqual(result.loc[0, "date_is_month_start"], 1)
        self.assertEqual(result.loc[0, "date_is_weekend"], 1)


if __name__ == "__main__":
    unittest.main()
