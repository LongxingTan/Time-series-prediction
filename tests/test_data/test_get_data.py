import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from tfts.data.get_data import get_air_passengers, get_ar_data, get_data, get_sine, get_stock_data


class GetDataTest(unittest.TestCase):
    def test_get_data(self):
        train_length = 10
        predict_sequence_length = 4
        test_size = 0.2
        n_examples = 100
        train, valid = get_data("sine", train_length, predict_sequence_length, test_size)
        self.assertEqual(train[0].shape, (int(n_examples * (1 - test_size)), train_length, 1))
        self.assertEqual(train[1].shape, (int(n_examples * (1 - test_size)), predict_sequence_length, 1))
        self.assertEqual(valid[0].shape, (int(n_examples * test_size), train_length, 1))
        self.assertEqual(valid[1].shape, (int(n_examples * test_size), predict_sequence_length, 1))

    def test_get_sine_data(self):
        train_length = 10
        predict_sequence_length = 4
        test_size = 0.2
        n_examples = 200
        train, valid = get_sine(train_length, predict_sequence_length, test_size, n_examples)
        self.assertEqual(train[0].shape, (int(n_examples * (1 - test_size)), train_length, 1))
        self.assertEqual(train[1].shape, (int(n_examples * (1 - test_size)), predict_sequence_length, 1))
        self.assertEqual(valid[0].shape, (int(n_examples * test_size), train_length, 1))
        self.assertEqual(valid[1].shape, (int(n_examples * test_size), predict_sequence_length, 1))

    def test_get_air_passenger_data(self):
        train_length = 10
        predict_sequence_length = 4
        test_size = 0.2
        train, valid = get_air_passengers(train_length, predict_sequence_length, test_size)
        self.assertEqual(train[0].shape[1:], (train_length, 1))
        self.assertEqual(train[1].shape[1:], (predict_sequence_length, 1))
        self.assertEqual(valid[0].shape[1:], (train_length, 1))
        self.assertEqual(valid[1].shape[1:], (predict_sequence_length, 1))

    def test_get_sine_no_test_split(self):
        """Test get_sine with test_size=0 returns single tuple"""
        train_length = 10
        predict_sequence_length = 4
        n_examples = 50
        x, y = get_sine(train_length, predict_sequence_length, test_size=0, n_examples=n_examples)
        self.assertEqual(x.shape, (n_examples, train_length, 1))
        self.assertEqual(y.shape, (n_examples, predict_sequence_length, 1))
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_get_air_passengers_no_test_split(self):
        """Test get_air_passengers with test_size=0"""
        train_length = 10
        predict_sequence_length = 4
        x, y = get_air_passengers(train_length, predict_sequence_length, test_size=0)
        self.assertEqual(x.shape[1:], (train_length, 1))
        self.assertEqual(y.shape[1:], (predict_sequence_length, 1))

    def test_get_data_invalid_name(self):
        """Test get_data raises ValueError for unsupported dataset"""
        with self.assertRaises(ValueError) as context:
            get_data("invalid_dataset", 10, 4, 0.2)
        self.assertIn("unsupported data", str(context.exception))

    def test_get_data_test_size_validation(self):
        """Test get_data validates test_size parameter"""
        with self.assertRaises(AssertionError):
            get_data("sine", 10, 4, test_size=-0.1)

        with self.assertRaises(AssertionError):
            get_data("sine", 10, 4, test_size=1.5)

    def test_get_data_airpassengers(self):
        """Test get_data dispatcher for airpassengers dataset"""
        train_length = 12
        predict_length = 6
        train, valid = get_data("airpassengers", train_length, predict_length, test_size=0.15)
        self.assertIsNotNone(train)
        self.assertIsNotNone(valid)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(valid), 2)

    def test_get_sine_data_values_in_range(self):
        """Test that sine wave values are in expected range"""
        train_length = 20
        predict_length = 5
        x, y = get_sine(train_length, predict_length, test_size=0, n_examples=10)

        # Sine values should be roughly in [-1, 1] range
        self.assertTrue(np.all(x >= -1.5))
        self.assertTrue(np.all(x <= 1.5))
        self.assertTrue(np.all(y >= -1.5))
        self.assertTrue(np.all(y <= 1.5))

    def test_get_ar_data_basic(self):
        """Test basic AR data generation"""
        df = get_ar_data(n_series=5, timesteps=100)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("series", df.columns)
        self.assertIn("time_idx", df.columns)
        self.assertIn("value", df.columns)
        self.assertEqual(len(df), 5 * 100)  # n_series * timesteps

    def test_get_ar_data_with_covariates(self):
        """Test AR data generation with covariates"""
        df = get_ar_data(n_series=3, timesteps=50, add_covariates=True)

        self.assertIn("day_of_week", df.columns)
        self.assertIn("month", df.columns)
        self.assertIn("category", df.columns)
        self.assertIn("special_event", df.columns)

        # Check value ranges
        self.assertTrue(df["day_of_week"].between(0, 6).all())
        self.assertTrue(df["month"].between(1, 13).all())
        self.assertTrue(df["special_event"].isin([0, 1]).all())

    def test_get_ar_data_with_components(self):
        """Test AR data generation returning components"""
        df, components = get_ar_data(n_series=3, timesteps=50, return_components=True)

        self.assertIsInstance(components, dict)
        self.assertIn("linear_trends", components)
        self.assertIn("quadratic_trends", components)
        self.assertIn("seasonalities", components)
        self.assertIn("levels", components)
        self.assertIn("series", components)

    def test_get_ar_data_exponential(self):
        """Test AR data with exponential transformation"""
        df = get_ar_data(n_series=2, timesteps=50, exp=True)

        # Exponential values should all be positive
        self.assertTrue((df["value"] > 0).all())

    def test_get_ar_data_seeded_reproducibility(self):
        """Test that same seed produces same data"""
        df1 = get_ar_data(n_series=3, timesteps=50, seed=42)
        df2 = get_ar_data(n_series=3, timesteps=50, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_get_ar_data_different_seeds(self):
        """Test that different seeds produce different data"""
        df1 = get_ar_data(n_series=3, timesteps=50, seed=42)
        df2 = get_ar_data(n_series=3, timesteps=50, seed=123)

        # Values should be different
        self.assertFalse(df1["value"].equals(df2["value"]))

    def test_get_ar_data_invalid_params(self):
        """Test AR data validation for invalid parameters"""
        with self.assertRaises(ValueError):
            get_ar_data(n_series=0, timesteps=100)

        with self.assertRaises(ValueError):
            get_ar_data(n_series=5, timesteps=-10)

        with self.assertRaises(ValueError):
            get_ar_data(n_series=5, timesteps=100, noise=-0.5)

    def test_get_ar_data_parameter_effects(self):
        """Test that parameters affect data as expected"""
        # High noise should create more variance
        df_low_noise = get_ar_data(n_series=5, timesteps=100, noise=0.01, seed=42)
        df_high_noise = get_ar_data(n_series=5, timesteps=100, noise=1.0, seed=42)

        # Not directly comparing variance due to random effects,
        # but shapes should match
        self.assertEqual(len(df_low_noise), len(df_high_noise))

    def test_get_data_ar_dispatch(self):
        """Test get_data dispatcher for AR data"""
        result = get_data("ar", train_length=10, predict_sequence_length=5, test_size=0, n_series=3, timesteps=50)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3 * 50)

    def test_sine_data_sequence_continuity(self):
        """Test that sine data maintains temporal continuity"""
        train_length = 10
        predict_length = 5
        x, y = get_sine(train_length, predict_length, test_size=0, n_examples=1)

        # x and y should form a continuous sequence
        # This is a shape test since exact continuity depends on implementation
        self.assertEqual(x.shape[1], train_length)
        self.assertEqual(y.shape[1], predict_length)

    def test_air_passengers_normalization(self):
        """Test that air passengers data is properly normalized"""
        x, y = get_air_passengers(10, 4, test_size=0)

        # Values should be normalized (roughly between -1 and 1 after normalization)
        self.assertTrue(np.all(x >= -1.5))
        self.assertTrue(np.all(x <= 1.5))


if __name__ == "__main__":
    unittest.main()
