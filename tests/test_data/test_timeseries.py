"""Tests for TimeSeriesSequence class."""

import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.data.timeseries import TimeSeriesSequence


class TimeSeriesSequenceTest(unittest.TestCase):
    """Test cases for TimeSeriesSequence class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.data = pd.DataFrame(
            {
                "date": dates,
                "value": np.random.randn(100).cumsum(),
                "group": ["A"] * 50 + ["B"] * 50,
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )

        # Create feature config
        self.feature_config = {
            "date_features": {"type": "datetime", "features": ["day", "dayofweek", "month"], "time_col": "date"},
            "lag_features": {
                "type": "lag",
                "columns": "value",
                "lags": [1, 2, 3],
            },
            "rolling_features": {
                "type": "rolling",
                "columns": "value",
                "windows": [3, 5],
                "functions": ["mean", "std"],
            },
            "transform_features": {
                "type": "transform",
                "columns": "value",
                "functions": ["log1p", "sqrt"],
            },
            "moving_average_features": {
                "type": "moving_average",
                "columns": "value",
                "windows": [3, 5],
            },
        }

    def test_initialization(self):
        """Test basic initialization."""
        # Test basic initialization
        seq = TimeSeriesSequence(
            data=self.data,
            time_idx="date",
            target_column="value",
            train_sequence_length=10,
            predict_sequence_length=5,
            batch_size=32,
        )
        self.assertEqual(seq.train_sequence_length, 10)
        self.assertEqual(seq.predict_sequence_length, 5)
        self.assertEqual(seq.batch_size, 32)
        self.assertEqual(seq.mode, "train")

        # Test with group column
        seq = TimeSeriesSequence(
            data=self.data,
            time_idx="date",
            target_column="value",
            train_sequence_length=10,
            predict_sequence_length=5,
            batch_size=32,
            group_column=["group"],
        )
        self.assertEqual(seq.group_ids, ["group"])

        # Test with feature config
        seq = TimeSeriesSequence(
            data=self.data,
            time_idx="date",
            target_column="value",
            train_sequence_length=10,
            predict_sequence_length=5,
            batch_size=32,
            feature_config=self.feature_config,
        )
        self.assertIsNotNone(seq.feature_config)

    def test_validation(self):
        """Test input validation."""
        # Test missing required column
        with self.assertRaises(ValueError):
            TimeSeriesSequence(
                data=self.data.drop(columns=["value"]),
                time_idx="date",
                target_column="value",
                train_sequence_length=10,
            )

        # Test invalid sequence length
        with self.assertRaises(ValueError):
            TimeSeriesSequence(
                data=self.data,
                time_idx="date",
                target_column="value",
                train_sequence_length=0,
            )

        # Test invalid mode
        with self.assertRaises(ValueError):
            TimeSeriesSequence(
                data=self.data,
                time_idx="date",
                target_column="value",
                train_sequence_length=10,
                mode="invalid",
            )

        # Test invalid feature config
        invalid_config = {"invalid_feature": {"type": "invalid_type"}}
        with self.assertRaises(ValueError):
            TimeSeriesSequence(
                data=self.data,
                time_idx="date",
                target_column="value",
                train_sequence_length=10,
                feature_config=invalid_config,
            )

    def test_feature_transformation(self):
        """Test feature transformations."""
        seq = TimeSeriesSequence(
            data=self.data,
            time_idx="date",
            target_column="value",
            train_sequence_length=10,
            predict_sequence_length=5,
            batch_size=32,
            feature_config=self.feature_config,
        )

        # Print all column names for debugging
        print("\nActual columns in the dataframe:")
        for col in seq.data.columns:
            print(f"- {col}")

        # Check if datetime features were added
        self.assertTrue(any(col.startswith("date_") for col in seq.data.columns))

        # Check if lag features were added
        self.assertTrue(any(col.startswith("value_lag_") for col in seq.data.columns))

        # Check if rolling features were added
        self.assertTrue(any(col.startswith("value_roll_") for col in seq.data.columns))

        # Check if transform features were added
        self.assertTrue(any(col.startswith("value_") and col.endswith(("_log1p", "_sqrt")) for col in seq.data.columns))

        # Check if moving average features were added
        # self.assertTrue(any(col.startswith("value") and col.endswith(("_sma_", "_ema_")) for col in seq.data.columns))

    def test_sequence_generation(self):
        """Test sequence generation."""
        seq = TimeSeriesSequence(
            data=self.data,
            time_idx="date",
            target_column="value",
            train_sequence_length=10,
            predict_sequence_length=5,
            batch_size=32,
        )

        # Check if sequences were generated
        self.assertGreater(len(seq.sequences), 0)

        # Check sequence shapes
        encoder_input, decoder_target = seq.sequences[0]
        self.assertEqual(len(encoder_input), seq.train_sequence_length)
        self.assertEqual(len(decoder_target), seq.predict_sequence_length)

        # Test with group column
        seq = TimeSeriesSequence(
            data=self.data,
            time_idx="date",
            target_column="value",
            train_sequence_length=10,
            predict_sequence_length=5,
            batch_size=32,
            group_column=["group"],
        )
        self.assertGreater(len(seq.sequences), 0)

    def test_batch_generation(self):
        """Test batch generation."""
        seq = TimeSeriesSequence(
            data=self.data,
            time_idx="date",
            target_column="value",
            group_column=["group"],
            train_sequence_length=3,
            predict_sequence_length=2,
            stride=1,
            batch_size=2,
            feature_config=self.feature_config,
        )

        # Ensure we have sequences
        self.assertGreater(len(seq.sequences), 0, "No sequences were generated")

        # Get a batch
        batch = seq[0]

        # Check batch structure
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)  # encoder_input and decoder_target

        # Check encoder input shape
        self.assertEqual(batch[0].shape[0], 2)  # batch size
        self.assertEqual(batch[0].shape[1], seq.train_sequence_length)

        # Check decoder target shape
        self.assertEqual(batch[1].shape[0], 2)  # batch size
        self.assertEqual(batch[1].shape[1], seq.predict_sequence_length)

    def test_tf_dataset(self):
        """Test TensorFlow dataset conversion."""
        seq = TimeSeriesSequence(
            data=self.data,
            time_idx="date",
            target_column="value",
            train_sequence_length=10,
            predict_sequence_length=5,
            batch_size=32,
        )

        # Convert to TF dataset
        dataset = seq.get_tf_dataset()
        self.assertIsInstance(dataset, tf.data.Dataset)

        # Check dataset structure
        for batch in dataset.take(1):
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0].shape[1], seq.train_sequence_length)
            self.assertEqual(batch[1].shape[1], seq.predict_sequence_length)

    def test_multiple_targets(self):
        """Test handling of multiple target columns."""
        # Create data with multiple targets
        data = self.data.copy()
        data["value2"] = np.random.randn(100).cumsum()

        seq = TimeSeriesSequence(
            data=data,
            time_idx="date",
            target_column=["value", "value2"],
            train_sequence_length=10,
            predict_sequence_length=5,
            batch_size=32,
        )

        self.assertEqual(len(seq.target), 2)
        self.assertIn("value", seq.target)
        self.assertIn("value2", seq.target)

    def test_different_modes(self):
        """Test different operation modes."""
        modes = ["train", "validation", "test", "inference"]
        for mode in modes:
            seq = TimeSeriesSequence(
                data=self.data,
                time_idx="date",
                target_column="value",
                train_sequence_length=10,
                predict_sequence_length=5,
                batch_size=32,
                mode=mode,
            )
            self.assertEqual(seq.mode, mode)

    def test_from_df_basic(self):
        """Test from_df with basic parameters."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "value": np.random.randn(100).cumsum(),
            }
        )

        seq = TimeSeriesSequence.from_df(
            df,
            time_col="date",
            target_col="value",
            train_length=10,
            predict_length=5,
        )

        self.assertEqual(seq.train_sequence_length, 10)
        self.assertEqual(seq.predict_sequence_length, 5)
        self.assertGreater(len(seq.sequences), 0)

    def test_from_df_with_index(self):
        """Test from_df using DataFrame index as time column."""
        df = pd.DataFrame(
            {
                "value": np.random.randn(100).cumsum(),
            },
            index=pd.date_range("2023-01-01", periods=100, freq="D"),
        )

        seq = TimeSeriesSequence.from_df(
            df,
            target_col="value",
            train_length=10,
            predict_length=5,
        )

        self.assertEqual(seq.train_sequence_length, 10)
        self.assertEqual(seq.predict_sequence_length, 5)
        self.assertGreater(len(seq.sequences), 0)

    def test_from_df_with_groups(self):
        """Test from_df with grouped time series."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=200, freq="D").tolist() * 2,
                "group": ["A"] * 200 + ["B"] * 200,
                "value": np.random.randn(400).cumsum(),
            }
        )

        seq = TimeSeriesSequence.from_df(
            df,
            time_col="date",
            target_col="value",
            group_col="group",
            train_length=10,
            predict_length=5,
        )

        self.assertEqual(seq.group_ids, ["group"])
        self.assertGreater(len(seq.sequences), 0)

    def test_from_df_multiple_targets(self):
        """Test from_df with multiple target columns."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "value1": np.random.randn(100).cumsum(),
                "value2": np.random.randn(100).cumsum(),
            }
        )

        seq = TimeSeriesSequence.from_df(
            df,
            time_col="date",
            target_col=["value1", "value2"],
            train_length=10,
            predict_length=5,
        )

        self.assertEqual(len(seq.target), 2)
        self.assertIn("value1", seq.target)
        self.assertIn("value2", seq.target)

    def test_from_df_fill_missing_dates(self):
        """Test from_df with missing date filling."""
        # Create data with missing dates
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        # Remove some dates
        dates_with_gaps = dates.delete([10, 20, 30, 40])

        df = pd.DataFrame(
            {
                "date": dates_with_gaps,
                "value": np.random.randn(len(dates_with_gaps)).cumsum(),
            }
        )

        seq = TimeSeriesSequence.from_df(
            df,
            time_col="date",
            target_col="value",
            train_length=10,
            predict_length=5,
            fill_missing_dates=True,
            freq="D",
        )

        # Should have filled the missing dates
        self.assertEqual(len(seq.data), 100)

    def test_from_df_fillna(self):
        """Test from_df with NaN filling."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "value": np.random.randn(100).cumsum(),
            }
        )
        # Add some NaN values
        df.loc[10:15, "value"] = np.nan

        seq = TimeSeriesSequence.from_df(
            df,
            time_col="date",
            target_col="value",
            train_length=10,
            predict_length=5,
            fillna_value=0.0,
        )

        # Check that NaN values were filled
        self.assertFalse(seq.data["value"].isna().any())

    def test_from_df_with_feature_config(self):
        """Test from_df with feature configuration."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "value": np.random.randn(100).cumsum(),
            }
        )

        feature_config = {
            "date_features": {
                "type": "datetime",
                "features": ["dayofweek", "month"],
                "time_col": "date",
            }
        }

        seq = TimeSeriesSequence.from_df(
            df,
            time_col="date",
            target_col="value",
            train_length=10,
            predict_length=5,
            feature_config=feature_config,
        )

        # Check if datetime features were added
        self.assertTrue(any(col.startswith("date_") for col in seq.data.columns))

    def test_from_df_validation_errors(self):
        """Test from_df validation errors."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "value": np.random.randn(100).cumsum(),
            }
        )

        # Test missing target_col
        with self.assertRaises(ValueError):
            TimeSeriesSequence.from_df(
                df,
                time_col="date",
                train_length=10,
            )

        # Test missing train_length
        with self.assertRaises(ValueError):
            TimeSeriesSequence.from_df(
                df,
                time_col="date",
                target_col="value",
            )

        # Test invalid time_col
        with self.assertRaises(KeyError):
            TimeSeriesSequence.from_df(
                df,
                time_col="invalid_col",
                target_col="value",
                train_length=10,
            )

        # Test invalid target_col
        with self.assertRaises(KeyError):
            TimeSeriesSequence.from_df(
                df,
                time_col="date",
                target_col="invalid_col",
                train_length=10,
            )

        # Test insufficient data length
        with self.assertRaises(ValueError):
            TimeSeriesSequence.from_df(
                df,
                time_col="date",
                target_col="value",
                train_length=90,
                predict_length=20,
            )

    def test_from_df_numeric_time_index(self):
        """Test from_df with numeric time index."""
        df = pd.DataFrame(
            {
                "time": range(100),
                "value": np.random.randn(100).cumsum(),
            }
        )

        seq = TimeSeriesSequence.from_df(
            df,
            time_col="time",
            target_col="value",
            train_length=10,
            predict_length=5,
        )

        self.assertEqual(seq.train_sequence_length, 10)
        self.assertGreater(len(seq.sequences), 0)

    def test_from_df_with_stride(self):
        """Test from_df with custom stride."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "value": np.random.randn(100).cumsum(),
            }
        )

        seq = TimeSeriesSequence.from_df(
            df,
            time_col="date",
            target_col="value",
            train_length=10,
            predict_length=5,
            stride=2,
        )

        self.assertEqual(seq.stride, 2)
        # With stride=2, we should have fewer sequences
        seq_stride1 = TimeSeriesSequence.from_df(
            df,
            time_col="date",
            target_col="value",
            train_length=10,
            predict_length=5,
            stride=1,
        )
        self.assertLess(len(seq.sequences), len(seq_stride1.sequences))
