"""Tests for TimeSeriesSequence class."""

from datetime import datetime, timedelta
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
