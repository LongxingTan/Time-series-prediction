"""TFTS Dataset

This module provides a TimeSeriesSequence class for handling time series data in TensorFlow.
It supports various feature transformations, data augmentation, and efficient data loading.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from ..features import (
    FeatureRegistry,
    add_2order_feature,
    add_datetime_feature,
    add_lag_feature,
    add_moving_average_feature,
    add_roll_feature,
    add_transform_feature,
)

logger = logging.getLogger(__name__)


class TimeSeriesSequence(Sequence):
    """TFTS Dataset for fitting timeseries models.

    This class provides a sequence-based dataset for time series data that supports:
    * scaling and encoding of variables
    * normalizing the target variable
    * efficiently converting timeseries in pandas dataframes to tf tensors
    * holding information about static and time-varying variables known and unknown in the future
    * holding information about related categories (such as holidays)
    * down-sampling for data augmentation
    * generating inference, validation and test data

    Args:
        data (pd.DataFrame): Input time series data
        time_idx (str): Name of the time index column
        target_column (str): Name of the target column to predict
        train_sequence_length (int): Length of input sequences for training
        predict_sequence_length (int, optional): Length of sequences to predict. Defaults to 1.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        group_column (List[str], optional): Column(s) to group by. Defaults to None.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        feature_config (Dict, optional): Configuration for feature generation. Defaults to None.
        mode (str, optional): Mode of operation ('train', 'validation', 'test', 'inference'). Defaults to 'train'.
        stride (int, optional): Step size for sequence generation. Defaults to 1.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_idx: str,
        target_column: str,
        train_sequence_length: int,
        predict_sequence_length: int = 1,
        batch_size: int = 32,
        group_column: Optional[List[str]] = None,
        drop_last: bool = False,
        feature_config: Optional[Dict] = None,
        mode: str = "train",
        stride: int = 1,
    ):
        """Initialize the TimeSeriesSequence."""
        self.data = data
        self.time_idx = time_idx
        self.target = [target_column] if isinstance(target_column, str) else target_column
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.group_column = group_column
        self.drop_last = drop_last
        self.feature_config = feature_config or {}
        self.mode = mode
        self.group_ids = group_column or []

        # Initialize feature registry
        self.feature_registry = FeatureRegistry()

        # Validate inputs and apply feature transformations
        self._validate_inputs()
        self._apply_feature_transforms()

        # Generate sequences
        self.sequences = []
        if group_column is not None:
            for _, group in data.groupby(group_column, observed=True):
                self.sequences.extend(self._generate_sequences(group, time_idx=time_idx, target_column=target_column))
        else:
            self.sequences.extend(self._generate_sequences(data, time_idx=time_idx, target_column=target_column))

        logger.info(
            f"Initialized TimeSeriesSequence with {len(self.sequences)} sequences, "
            f"batch_size={batch_size}, mode={mode}"
        )

    def __len__(self) -> int:
        """Get the number of batches in the sequence.

        Returns:
            int: Number of batches
        """
        if self.drop_last:
            return len(self.sequences) // self.batch_size
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of data.

        Args:
            idx (int): Batch index

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (encoder_inputs, decoder_targets)
                encoder_inputs shape: (batch_size, train_sequence_length, num_targets)
                decoder_targets shape: (batch_size, predict_sequence_length, num_targets)
        """
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.sequences))

        batch_sequences = self.sequences[start_idx:end_idx]

        # Stack encoder inputs and decoder targets using np.stack
        encoder_inputs = np.stack([seq[0] for seq in batch_sequences])
        decoder_targets = np.stack([seq[1] for seq in batch_sequences])

        return encoder_inputs, decoder_targets

    def get_tf_dataset(self) -> tf.data.Dataset:
        """Convert to TensorFlow Dataset.

        Returns:
            tf.data.Dataset: TensorFlow dataset with 3D tensors
        """

        def generator():
            for i in range(len(self)):
                yield self[i]

        # Get number of target variables
        num_targets = len(self.target)

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, self.train_sequence_length, num_targets), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.predict_sequence_length, num_targets), dtype=tf.float32),
            ),
        )

    def _generate_sequences(
        self, group: pd.DataFrame, time_idx: str, target_column: str
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Generate sequences from a group of data.

        Args:
            group (pd.DataFrame): Group of data to generate sequences from
            time_idx (str): Name of the time index column
            target_column (str): Name of the target column

        Returns:
            List[Tuple[np.ndarray, np.ndarray, int]]: List of (encoder_input, decoder_target, start_idx) tuples
                Each sequence is a 2D array with shape (length, num_features)
        """
        group = group.sort_values(by=time_idx)
        target_values = group[target_column].values
        time_values = group[time_idx].values

        # Convert time values to numeric if they are datetime
        if pd.api.types.is_datetime64_any_dtype(time_values):
            time_values = time_values.astype(np.int64) // 10**9  # Convert to seconds

        sequences = []
        max_start_idx = len(group) - self.train_sequence_length - self.predict_sequence_length + 1

        for i in range(0, max_start_idx, self.stride):
            # Get indices for encoder sequence
            encoder_start = i
            encoder_end = i + self.train_sequence_length
            encoder_indices = np.arange(encoder_start, encoder_end)

            # Get indices for decoder sequence
            decoder_start = encoder_end
            decoder_end = decoder_start + self.predict_sequence_length
            decoder_indices = np.arange(decoder_start, decoder_end)

            # Check if sequences are continuous
            encoder_time_diffs = np.diff(time_values[encoder_indices])
            decoder_time_diffs = np.diff(time_values[decoder_indices])

            # For datetime values, check if differences are consistent
            if pd.api.types.is_datetime64_any_dtype(group[time_idx]):
                expected_diff = (group[time_idx].iloc[1] - group[time_idx].iloc[0]).total_seconds()
                is_continuous = np.all(np.abs(encoder_time_diffs - expected_diff) < 1e-6) and np.all(
                    np.abs(decoder_time_diffs - expected_diff) < 1e-6
                )
            else:
                is_continuous = np.all(encoder_time_diffs == encoder_time_diffs[0]) and np.all(
                    decoder_time_diffs == decoder_time_diffs[0]
                )

            if (
                len(encoder_indices) == self.train_sequence_length
                and len(decoder_indices) == self.predict_sequence_length
                and is_continuous
            ):
                # Ensure 2D arrays with shape (length, 1) for single feature
                encoder_sequence = target_values[encoder_indices].reshape(-1, 1)
                decoder_sequence = target_values[decoder_indices].reshape(-1, 1)
                sequences.append((encoder_sequence, decoder_sequence, i))

        return sequences

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        # Check required columns
        required_cols = [self.time_idx] + self.target + self.group_ids
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Data is missing required columns: {missing_cols}")

        # Validate sequence lengths
        if self.train_sequence_length < 1:
            raise ValueError("train_sequence_length must be at least 1")
        if self.predict_sequence_length < 1:
            raise ValueError("predict_sequence_length must be at least 1")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")

        # Validate mode
        valid_modes = ["train", "validation", "test", "inference"]
        if self.mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {self.mode}")

        # Validate data length
        if self.train_sequence_length > len(self.data) and not self.group_ids:
            warnings.warn("train_sequence_length is greater than data length")

        # Validate feature configuration
        if self.feature_config:
            valid_feature_types = ["datetime", "lag", "rolling", "transform", "moving_average", "2order"]
            for feature_name, config in self.feature_config.items():
                if not isinstance(config, dict):
                    raise ValueError(f"Feature config for {feature_name} must be a dictionary")
                if "type" not in config:
                    raise ValueError(f"Feature config for {feature_name} must specify 'type'")
                if config["type"] not in valid_feature_types:
                    raise ValueError(f"Invalid feature type {config['type']} for {feature_name}")

    def _apply_feature_transforms(self) -> None:
        """Apply feature transformations based on feature_config."""
        for feature_name, config in self.feature_config.items():
            if feature_name not in self.data.columns and feature_name not in self.target:
                transform_type = config.get("type")
                try:
                    if transform_type == "datetime":
                        self.data = add_datetime_feature(
                            self.data,
                            time_col=config.get("time_col", self.time_idx),
                            features=config.get("features", []),
                        )
                    elif transform_type == "lag":
                        self.data = add_lag_feature(
                            data=self.data,
                            columns=config.get("columns", self.target[0]),
                            lags=config.get("lags", [1]),
                            time_col=self.time_idx,
                            group_cols=self.group_ids if self.group_ids else None,
                        )
                    elif transform_type == "rolling":
                        self.data = add_roll_feature(
                            data=self.data,
                            columns=config.get("columns", self.target[0]),
                            windows=config.get("windows", [3]),
                            functions=config.get("functions", ["mean"]),
                            time_col=self.time_idx,
                            group_cols=self.group_ids if self.group_ids else None,
                        )
                    elif transform_type == "transform":
                        self.data = add_transform_feature(
                            data=self.data,
                            columns=config.get("columns", self.target[0]),
                            functions=config.get("functions", ["log1p", "sqrt"]),
                        )
                    elif transform_type == "moving_average":
                        self.data = add_moving_average_feature(
                            data=self.data,
                            columns=config.get("columns", self.target[0]),
                            windows=config.get("windows", [3]),
                            time_col=self.time_idx,
                            group_cols=self.group_ids if self.group_ids else None,
                        )
                    elif transform_type == "2order":
                        self.data = add_2order_feature(
                            data=self.data,
                            columns=config.get("columns", self.target[0]),
                            config=config,
                            time_col=self.time_idx,
                            group_cols=self.group_ids if self.group_ids else None,
                        )
                    else:
                        logger.warning(f"Unknown transform type {transform_type} for feature {feature_name}")
                except Exception as e:
                    logger.error(f"Error applying feature transform {transform_type} for {feature_name}: {str(e)}")
                    raise
