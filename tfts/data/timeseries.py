"""TFTS Dataset"""

import logging
from typing import Dict, List, Optional
import warnings

import numpy as np
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

    * scaling and encoding of variables
    * normalizing the target variable
    * efficiently converting timeseries in pandas dataframes to tf tensors
    * holding information about static and time-varying variables known and unknown in the future
    * holding information about related categories (such as holidays)
    * down-sampling for data augmentation
    * generating inference, validation and test data
    """

    def __init__(
        self,
        data,
        time_idx: str,
        target_column: str,
        train_sequence_length,
        predict_sequence_length: int = 1,
        batch_size: int = 32,
        group_column: List[str] = None,
        drop_last: bool = False,
        feature_config: Optional[Dict] = None,
    ):
        self.data = data
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.stride = 1
        self.batch_size = batch_size
        self.group_column = group_column
        self.drop_last = drop_last

        self._validate_inputs()
        self._apply_feature_transforms()

        self.sequences = []
        if group_column is not None:
            for _, group in data.groupby(group_column, observed=True):
                self.sequences.extend(self._generate_sequences(group, time_idx=time_idx, target_column=target_column))
        else:
            self.sequences.extend(self._generate_sequences(data, time_idx=time_idx, target=self.target[0]))

    def __len__(self):
        if self.drop_last:
            return len(self.sequences) // self.batch_size
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.sequences))

        batch_sequences = self.sequences[start_idx:end_idx]
        encoder_inputs = np.array([seq[0] for seq in batch_sequences])
        decoder_targets = np.array([seq[1] for seq in batch_sequences])
        return encoder_inputs, decoder_targets

    def get_tf_dataset(self):
        """Convert to TensorFlow Dataset."""

        def generator():
            for i in range(len(self)):
                yield self[i]

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, self.train_sequence_length), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.predict_sequence_length), dtype=tf.float32),
            ),
        )

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        required_cols = [self.time_idx] + self.target + self.group_ids
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Data is missing required columns: {missing_cols}")

        if self.train_sequence_length < 1:
            raise ValueError("train_sequence_length must be at least 1")
        if self.predict_sequence_length < 1:
            raise ValueError("predict_sequence_length must be at least 1")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")

        valid_modes = ["train", "validation", "test", "inference"]
        if self.mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {self.mode}")

        if self.train_sequence_length > len(self.data) and not self.group_ids:
            warnings.warn("train_sequence_length is greater than data length")

    def _apply_feature_transforms(self) -> None:
        """Apply feature transformations based on feature_config."""
        for feature_name, config in self.feature_config.items():
            if feature_name not in self.data.columns and feature_name not in self.target:
                transform_type = config.get("type")
                if transform_type == "datetime":
                    self.data = add_datetime_feature(self.data, feature_name, config)
                elif transform_type == "lag":
                    self.data = add_lag_feature(
                        self.data, feature_name, config, time_idx=self.time_idx, group_ids=self.group_ids
                    )
                elif transform_type == "rolling":
                    self.data = add_roll_feature(
                        self.data, feature_name, config, time_idx=self.time_idx, group_ids=self.group_ids
                    )
                elif transform_type == "transform":
                    self.data = add_transform_feature(self.data, feature_name, config)
                elif transform_type == "moving_average":
                    self.data = add_moving_average_feature(
                        self.data, feature_name, config, time_idx=self.time_idx, group_ids=self.group_ids
                    )
                elif transform_type == "2order":
                    self.data = add_2order_feature(
                        self.data, feature_name, config, time_idx=self.time_idx, group_ids=self.group_ids
                    )
                else:
                    self.logger.warning(f"Unknown transform type {transform_type} for feature {feature_name}")

    def _generate_sequences(self, group, time_idx, target_column):
        group = group.sort_values(by=time_idx)
        target_values = group[target_column].values
        time_values = group[time_idx].values

        sequences = []
        for i in range(0, len(group) - self.train_sequence_length - self.predict_sequence_length + 1, self.stride):
            encoder_indices = np.where(
                (time_values >= time_values[i]) & (time_values < time_values[i + self.train_sequence_length])
            )[0]
            decoder_indices = np.where(
                (time_values >= time_values[i + self.train_sequence_length])
                & (time_values < time_values[i + self.train_sequence_length + self.predict_sequence_length])
            )[0]

            if (
                len(encoder_indices) == self.train_sequence_length
                and len(decoder_indices) == self.predict_sequence_length
                and np.all(np.diff(time_values[encoder_indices]) == 1)
                and np.all(np.diff(time_values[decoder_indices]) == 1)
            ):
                encoder_sequence = target_values[encoder_indices]
                decoder_sequence = target_values[decoder_indices]
                sequences.append((encoder_sequence, decoder_sequence))
        return sequences
