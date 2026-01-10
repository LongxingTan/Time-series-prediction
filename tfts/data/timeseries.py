"""TFTS Dataset

This module provides a TimeSeriesSequence class for handling time series data in TensorFlow.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
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

    Example:
        >>> # Using from_df for easy dataset creation
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=100),
        ...     'value': np.random.randn(100),
        ...     'group': ['A'] * 50 + ['B'] * 50
        ... })
        >>> dataset = TimeSeriesSequence.from_df(
        ...     df,
        ...     time_col='date',
        ...     target_col='value',
        ...     train_length=24,
        ...     predict_length=12,
        ...     group_col='group'
        ... )
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        train_sequence_length: int,
        predict_sequence_length: int = 1,
        time_idx: Optional[str] = None,
        batch_size: int = 32,
        group_column: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        drop_last: bool = False,
        feature_config: Optional[Dict] = None,
        mode: str = "train",
        stride: int = 1,
        processor: Optional[List[Callable]] = None,
    ):
        """Initialize the TimeSeriesSequence."""
        self.data = data.copy()
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

    def _build_sequences(self):
        """Builds a lookup table for sequences to avoid heavy DataFrame slicing during training."""
        sequence_indices = []

        if self.group_column:
            grouped = self.data.groupby(self.group_column)
        else:
            grouped = [("all", self.data)]

        for _, group in grouped:
            group = group.sort_values(self.time_idx)
            n_rows = len(group)
            max_idx = n_rows - self.train_sequence_length - self.predict_sequence_length + 1

            # Pre-extract numpy arrays for speed
            feature_data = group[self.features].values.astype(np.float32)
            target_data = group[self.target].values.astype(np.float32)

            for i in range(0, max_idx, self.stride):
                sequence_indices.append(
                    {
                        "x": feature_data[i : i + self.train_sequence_length],
                        "y": target_data[
                            i
                            + self.train_sequence_length : i
                            + self.train_sequence_length
                            + self.predict_sequence_length
                        ],
                    }
                )

        return sequence_indices

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
        """Convert to high-performance tf.data pipeline."""
        # Get feature dimension from actual sequence data
        if len(self.sequences) > 0:
            num_features = self.sequences[0][0].shape[-1]
        else:
            num_features = len(self.target)

        output_signature = (
            tf.TensorSpec(shape=(None, self.train_sequence_length, num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.predict_sequence_length, len(self.target)), dtype=tf.float32),
        )
        return tf.data.Dataset.from_generator(
            lambda: (self[i] for i in range(len(self))), output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

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
                sequences.append((encoder_sequence, decoder_sequence))

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

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        target_col: Union[str, List[str]] = None,
        train_length: int = None,
        predict_length: int = 1,
        group_col: Optional[Union[str, List[str]]] = None,
        feature_cols: Optional[List[str]] = None,
        batch_size: int = 32,
        stride: int = 1,
        mode: str = "train",
        drop_last: bool = False,
        feature_config: Optional[Dict] = None,
        fill_missing_dates: bool = False,
        freq: Optional[str] = None,
        fillna_value: Optional[float] = None,
        **kwargs,
    ) -> "TimeSeriesSequence":
        """Create a TimeSeriesSequence from a pandas DataFrame.

        This is a convenient factory method for creating datasets from DataFrames,
        similar to darts' from_dataframe API. It provides a more intuitive interface
        with automatic validation and preprocessing.

        Args:
            df (pd.DataFrame): Input DataFrame containing time series data.
            time_col (str, optional): Name of the time column. If None, uses DataFrame index.
                The column should contain datetime or numeric values representing time.
            target_col (str or List[str]): Name(s) of the target column(s) to predict.
                Can be a single column name (str) or list of column names for multivariate prediction.
            train_length (int): Length of input sequences for training (lookback window).
            predict_length (int, optional): Length of sequences to predict (forecast horizon). Defaults to 1.
            group_col (str or List[str], optional): Column name(s) for grouping multiple time series.
                Use this for hierarchical or grouped time series data. Defaults to None.
            feature_cols (List[str], optional): Additional feature columns to include. Defaults to None.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            stride (int, optional): Step size for sequence generation. Use stride > 1 for downsampling. Defaults to 1.
            mode (str, optional): Mode of operation ('train', 'validation', 'test', 'inference'). Defaults to 'train'.
            drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
            feature_config (Dict, optional): Configuration for automatic feature engineering.
                Supports: datetime, lag, rolling, transform, moving_average, 2order features. Defaults to None.
            fill_missing_dates (bool, optional): Whether to fill missing dates in the time series. Defaults to False.
            freq (str, optional): Frequency of the time series (e.g., 'D' for daily, 'H' for hourly).
                Required if fill_missing_dates is True. Defaults to None.
            fillna_value (float, optional): Value to use for filling missing values. Defaults to None (no filling).
            **kwargs: Additional keyword arguments passed to TimeSeriesSequence.__init__.

        Returns:
            TimeSeriesSequence: A configured TimeSeriesSequence instance ready for training.

        Raises:
            ValueError: If required parameters are missing or invalid.
            KeyError: If specified columns are not found in the DataFrame.

        Example:
            >>> # Basic usage with single time series
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            ...     'value': np.random.randn(100).cumsum()
            ... })
            >>> dataset = TimeSeriesSequence.from_df(
            ...     df,
            ...     time_col='date',
            ...     target_col='value',
            ...     train_length=30,
            ...     predict_length=7
            ... )
            >>>
            >>> # Multi-series with grouping
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2023-01-01', periods=200, freq='D').repeat(2),
            ...     'store_id': ['A', 'B'] * 100,
            ...     'sales': np.random.randn(200).cumsum()
            ... })
            >>> dataset = TimeSeriesSequence.from_df(
            ...     df,
            ...     time_col='date',
            ...     target_col='sales',
            ...     group_col='store_id',
            ...     train_length=30,
            ...     predict_length=7
            ... )
            >>>
            >>> # With feature engineering
            >>> feature_config = {
            ...     'date_features': {
            ...         'type': 'datetime',
            ...         'features': ['dayofweek', 'month']
            ...     },
            ...     'lag_features': {
            ...         'type': 'lag',
            ...         'lags': [1, 7, 14]
            ...     }
            ... }
            >>> dataset = TimeSeriesSequence.from_df(
            ...     df,
            ...     time_col='date',
            ...     target_col='sales',
            ...     train_length=30,
            ...     predict_length=7,
            ...     feature_config=feature_config
            ... )
        """
        # Validate required parameters
        if target_col is None:
            raise ValueError("target_col is required")
        if train_length is None:
            raise ValueError("train_length is required")

        # Make a copy to avoid modifying the original DataFrame
        data = df.copy()

        # Handle time column
        if time_col is None:
            # Use DataFrame index as time column
            if not isinstance(data.index, (pd.DatetimeIndex, pd.RangeIndex)):
                # Try to convert index to datetime or numeric
                try:
                    data.index = pd.to_datetime(data.index)
                except (ValueError, TypeError):
                    try:
                        data.index = pd.to_numeric(data.index)
                    except (ValueError, TypeError):
                        raise ValueError(
                            "DataFrame index must be datetime-like or numeric, " "or specify time_col parameter"
                        )
            # Reset index to make it a column
            data = data.reset_index()
            time_col = data.columns[0]
        else:
            # Validate time column exists
            if time_col not in data.columns:
                raise KeyError(f"time_col '{time_col}' not in DataFrame columns: {list(data.columns)}")

            # Try to convert time column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
                try:
                    data[time_col] = pd.to_datetime(data[time_col])
                except (ValueError, TypeError):
                    # If conversion fails, assume it's a numeric time index
                    if not pd.api.types.is_numeric_dtype(data[time_col]):
                        logger.warning(
                            f"time_col '{time_col}' is not datetime or numeric. "
                            "This may cause issues with sequence generation."
                        )

        # Handle target columns
        if isinstance(target_col, str):
            target_cols = [target_col]
        else:
            target_cols = target_col

        # Validate target columns exist
        missing_targets = [col for col in target_cols if col not in data.columns]
        if missing_targets:
            raise KeyError(f"target_col(s) {missing_targets} not in DataFrame columns: {list(data.columns)}")

        # Handle group columns
        if group_col is not None:
            if isinstance(group_col, str):
                group_cols = [group_col]
            else:
                group_cols = group_col

            # Validate group columns exist
            missing_groups = [col for col in group_cols if col not in data.columns]
            if missing_groups:
                raise KeyError(f"group_col(s) {missing_groups} not in DataFrame columns: {list(data.columns)}")
        else:
            group_cols = None

        # Handle missing dates
        if fill_missing_dates:
            if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
                raise ValueError("fill_missing_dates requires time_col to be datetime type")

            if freq is None:
                # Try to infer frequency
                freq = pd.infer_freq(data[time_col].sort_values())
                if freq is None:
                    raise ValueError(
                        "Could not infer frequency from time_col. Please specify freq parameter "
                        "(e.g., 'D' for daily, 'H' for hourly)"
                    )
                logger.info(f"Inferred frequency: {freq}")

            if group_cols is not None:
                # Fill missing dates for each group separately
                filled_groups = []
                for group_name, group_data in data.groupby(group_cols, observed=True):
                    # Create full date range for this group
                    min_date = group_data[time_col].min()
                    max_date = group_data[time_col].max()
                    full_dates = pd.date_range(start=min_date, end=max_date, freq=freq)

                    # Reindex and forward fill
                    group_data = group_data.set_index(time_col).reindex(full_dates).reset_index()
                    group_data = group_data.rename(columns={"index": time_col})

                    # Restore group column values
                    if isinstance(group_name, tuple):
                        for i, col in enumerate(group_cols):
                            group_data[col] = group_name[i]
                    else:
                        group_data[group_cols[0]] = group_name

                    filled_groups.append(group_data)

                data = pd.concat(filled_groups, ignore_index=True)
            else:
                # Fill missing dates for single time series
                min_date = data[time_col].min()
                max_date = data[time_col].max()
                full_dates = pd.date_range(start=min_date, end=max_date, freq=freq)
                data = data.set_index(time_col).reindex(full_dates).reset_index()
                data = data.rename(columns={"index": time_col})

        # Handle missing values
        if fillna_value is not None:
            # Fill NaN values in target columns
            for col in target_cols:
                data[col] = data[col].fillna(fillna_value)

        # Validate data length
        min_required_length = train_length + predict_length
        if group_cols is not None:
            # Check each group has sufficient data
            for group_name, group_data in data.groupby(group_cols, observed=True):
                if len(group_data) < min_required_length:
                    warnings.warn(
                        f"Group {group_name} has only {len(group_data)} rows, "
                        f"but requires at least {min_required_length} rows "
                        f"(train_length={train_length} + predict_length={predict_length}). "
                        "This group will produce no sequences."
                    )
        else:
            if len(data) < min_required_length:
                raise ValueError(
                    f"DataFrame has only {len(data)} rows, "
                    f"but requires at least {min_required_length} rows "
                    f"(train_length={train_length} + predict_length={predict_length})"
                )

        # Create the TimeSeriesSequence instance
        return cls(
            data=data,
            time_idx=time_col,
            target_column=target_col,
            train_sequence_length=train_length,
            predict_sequence_length=predict_length,
            batch_size=batch_size,
            group_column=group_cols,
            feature_columns=feature_cols,
            drop_last=drop_last,
            feature_config=feature_config,
            mode=mode,
            stride=stride,
            **kwargs,
        )
