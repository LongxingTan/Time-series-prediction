"""DataProcessor — the "Tokenizer" for time series.

Provides a clean, high-level interface for preparing time series data for
training, validation, and prediction.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .timeseries import TimeSeriesSequence

logger = logging.getLogger(__name__)


class DataProcessor:
    """Unified data preprocessor for time series tasks.

    Wraps TimeSeriesSequence and provides a transformers-like interface.
    Handles sliding windows, normalization, train/valid/test splits
    automatically.

    Args:
        lookback: Number of past time steps used as input.
        horizon: Number of future time steps to predict.
        batch_size: Batch size for tf.data datasets.
        stride: Step size for sliding window (1 = every step, >1 = downsampling).
        normalize: Normalization method — ``'minmax'``, ``'standard'``, or ``None``.
        group_col: Column(s) to group multiple time series.
        feature_cols: Additional feature columns to include.
        fill_missing_dates: Whether to fill gaps in the time index.
        freq: Frequency string (e.g. ``'D'``, ``'H'``) when filling dates.
        validation_split: Fraction of training data to use for validation.
        shuffle: Whether to shuffle the training dataset.
        seed: Random seed for reproducibility.

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=365),
        ...     'sales': np.random.randn(365).cumsum(),
        ... })
        >>> processor = DataProcessor(lookback=30, horizon=7)
        >>> train_ds, valid_ds = processor.prepare(df, target_col='sales')
        >>> for x, y in train_ds.take(1):
        ...     print(x.shape, y.shape)
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 24,
        batch_size: int = 32,
        stride: int = 1,
        normalize: Optional[str] = "minmax",
        group_col: Optional[Union[str, List[str]]] = None,
        feature_cols: Optional[List[str]] = None,
        fill_missing_dates: bool = False,
        freq: Optional[str] = None,
        validation_split: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if normalize not in (None, "minmax", "standard"):
            raise ValueError(f"normalize must be 'minmax', 'standard', or None, got {normalize}")

        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.stride = stride
        self.normalize = normalize
        self.group_col = group_col
        self.feature_cols = feature_cols or []
        self.fill_missing_dates = fill_missing_dates
        self.freq = freq
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed

        # Set during prepare()
        self._scaler_params: Optional[Dict] = None
        self._feature_names: List[str] = []
        self._target_col: Optional[str] = None
        self._time_col: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
        tf.data.Dataset,
    ]:
        """Prepare data and return tf.data.Dataset(s).

        Args:
            df: Input DataFrame.
            target_col: Name of the column to forecast. Auto-detected if None.
            time_col: Name of the time column. Auto-detected if None.

        Returns:
            If ``validation_split > 0``:
                ``(train_ds, valid_ds)``
            If ``validation_split == 0``:
                ``train_ds``
        """
        df = df.copy()

        self._target_col = target_col or self._infer_target(df)
        self._time_col = time_col or self._infer_time(df)

        # Normalize
        if self.normalize is not None:
            df = self._apply_normalization(df)

        # Build sequence
        seq = self._build_sequence(df)
        ds = seq.get_tf_dataset()

        if self.validation_split > 0:
            return self._split_dataset(ds)
        return ds

    def prepare_for_inference(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> tf.data.Dataset:
        """Prepare data for inference (no shuffle, batch_size=1 by default)."""
        original_bs = self.batch_size
        original_shuffle = self.shuffle
        self.batch_size = 1
        self.shuffle = False
        try:
            result = self.prepare(df, target_col=target_col, time_col=time_col)
        finally:
            self.batch_size = original_bs
            self.shuffle = original_shuffle
        # If validation split was used, return train (which contains everything when shuffle=False)
        return result if isinstance(result, tf.data.Dataset) else result[0]

    def inverse_transform(self, values: Union[np.ndarray, tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
        """Reverse the normalization applied during prepare().

        Args:
            values: Normalized predictions or targets.

        Returns:
            Values in the original scale.
        """
        if self._scaler_params is None:
            return values
        min_val = self._scaler_params["min"]
        max_val = self._scaler_params["max"]
        return values * (max_val - min_val) + min_val

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_target(df: pd.DataFrame) -> str:
        """Auto-detect target column (last numeric column)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in DataFrame. Please specify target_col.")
        # Exclude obvious time/index columns
        candidates = [c for c in numeric_cols if not _looks_like_time(df[c])]
        if candidates:
            return candidates[-1]  # usually the value column is last
        return numeric_cols[-1]

    @staticmethod
    def _infer_time(df: pd.DataFrame) -> str:
        """Auto-detect time column."""
        # Check index first
        if isinstance(df.index, pd.DatetimeIndex):
            col_name = df.index.name or "time_idx"
            df[col_name] = df.index
            return col_name
        # Look for datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        # Look for columns named like time
        for name in ("date", "time", "datetime", "timestamp", "ds"):
            if name in df.columns:
                return name
        # Fallback to first column
        return df.columns[0]

    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max or standard normalization to target column."""
        target = self._target_col
        if self.normalize == "minmax":
            min_val = df[target].min()
            max_val = df[target].max()
            df[target] = (df[target] - min_val) / (max_val - min_val + 1e-8)
            self._scaler_params = {"min": min_val, "max": max_val}
        elif self.normalize == "standard":
            mean = df[target].mean()
            std = df[target].std()
            df[target] = (df[target] - mean) / (std + 1e-8)
            self._scaler_params = {"mean": mean, "std": std}
        return df

    def _build_sequence(self, df: pd.DataFrame) -> TimeSeriesSequence:
        """Build a TimeSeriesSequence from the DataFrame."""
        return TimeSeriesSequence(
            data=df,
            time_idx=self._time_col,
            target_column=self._target_col,
            train_sequence_length=self.lookback,
            predict_sequence_length=self.horizon,
            batch_size=self.batch_size,
            group_column=self.group_col,
            feature_columns=self.feature_cols if self.feature_cols else None,
            stride=self.stride,
        )

    def _split_dataset(self, ds: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Split a tf.data.Dataset into train / validation."""
        # Count total batches
        total = sum(1 for _ in ds)
        train_batches = max(1, int(total * (1 - self.validation_split)))

        # Rebuild the source so we can split from a fresh iterator
        ds = ds.unbatch()
        ds = ds.shuffle(buffer_size=10000, seed=self.seed) if self.shuffle else ds
        ds = ds.batch(self.batch_size)

        # Collect into lists for splitting
        all_batches = list(ds.as_numpy_iterator())
        train_ds = tf.data.Dataset.from_tensor_slices(
            (
                np.concatenate([b[0] for b in all_batches[:train_batches]], axis=0),
                np.concatenate([b[1] for b in all_batches[:train_batches]], axis=0),
            )
        )
        train_ds = train_ds.batch(self.batch_size)
        if self.shuffle:
            train_ds = train_ds.shuffle(buffer_size=10000, seed=self.seed)

        if train_batches < len(all_batches):
            valid_x = np.concatenate([b[0] for b in all_batches[train_batches:]], axis=0)
            valid_y = np.concatenate([b[1] for b in all_batches[train_batches:]], axis=0)
            valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(self.batch_size)
            return train_ds, valid_ds

        return train_ds, train_ds  # fallback: use all data


def _looks_like_time(series: pd.Series) -> bool:
    """Heuristic to detect time-like columns."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Check if values look like timestamps or sequential integers
    if series.dtype.kind in "iu":  # integer
        if len(series) > 1 and series.is_monotonic_increasing:
            diffs = series.diff().dropna()
            if diffs.nunique() <= 3:  # regular step
                return True
    return False
