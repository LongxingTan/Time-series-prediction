"""DataProcessor — the "Tokenizer" for time series.

Provides a clean, high-level interface for preparing time series data for
training, validation, and prediction.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.features import FeatureRole, FeatureSpec, TimeSeriesSchema

from .frame import FramePreprocessor

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
        fill_missing_dates: Reserved; regularize missing dates explicitly before prepare().
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
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if not 0 <= validation_split < 1:
            raise ValueError(f"validation_split must be in [0, 1), got {validation_split}")
        if normalize not in (None, "minmax", "standard"):
            raise ValueError(f"normalize must be 'minmax', 'standard', or None, got {normalize}")
        if fill_missing_dates:
            raise ValueError("Regularize missing dates explicitly before prepare()")

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
            - If ``validation_split > 0``: a tuple ``(train_ds, valid_ds)``.
            - If ``validation_split == 0``: a single ``train_ds``.
        """
        df = df.copy()

        self._target_col = target_col or self._infer_target(df)
        self._time_col = time_col or self._infer_time(df)

        groups = (
            ()
            if self.group_col is None
            else (self.group_col,) if isinstance(self.group_col, str) else tuple(self.group_col)
        )
        schema = TimeSeriesSchema(
            self._time_col,
            (self._target_col,),
            tuple(
                FeatureSpec(name, FeatureRole.OBSERVED_PAST) for name in self.feature_cols if name != self._target_col
            ),
            groups,
        )
        self._processor = FramePreprocessor(schema, self.lookback, self.horizon, self.normalize, freq=self.freq)
        prepared, train_windows, valid_windows = self._processor.fit(df, self.validation_split)
        if self.stride > 1:
            from dataclasses import replace

            train_windows = replace(train_windows, windows=train_windows.windows[:: self.stride])
            if valid_windows is not None:
                valid_windows = replace(valid_windows, windows=valid_windows.windows[:: self.stride])
        # Preserve the historical sample-standard-deviation convention at this adapter.
        scaler = self._processor.target_scaler
        if self.normalize == "standard":
            training = self._normalization_fit_frame(df.sort_values(list(groups) + [self._time_col]))
            scale = float(training[self._target_col].std())
            previous_scale = scaler.scale.copy()
            scaler.scale = np.asarray([scale if scale > 1e-8 else 1.0], dtype=np.float32)
            prepared.frame[self._target_col] *= previous_scale[0] / scaler.scale[0]
        if self.normalize == "standard":
            self._scaler_params = {
                "mean": float(training[self._target_col].mean()),
                "std": scale if scale > 1e-8 else 1.0,
            }
        elif self.normalize == "minmax":
            self._scaler_params = {"min": float(scaler.offset[0]), "max": float(scaler.offset[0] + scaler.scale[0])}
        train = self._legacy_dataset(
            self._processor.dataset(
                prepared, train_windows, batch_size=self.batch_size, shuffle=self.shuffle, seed=self.seed
            )
        )
        if valid_windows is None:
            return train
        valid = self._legacy_dataset(self._processor.dataset(prepared, valid_windows, batch_size=self.batch_size))
        return train, valid

    def prepare_for_inference(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> tf.data.Dataset:
        """Prepare data for inference (no shuffle, batch_size=1 by default)."""
        if not hasattr(self, "_processor"):
            raise RuntimeError("DataProcessor must be fitted with prepare() before inference")
        if target_col not in (None, self._target_col) or time_col not in (None, self._time_col):
            raise ValueError("Inference columns must match the fitted schema")
        prepared, windows, _ = self._processor.prediction(df)
        return self._legacy_dataset(self._processor.dataset(prepared, windows, batch_size=1), inference=True)

    def _legacy_dataset(self, dataset, inference=False):
        def convert(inputs, labels=None):
            values = inputs["past_values"]
            if "past_time_features" in inputs:
                values = tf.concat([values, inputs["past_time_features"]], axis=-1)
            if labels is None:
                labels = tf.zeros([tf.shape(values)[0], self.horizon, 1], dtype=values.dtype)
            return values, labels

        return dataset.map(convert)

    def inverse_transform(self, values: Union[np.ndarray, tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
        """Reverse the normalization applied during prepare().

        Args:
            values: Normalized predictions or targets.

        Returns:
            Values in the original scale.
        """
        if self._scaler_params is None:
            return values
        if self.normalize == "standard":
            return values * self._scaler_params["std"] + self._scaler_params["mean"]
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

    def _normalization_fit_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the chronological training portion used to fit scaling."""
        if self.validation_split <= 0:
            return df.copy()
        if self.group_col:
            group_cols = [self.group_col] if isinstance(self.group_col, str) else self.group_col
            parts = []
            for _, group in df.groupby(group_cols, observed=True, sort=False):
                split_idx = max(1, int(len(group) * (1 - self.validation_split)))
                parts.append(group.iloc[:split_idx])
            return pd.concat(parts, axis=0).copy()
        split_idx = max(1, int(len(df) * (1 - self.validation_split)))
        return df.iloc[:split_idx].copy()

    def _split_dataset(self, ds):
        raise ValueError("Split raw rows with prepare(); splitting overlapping windows is unsupported")


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
