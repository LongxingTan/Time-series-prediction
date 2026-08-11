"""AutoFeatureEngineer — sklearn-style automatic feature engineering for time series.

Wraps the existing feature functions (lag, rolling, datetime, …) behind a
single class with fit / transform semantics so users don't need to call
individual functions manually.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .datetime_feature import add_datetime_feature
from .one_order_feature import add_lag_feature, add_moving_average_feature, add_roll_feature, add_transform_feature
from .registry import FeatureRegistry

logger = logging.getLogger(__name__)


class AutoFeatureEngineer:
    """Automatic feature engineering for time series DataFrames.

    Generates lag features, rolling-window features, datetime features,
    and optional Fourier terms behind a simple ``fit_transform`` API.
    Tracks generated feature names via an internal
    :class:`~tfts.features.registry.FeatureRegistry`.

    Args:
        lags: Lag offsets to generate (e.g. ``[1, 7, 14]``).
        windows: Rolling window sizes (e.g. ``[7, 30]``).
        rolling_functions: Aggregations for rolling windows.
            Defaults to ``['mean', 'std']`` for speed.  Use ``'all'``
            to get ``['mean', 'std', 'min', 'max']``.
        add_datetime: If ``True``, add calendar features (month, day-of-
            week, …).
        datetime_features: Specific datetime features to generate.
            ``None`` means ``['month', 'dayofweek', 'hour']`` (where
            applicable).
        add_fourier: If ``True``, add cyclical (sin/cos) features for
            month and day-of-week.
        group_cols: Columns to group by (for multi-series data).

    Examples:
        >>> from tfts.features import AutoFeatureEngineer
        >>> eng = AutoFeatureEngineer(lags=[1, 7], windows=[7, 30],
        ...                            add_datetime=True)
        >>> df_feat = eng.fit_transform(df, time_col="date", target_col="value")
        >>> eng.get_feature_names()[:3]
        ['value_lag_1', 'value_lag_7', 'value_roll_7_mean']
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
        rolling_functions: Union[str, List[str]] = "default",
        add_datetime: bool = False,
        datetime_features: Optional[List[str]] = None,
        add_fourier: bool = False,
        group_cols: Optional[List[str]] = None,
    ) -> None:
        self.lags = lags or [1, 7]
        self.windows = windows or [7]
        self.rolling_functions = rolling_functions
        self.add_datetime = add_datetime
        self.datetime_features = datetime_features
        self.add_fourier = add_fourier
        self.group_cols = group_cols

        self._registry = FeatureRegistry()
        self._original_columns: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, time_col: str, target_col: str) -> "AutoFeatureEngineer":
        """Fit the engineer — records original columns so ``transform``
        can later identify which columns are new features.

        Args:
            df: Training DataFrame.
            time_col: Name of the time column.
            target_col: Name of the target column.

        Returns:
            self (for chaining).
        """
        self._time_col = time_col
        self._target_col = target_col
        self._original_columns = df.columns.tolist()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to *df*.

        Must be called after :meth:`fit`.

        Args:
            df: DataFrame with the same time/target columns used in
                :meth:`fit`.

        Returns:
            DataFrame with new feature columns appended.
        """
        self._check_fitted()
        result = df.copy()

        # 1. Lag features
        result = add_lag_feature(
            data=result,
            columns=self._target_col,
            lags=self.lags,
            time_col=self._time_col,
            group_cols=self.group_cols,
        )

        # 2. Rolling features
        roll_funcs = self._resolve_rolling_functions()
        result = add_roll_feature(
            data=result,
            columns=self._target_col,
            windows=self.windows,
            functions=roll_funcs,
            time_col=self._time_col,
            group_cols=self.group_cols,
        )

        # 3. Datetime features
        if self.add_datetime:
            features = self.datetime_features
            if features is None:
                # Pick sensible defaults depending on the data's resolution
                features = _default_datetime_features(result[self._time_col])
            result = add_datetime_feature(data=result, time_col=self._time_col, features=features)

        # 4. Fourier (cyclical) features
        if self.add_fourier:
            fourier_features = _default_fourier_features()
            result = add_datetime_feature(data=result, time_col=self._time_col, features=fourier_features)

        # Register the new feature columns
        new_cols = [c for c in result.columns if c not in self._original_columns]
        self._registry.register(new_cols)

        # Drop rows with NaN introduced by lag/rolling
        result = result.dropna().reset_index(drop=True)

        logger.info(f"AutoFeatureEngineer added {len(new_cols)} features")
        return result

    def fit_transform(self, df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df, time_col, target_col).transform(df)

    def get_feature_names(self) -> List[str]:
        """Return names of the generated features (excluding originals)."""
        return self._registry.get_features()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_rolling_functions(self) -> List[str]:
        if self.rolling_functions == "all":
            return ["mean", "std", "min", "max"]
        if self.rolling_functions == "default":
            return ["mean", "std"]
        if isinstance(self.rolling_functions, str):
            return [self.rolling_functions]
        return list(self.rolling_functions)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("AutoFeatureEngineer is not fitted yet. Call .fit() or .fit_transform() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"AutoFeatureEngineer(lags={self.lags}, windows={self.windows}, "
            f"add_datetime={self.add_datetime}, add_fourier={self.add_fourier}, {status})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_datetime_features(time_series: pd.Series) -> List[str]:
    """Pick sensible datetime features based on the time resolution."""
    if not pd.api.types.is_datetime64_any_dtype(time_series):
        return ["month", "dayofweek"]

    sample = time_series.dropna().iloc[:3]
    has_sub_daily = any(ts.hour != 0 or ts.minute != 0 for ts in sample if pd.notna(ts))
    features = ["month", "dayofweek", "day"]
    if has_sub_daily:
        features.append("hour")
    return features


def _default_fourier_features() -> List[str]:
    """Default cyclical (sin/cos) features."""
    return ["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]
