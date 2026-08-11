"""AutoPreprocessor — sklearn-style data preprocessing for time series.

Provides a single class that handles missing values, outlier clipping,
and normalization with fit / transform / inverse_transform semantics.
Fitted parameters are stored so that transform can be applied to new
data (e.g. inference) and inverse_transform can return predictions
to the original scale.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AutoPreprocessor:
    """Automatic data preprocessor for time series DataFrames.

    Handles three common preprocessing steps in order:
    1. Missing values  (``handle_missing``)
    2. Outliers        (``handle_outliers``)
    3. Normalization   (``normalize``)

    Follow the sklearn convention: call :meth:`fit_transform` on training
    data, then :meth:`transform` on test / inference data, and
    :meth:`inverse_transform` on predictions to get the original scale.

    Args:
        handle_missing: How to fill NaN values.
            - ``'ffill'``: forward fill, then back-fill leading NaNs.
            - ``'interpolate'``: linear interpolation, then edge-fill.
            - ``'drop'``: drop rows containing NaN.
            - ``None``: skip.
        handle_outliers: How to cap extreme values.
            - ``'clip'``: clip to ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`` per column.
            - ``None``: skip.
        normalize: Normalization method.
            - ``'standard'``: ``(x - mean) / std``.
            - ``'minmax'``: ``(x - min) / (max - min)``.
            - ``None``: skip.
        columns: Column names to process.  If ``None``, all numeric columns
            are selected during :meth:`fit`.

    Examples:
        >>> from tfts.data import AutoPreprocessor
        >>> pre = AutoPreprocessor(handle_missing="interpolate",
        ...                         handle_outliers="clip",
        ...                         normalize="standard")
        >>> df_clean = pre.fit_transform(df)
        >>> df_orig = pre.inverse_transform(df_clean)
    """

    _VALID_MISSING = ("ffill", "interpolate", "drop", None)
    _VALID_OUTLIERS = ("clip", None)
    _VALID_NORMALIZE = ("standard", "minmax", None)

    def __init__(
        self,
        handle_missing: Optional[str] = "ffill",
        handle_outliers: Optional[str] = None,
        normalize: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        if handle_missing not in self._VALID_MISSING:
            raise ValueError(f"handle_missing must be one of {self._VALID_MISSING}, got {handle_missing!r}")
        if handle_outliers not in self._VALID_OUTLIERS:
            raise ValueError(f"handle_outliers must be one of {self._VALID_OUTLIERS}, got {handle_outliers!r}")
        if normalize not in self._VALID_NORMALIZE:
            raise ValueError(f"normalize must be one of {self._VALID_NORMALIZE}, got {normalize!r}")

        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.normalize = normalize
        self.columns = columns

        # Fitted parameters (populated by fit)
        self._fitted_columns: List[str] = []
        self._clip_bounds: Dict[str, tuple] = {}
        self._norm_params: Dict[str, dict] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "AutoPreprocessor":
        """Fit the preprocessor on *df* (learn clip bounds & norm params).

        Args:
            df: Training DataFrame.

        Returns:
            self (for chaining).
        """
        cols = self.columns if self.columns is not None else df.select_dtypes(include=[np.number]).columns.tolist()
        self._fitted_columns = list(cols)

        # Learn outlier bounds
        if self.handle_outliers == "clip":
            for col in self._fitted_columns:
                if col not in df.columns:
                    continue
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                self._clip_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # Learn normalization parameters on the *already cleaned* data
        # so that outliers don't skew mean/std. We apply the same
        # cleaning steps first, then compute stats.
        df_clean = self._fill_missing(df)
        df_clean = self._clip_outliers(df_clean)

        if self.normalize == "standard":
            for col in self._fitted_columns:
                if col not in df_clean.columns:
                    continue
                self._norm_params[col] = {"mean": df_clean[col].mean(), "std": df_clean[col].std()}
        elif self.normalize == "minmax":
            for col in self._fitted_columns:
                if col not in df_clean.columns:
                    continue
                self._norm_params[col] = {"min": df_clean[col].min(), "max": df_clean[col].max()}

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to *df*.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed copy of *df*.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._check_fitted()
        result = df.copy()
        result = self._fill_missing(result)
        result = self._clip_outliers(result)
        result = self._apply_normalization(result)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse the normalization step only.

        Missing-value filling and outlier clipping are not reversible,
        so only the normalization is undone.

        Args:
            df: Normalized DataFrame (or a subset of columns).

        Returns:
            DataFrame with normalization reversed.
        """
        self._check_fitted()
        result = df.copy()
        for col in self._fitted_columns:
            if col not in result.columns or col not in self._norm_params:
                continue
            params = self._norm_params[col]
            if self.normalize == "standard":
                result[col] = result[col] * params["std"] + params["mean"]
            elif self.normalize == "minmax":
                result[col] = result[col] * (params["max"] - params["min"]) + params["min"]
        return result

    def get_fitted_columns(self) -> List[str]:
        """Return the list of columns this preprocessor was fitted on."""
        return list(self._fitted_columns)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the fitted columns only."""
        if self.handle_missing is None:
            return df

        if self.handle_missing == "ffill":
            result = df.copy()
            columns = self._fitted_columns or list(result.columns)
            result[columns] = result[columns].ffill().bfill()
            return result
        elif self.handle_missing == "interpolate":
            result = df.copy()
            result[self._fitted_columns] = (
                result[self._fitted_columns].interpolate(limit_direction="both").bfill().ffill()
            )
            return result
        elif self.handle_missing == "drop":
            return df.dropna(subset=self._fitted_columns if self._fitted_columns else None)
        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using fitted bounds."""
        if self.handle_outliers != "clip":
            return df

        for col, (lower, upper) in self._clip_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
        return df

    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize using fitted parameters."""
        if self.normalize is None:
            return df

        for col in self._fitted_columns:
            if col not in df.columns or col not in self._norm_params:
                continue
            params = self._norm_params[col]
            if self.normalize == "standard":
                df[col] = (df[col] - params["mean"]) / (params["std"] + 1e-8)
            elif self.normalize == "minmax":
                df[col] = (df[col] - params["min"]) / (params["max"] - params["min"] + 1e-8)
        return df

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("AutoPreprocessor is not fitted yet. Call .fit() or .fit_transform() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"AutoPreprocessor(handle_missing={self.handle_missing!r}, "
            f"handle_outliers={self.handle_outliers!r}, "
            f"normalize={self.normalize!r}, {status})"
        )
