"""Metrics computation for the TFTS benchmark system.

Supports standard time-series forecasting metrics, with both NumPy and
tf.keras metric implementations."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class BenchmarkMetrics:
    """Compute and manage time-series forecasting metrics.

    Attributes:
        metrics: List of metric names to compute.
    """

    AVAILABLE_METRICS = {
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "r2",
        "mape_pct",
    }

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or ["mae", "rmse", "mape"]
        invalid = set(self.metrics) - self.AVAILABLE_METRICS
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}. " f"Available: {self.AVAILABLE_METRICS}")

    def compute(
        self,
        y_true: Union[np.ndarray, tf.Tensor],
        y_pred: Union[np.ndarray, tf.Tensor],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute all requested metrics between y_true and y_pred.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            metrics: Optional subset of metrics to compute (uses self.metrics if None).

        Returns:
            Dictionary mapping metric name to float value.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        to_compute = metrics if metrics is not None else self.metrics
        results: Dict[str, float] = {}
        for metric in to_compute:
            fn = getattr(self, metric, None)
            if fn is None:
                logger.warning("Unknown metric: %s", metric)
                continue
            try:
                results[metric] = float(fn(y_true, y_pred))
            except Exception as exc:
                logger.warning("Metric %s failed: %s", metric, exc)
                results[metric] = float("nan")
        return results

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error (avoids division by zero)."""
        mask = y_true != 0
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        mask = denom != 0
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return float("nan")
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def mape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error as a percentage (0-100 scale)."""
        return BenchmarkMetrics.mape(y_true, y_pred)
