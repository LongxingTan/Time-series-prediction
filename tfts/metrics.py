"""Time series evaluation metrics.

Standard point-forecast metrics usable with numpy arrays or TensorFlow tensors.
"""

from typing import Union

import numpy as np
import tensorflow as tf


def mse(y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
    """Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Scalar or array of MSE values.
    """
    return _reduce(np.square(_sub(y_true, y_pred)))


def mae(y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
    """Mean Absolute Error."""
    return _reduce(np.abs(_sub(y_true, y_pred)))


def rmse(y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
    """Root Mean Squared Error."""
    return _sqrt(mse(y_true, y_pred))


def mape(
    y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor], eps: float = 1e-8
) -> Union[np.ndarray, tf.Tensor]:
    """Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        eps: Small constant to avoid division by zero.

    Returns:
        MAPE as a percentage (0-100 scale).
    """
    backend = _backend(y_true)
    denominator = backend.maximum(backend.abs(y_true), backend.array(eps, dtype=y_true.dtype))
    return 100.0 * _reduce(backend.abs(_sub(y_true, y_pred)) / denominator)


def smape(
    y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor], eps: float = 1e-8
) -> Union[np.ndarray, tf.Tensor]:
    """Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        eps: Small constant to avoid division by zero.

    Returns:
        SMAPE as a percentage (0-200 scale).
    """
    backend = _backend(y_true)
    numerator = backend.abs(_sub(y_true, y_pred))
    denominator = (backend.abs(y_true) + backend.abs(y_pred)) / 2.0 + backend.array(eps, dtype=y_true.dtype)
    return 100.0 * _reduce(numerator / denominator)


def r2_score(
    y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor]
) -> Union[np.ndarray, tf.Tensor]:
    """R² coefficient of determination."""
    backend = _backend(y_true)
    ss_res = backend.sum(backend.square(_sub(y_true, y_pred)))
    ss_tot = backend.sum(backend.square(_sub(y_true, backend.mean(y_true))))
    return 1.0 - ss_res / ss_tot


def evaluate(
    y_true: Union[np.ndarray, tf.Tensor],
    y_pred: Union[np.ndarray, tf.Tensor],
    metrics: Union[str, list] = "all",
) -> dict:
    """Evaluate predictions with one or more metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        metrics: Metric name, list of names, or "all" for all metrics.

    Returns:
        Dictionary mapping metric names to their values.
    """
    _METRICS = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2_score,
    }
    if metrics == "all":
        names = list(_METRICS.keys())
    elif isinstance(metrics, str):
        names = [metrics]
    else:
        names = metrics

    results = {}
    for name in names:
        if name not in _METRICS:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(_METRICS.keys())}")
        results[name] = float(_METRICS[name](y_true, y_pred))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _NumpyBackend:
    @staticmethod
    def array(x, dtype=None):
        return np.array(x, dtype=dtype)

    abs = staticmethod(np.abs)
    square = staticmethod(np.square)
    sqrt = staticmethod(np.sqrt)
    maximum = staticmethod(np.maximum)
    sum = staticmethod(np.sum)
    mean = staticmethod(np.mean)


class _TfBackend:
    @staticmethod
    def array(x, dtype=None):
        return tf.constant(x, dtype=dtype)

    abs = staticmethod(tf.abs)
    square = staticmethod(tf.square)
    sqrt = staticmethod(tf.sqrt)
    maximum = staticmethod(tf.maximum)
    sum = staticmethod(tf.reduce_sum)
    mean = staticmethod(tf.reduce_mean)


def _backend(x):
    return _TfBackend if isinstance(x, tf.Tensor) else _NumpyBackend


def _sub(a, b):
    return a - b


def _sqrt(x):
    backend = _backend(x)
    return backend.sqrt(x)


def _reduce(x):
    backend = _backend(x)
    return backend.mean(x)
