"""Random training windows and final context windows for array histories."""

from typing import Sequence, Tuple

import numpy as np

from .sequence_utils import pad_sequence

__all__ = ["final_windows", "sampled_windows"]


def _history_values(series, num_features):
    values = np.asarray(series, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[-1] < num_features:
        raise ValueError("each history must have shape (time, features) with at least num_features columns")
    return values[:, :num_features]


def final_windows(
    histories: Sequence[np.ndarray], seq_len: int = 26, num_features: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Last ``seq_len`` window of every series (the held-out test).

    Returns ``(values, mask)`` each of shape ``(n_series, seq_len, num_features)``.
    Series shorter than ``seq_len`` are left-padded and masked.
    """
    values = np.zeros((len(histories), seq_len, num_features), np.float32)
    mask = np.zeros_like(values)
    for index, series in enumerate(histories):
        values[index], valid = pad_sequence(
            _history_values(series, num_features), seq_len, padding_side="left", return_padding_mask=True
        )
        mask[index] = valid[:, None]
    return values, mask


def sampled_windows(
    histories: Sequence[np.ndarray],
    rng: np.random.Generator,
    seq_len: int = 26,
    pred_len: int = 13,
    history_size: int = 10,
    num_features: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One random lookback/forecast window per series, drawn from ``rng``.

    Returns ``(x, y, y_mask)`` of shapes ``(n, seq_len, f)``, ``(n, pred_len, f)``,
    ``(n, pred_len, f)`` (mask marks the valid/non-padded forecast steps).
    """
    x = np.zeros((len(histories), seq_len, num_features), np.float32)
    y = np.zeros((len(histories), pred_len, num_features), np.float32)
    y_mask = np.zeros_like(y)
    for index, series in enumerate(histories):
        series = _history_values(series, num_features)
        if len(series) < 2:
            raise ValueError("random windows require at least two timesteps per history")
        cutoff = int(rng.integers(max(1, len(series) - history_size * pred_len), len(series)))
        x[index] = pad_sequence(series[max(0, cutoff - seq_len) : cutoff], seq_len, padding_side="left")
        y[index], valid = pad_sequence(series[cutoff:], pred_len, padding_side="right", return_padding_mask=True)
        y_mask[index] = valid[:, None]
    return x, y, y_mask
