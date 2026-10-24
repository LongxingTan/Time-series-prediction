"""Framework-neutral window indexing for forecasting datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .padding import pad_sequence


@dataclass(frozen=True)
class WindowSpec:
    """Geometry and sampling policy for forecast windows."""

    context_length: int
    prediction_length: int
    stride: int = 1
    mode: str = "train"
    require_regular: bool = True

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0
            for value in (self.context_length, self.prediction_length)
        ):
            raise ValueError("context_length and prediction_length must be positive")
        if not isinstance(self.stride, (int, np.integer)) or isinstance(self.stride, bool) or self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.mode not in {"train", "predict"}:
            raise ValueError("mode must be either 'train' or 'predict'")


@dataclass(frozen=True)
class Window:
    """Row positions for one encoder/decoder sample."""

    group_key: Tuple[Any, ...]
    encoder_indices: Sequence[int]
    decoder_indices: Sequence[int]


class WindowIndex:
    """Compact immutable window table; Window objects are created only on access."""

    def __init__(self, spec, windows):
        self.spec = spec
        starts, group_ids, lengths, groups = [], [], [], {}
        for window in windows:
            encoder, decoder = window.encoder_indices, window.decoder_indices
            if len(encoder) != spec.context_length:
                raise ValueError("Window context does not match WindowSpec")
            if any(int(value) != int(encoder[0]) + i for i, value in enumerate(encoder)):
                raise ValueError("Window rows must be contiguous; sort by group and time first")
            if any(int(value) != int(encoder[-1]) + 1 + i for i, value in enumerate(decoder)):
                raise ValueError("Decoder must immediately follow the context")
            starts.append(encoder[0])
            group_ids.append(groups.setdefault(window.group_key, len(groups)))
            lengths.append(len(decoder))
        self.group_keys = tuple(groups)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.group_ids = np.asarray(group_ids, dtype=np.int32)
        self.decoder_lengths = np.asarray(lengths, dtype=np.int32)
        for array in (self.starts, self.group_ids, self.decoder_lengths):
            array.flags.writeable = False

    @classmethod
    def from_arrays(cls, spec, group_keys, starts, group_ids, decoder_lengths):
        result = cls.__new__(cls)
        result.spec, result.group_keys = spec, tuple(tuple(key) for key in group_keys)
        result.starts = np.array(starts, dtype=np.int64, copy=True)
        result.group_ids = np.array(group_ids, dtype=np.int32, copy=True)
        result.decoder_lengths = np.array(decoder_lengths, dtype=np.int32, copy=True)
        if (
            result.starts.ndim != 1
            or result.starts.shape != result.group_ids.shape
            or result.starts.shape != result.decoder_lengths.shape
        ):
            raise ValueError("Window index columns must be equal-length one-dimensional arrays")
        if (
            (result.starts < 0).any()
            or (result.group_ids < 0).any()
            or (result.group_ids >= len(result.group_keys)).any()
        ):
            raise ValueError("Invalid window starts or group ids")
        if (result.decoder_lengths < 0).any() or (result.decoder_lengths > spec.prediction_length).any():
            raise ValueError("Invalid decoder lengths")
        for array in (result.starts, result.group_ids, result.decoder_lengths):
            array.flags.writeable = False
        return result

    def select(self, selection):
        return type(self).from_arrays(
            self.spec,
            self.group_keys,
            self.starts[selection],
            self.group_ids[selection],
            self.decoder_lengths[selection],
        )

    @property
    def windows(self):
        return self

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.select(index)
        start = int(self.starts[index])
        middle = start + self.spec.context_length
        return Window(
            self.group_keys[self.group_ids[index]],
            range(start, middle),
            range(middle, middle + int(self.decoder_lengths[index])),
        )

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class WindowIndexer:
    """Build compact indices from NumPy row groups, independent of dataframes."""

    def build(self, row_groups, spec, *, time_values=None, target_observed=None, required_history=0):
        if not isinstance(required_history, (int, np.integer)) or required_history < 0:
            raise ValueError("required_history must be a nonnegative integer")
        starts, ids, lengths, keys = [], [], [], []
        times = None if time_values is None else np.asarray(time_values)
        observed = None if target_observed is None else np.asarray(target_observed, dtype=bool)
        for key, rows in row_groups:
            rows = np.asarray(rows, dtype=np.int64)
            if rows.ndim != 1 or not len(rows):
                raise ValueError("Each group requires a nonempty one-dimensional row index")
            if (rows < 0).any() or (np.diff(rows) != 1).any():
                raise ValueError("Group rows must be contiguous and sorted")
            group_id = len(keys)
            keys.append(tuple(key))
            context, horizon = spec.context_length, spec.prediction_length
            if spec.mode == "train":
                local = np.arange(required_history, len(rows) - context - horizon + 1, spec.stride, dtype=np.int64)
                decoder_lengths = np.full(len(local), horizon, dtype=np.int32)
            else:
                valid = np.flatnonzero(observed[rows]) if observed is not None else np.arange(len(rows))
                if not len(valid) or valid[-1] + 1 < context + required_history:
                    continue
                cutoff = int(valid[-1])
                local = np.array([cutoff - context + 1], dtype=np.int64)
                decoder_lengths = np.array([min(horizon, len(rows) - cutoff - 1)], dtype=np.int32)
            if spec.require_regular and times is not None:
                keep = np.array(
                    [
                        is_regular(times[rows[start : start + context + length]])
                        for start, length in zip(local, decoder_lengths)
                    ],
                    dtype=bool,
                )
                local, decoder_lengths = local[keep], decoder_lengths[keep]
            starts.append(rows[local])
            ids.append(np.full(len(local), group_id, dtype=np.int32))
            lengths.append(decoder_lengths)
        if not starts or not any(len(values) for values in starts):
            raise ValueError("no valid windows could be constructed")
        return WindowIndex.from_arrays(spec, keys, np.concatenate(starts), np.concatenate(ids), np.concatenate(lengths))


def sliding_bounds(length: int, width: int, stride: int = 1, start: int = 0) -> Iterator[Tuple[int, int]]:
    """Yield half-open bounds for every complete fixed-width window."""
    for left in range(start, length - width + 1, stride):
        yield left, left + width


def is_regular(values):
    values = np.asarray(values)
    if len(values) < 2:
        return True
    differences = np.diff(values)
    zero = np.timedelta64(0, "ns") if np.issubdtype(differences.dtype, np.timedelta64) else 0
    return bool(np.all(differences > zero) and np.all(differences == differences[0]))


def generate_sequence_windows(
    length: int,
    context_length: int,
    prediction_length: int,
    stride: int = 1,
    mode: str = "train",
    time_values: Optional[Iterable] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate valid encoder/decoder index pairs for one time series.

    ``mode="inference"`` creates every complete context window and an empty
    decoder index array.  Other modes create complete context-plus-target
    windows.  If ``time_values`` is provided, windows crossing an irregular
    time step are skipped.
    """
    if not isinstance(length, (int, np.integer)) or length < 0:
        raise ValueError("length must be a non-negative integer")
    if not isinstance(context_length, (int, np.integer)) or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    if not isinstance(prediction_length, (int, np.integer)) or prediction_length < 1:
        raise ValueError("prediction_length must be a positive integer")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if mode not in {"train", "validation", "test", "inference"}:
        raise ValueError("mode must be one of 'train', 'validation', 'test', or 'inference'")

    if time_values is None:
        time_array = None
    else:
        time_array = np.asarray(list(time_values))
        if len(time_array) != length:
            raise ValueError("time_values must have the same length as the series")

    window_width = context_length if mode == "inference" else context_length + prediction_length
    windows = []
    empty_decoder = np.empty(0, dtype=np.int64)
    for start, stop in sliding_bounds(length, window_width, int(stride)):
        encoder = np.arange(start, start + context_length, dtype=np.int64)
        decoder = empty_decoder if mode == "inference" else np.arange(start + context_length, stop, dtype=np.int64)
        indices = np.concatenate((encoder, decoder))
        if time_array is not None and not is_regular(time_array[indices]):
            continue
        windows.append((encoder, decoder))
    return windows


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
