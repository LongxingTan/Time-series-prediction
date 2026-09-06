"""Small, framework-neutral helpers for preparing time-series sequences.

The helpers in this module are deliberately NumPy based.  They borrow the
useful parts of KerasHub's preprocessing utilities (canonical sequence shape,
fixed-length padding, and an explicit validity mask) without importing
KerasHub or TensorFlow Text into TFTS.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._window_geometry import is_regular, sliding_bounds

ArrayLike = Union[np.ndarray, Sequence[float]]


def sequence_mask(lengths: Iterable[int], max_length: Optional[int] = None) -> np.ndarray:
    """Return a boolean mask where ``True`` identifies a valid timestep.

    Args:
        lengths: Number of valid timesteps for each sequence.  Values must be
            non-negative integers.
        max_length: Width of the returned mask.  If omitted, the largest
            length is used.

    Returns:
        A boolean array with shape ``(batch, max_length)``.
    """
    lengths_array = np.asarray(list(lengths))
    if lengths_array.ndim != 1:
        raise ValueError("lengths must be a one-dimensional sequence")
    if not np.issubdtype(lengths_array.dtype, np.integer):
        if not np.all(np.equal(lengths_array, np.floor(lengths_array))):
            raise ValueError("lengths must contain integers")
        lengths_array = lengths_array.astype(np.int64)
    else:
        lengths_array = lengths_array.astype(np.int64, copy=False)
    if np.any(lengths_array < 0):
        raise ValueError("lengths must be non-negative")

    inferred_length = int(lengths_array.max()) if lengths_array.size else 0
    if max_length is None:
        max_length = inferred_length
    if not isinstance(max_length, (int, np.integer)) or max_length < 0:
        raise ValueError("max_length must be a non-negative integer")
    if np.any(lengths_array > max_length):
        raise ValueError("max_length cannot be smaller than a sequence length")
    return np.arange(int(max_length))[None, :] < lengths_array[:, None]


def _as_sequence_array(sequence: ArrayLike, name: str = "sequence") -> np.ndarray:
    """Canonicalize one sequence to ``(timesteps, features)``."""
    try:
        array = np.asarray(sequence)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a dense numeric sequence") from error
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must have rank 1 or 2, got rank {array.ndim}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    return array


def pad_sequence(
    sequence: ArrayLike,
    sequence_length: int,
    padding_side: str = "right",
    pad_value: float = 0.0,
    return_padding_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Trim and pad one sequence to a fixed length.

    ``padding_side`` controls both where padding is inserted and which end is
    retained when the input is longer than ``sequence_length``.  Left padding
    therefore keeps the most recent timesteps, which is the usual forecasting
    behavior for a short context window.

    Args:
        sequence: Rank-1 values or rank-2 ``(timesteps, features)`` values.
        sequence_length: Desired number of timesteps.
        padding_side: Either ``"left"`` or ``"right"``.
        pad_value: Value used for padded rows.
        return_padding_mask: If true, also return a ``(sequence_length,)``
            boolean mask where true means the position contains input data.
    """
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    if not isinstance(sequence_length, (int, np.integer)) or sequence_length < 0:
        raise ValueError("sequence_length must be a non-negative integer")

    array = _as_sequence_array(sequence)
    input_length = len(array)
    sequence_length = int(sequence_length)
    if len(array) > sequence_length:
        if padding_side == "left":
            array = array[-sequence_length:] if sequence_length else array[:0]
        else:
            array = array[:sequence_length]

    pad_rows = sequence_length - len(array)
    if pad_rows:
        padding = np.full((pad_rows, array.shape[1]), pad_value, dtype=array.dtype)
        if padding_side == "left":
            array = np.concatenate((padding, array), axis=0)
        else:
            array = np.concatenate((array, padding), axis=0)

    if not return_padding_mask:
        return array
    valid_length = min(input_length, sequence_length)
    mask = sequence_mask([valid_length], sequence_length)[0]
    if padding_side == "left":
        mask = mask[::-1]
    return array, mask


def pad_sequences(
    sequences: Sequence[ArrayLike],
    sequence_length: Optional[int] = None,
    padding_side: str = "right",
    pad_value: float = 0.0,
    return_padding_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Pad a batch of rank-1 or rank-2 sequences to one common width.

    All sequences must have the same feature dimension.  The result always
    has shape ``(batch, timesteps, features)``; scalar sequences consequently
    receive a trailing feature dimension of one.
    """
    if not isinstance(sequences, (list, tuple)):
        raise ValueError("sequences must be a list or tuple of sequences")
    arrays = [_as_sequence_array(sequence, name=f"sequences[{index}]") for index, sequence in enumerate(sequences)]
    if not arrays:
        if sequence_length is None:
            sequence_length = 0
        if not isinstance(sequence_length, (int, np.integer)) or sequence_length < 0:
            raise ValueError("sequence_length must be a non-negative integer")
        empty = np.empty((0, int(sequence_length), 0), dtype=np.float32)
        mask = np.empty((0, int(sequence_length)), dtype=bool)
        return (empty, mask) if return_padding_mask else empty
    feature_dims = {array.shape[1] for array in arrays}
    if len(feature_dims) != 1:
        raise ValueError("all sequences must have the same feature dimension")
    if sequence_length is None:
        sequence_length = max(len(array) for array in arrays)
    if not isinstance(sequence_length, (int, np.integer)) or sequence_length < 0:
        raise ValueError("sequence_length must be a non-negative integer")

    padded = [
        pad_sequence(
            array,
            int(sequence_length),
            padding_side=padding_side,
            pad_value=pad_value,
            return_padding_mask=return_padding_mask,
        )
        for array in arrays
    ]
    if return_padding_mask:
        values, masks = zip(*padded)
        return np.stack(values), np.stack(masks)
    return np.stack(padded)


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


__all__ = ["generate_sequence_windows", "pad_sequence", "pad_sequences", "sequence_mask"]
