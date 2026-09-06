"""Authoritative framework-neutral window geometry primitives."""

from typing import Iterable, Iterator, Tuple

import numpy as np
import pandas as pd


def sliding_bounds(length: int, width: int, stride: int = 1, start: int = 0) -> Iterator[Tuple[int, int]]:
    """Yield half-open bounds for every complete fixed-width window."""
    for left in range(start, length - width + 1, stride):
        yield left, left + width


def is_regular(values: Iterable) -> bool:
    """Return whether consecutive numeric or datetime values have one spacing."""
    values = pd.Series(values)
    if len(values) < 3:
        return True
    if pd.api.types.is_datetime64_any_dtype(values):
        numeric = values.astype("datetime64[ns]").astype("int64").to_numpy()
    else:
        numeric = values.to_numpy()
    differences = np.diff(numeric)
    return bool(np.all(differences == differences[0]))
