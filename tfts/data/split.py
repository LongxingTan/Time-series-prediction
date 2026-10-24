"""Temporal splits expressed in raw rows and forecast label intervals."""

from dataclasses import dataclass

import numpy as np

from .regularize import groups


@dataclass(frozen=True)
class TemporalSplit:
    validation: float = 0.2
    gap: int = 0

    def __post_init__(self):
        if not 0 <= self.validation < 1:
            raise ValueError("validation must be a fraction in [0, 1)")
        if not isinstance(self.gap, int) or self.gap < 0:
            raise ValueError("gap must be a nonnegative number of time steps")

    def rows(self, frame, group_cols=()):
        cutoffs, training = {}, []
        for key, positions in groups(frame, group_cols):
            count = int(len(positions) * (1 - self.validation)) if self.validation else len(positions)
            cutoff = int(positions[0]) + count
            cutoffs[key] = cutoff
            training.extend(positions[: max(0, count - self.gap)])
        return np.asarray(training, dtype=np.int64), cutoffs

    def windows(self, windows, cutoffs):
        cutoffs = np.asarray([cutoffs[key] for key in windows.group_keys], dtype=np.int64)[windows.group_ids]
        starts = windows.starts + windows.spec.context_length
        stops = starts + windows.decoder_lengths
        train = windows.select(stops <= cutoffs - self.gap)
        valid = windows.select(starts >= cutoffs) if self.validation else None
        return train, valid
