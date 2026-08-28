"""Framework-neutral window indexing for forecasting datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd

from tfts.features.pipeline import PreparedTimeSeries


@dataclass(frozen=True)
class WindowSpec:
    """Geometry and sampling policy for forecast windows."""

    context_length: int
    prediction_length: int
    stride: int = 1
    mode: str = "train"
    require_regular: bool = True

    def __post_init__(self) -> None:
        if self.context_length <= 0 or self.prediction_length <= 0:
            raise ValueError("context_length and prediction_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.mode not in {"train", "predict"}:
            raise ValueError("mode must be either 'train' or 'predict'")


@dataclass(frozen=True)
class Window:
    """Row positions for one encoder/decoder sample."""

    group_key: Tuple[Any, ...]
    encoder_indices: Tuple[int, ...]
    decoder_indices: Tuple[int, ...]


@dataclass(frozen=True)
class WindowIndex:
    """Reusable sample boundaries shared by every materializer."""

    spec: WindowSpec
    windows: Tuple[Window, ...]

    def __len__(self) -> int:
        return len(self.windows)


class WindowIndexer:
    """Build sliding or final-prediction windows over a prepared frame."""

    def build(self, prepared: PreparedTimeSeries, spec: WindowSpec) -> WindowIndex:
        frame = prepared.frame
        schema = prepared.schema
        schema.validate_frame(frame)
        groups = self._groups(frame, schema.group_cols)
        windows: List[Window] = []
        for key, positions in groups:
            group = frame.iloc[positions]
            if spec.mode == "train":
                windows.extend(
                    self._training_windows(
                        frame, group, positions, key, schema, spec, prepared.manifest.required_history
                    )
                )
            else:
                prediction = self._prediction_window(
                    frame, group, positions, key, schema, spec, prepared.manifest.required_history
                )
                if prediction is not None:
                    windows.append(prediction)
        if not windows:
            raise ValueError("no valid windows could be constructed")
        return WindowIndex(spec, tuple(windows))

    @staticmethod
    def _groups(frame: pd.DataFrame, group_cols: Tuple[str, ...]):
        if not group_cols:
            return [((), np.arange(len(frame), dtype=int))]
        grouped = frame.groupby(list(group_cols), observed=True, sort=False).indices
        return [(_as_tuple(key), np.asarray(positions, dtype=int)) for key, positions in grouped.items()]

    def _training_windows(self, frame, group, positions, key, schema, spec, required_history):
        width = spec.context_length + spec.prediction_length
        windows = []
        # Generated lag/rolling columns use the leading rows as warm-up. A
        # materialized encoder must begin after that warm-up boundary.
        for start in range(required_history, len(group) - width + 1, spec.stride):
            local = positions[start : start + width]
            if spec.require_regular and not _is_regular(frame.iloc[local][schema.time_col]):
                continue
            decoder = local[spec.context_length :]
            if frame.iloc[decoder][list(schema.target_cols)].isna().any(axis=None):
                continue
            windows.append(
                Window(
                    key,
                    tuple(int(value) for value in local[: spec.context_length]),
                    tuple(int(value) for value in decoder),
                )
            )
        return windows

    def _prediction_window(self, frame, group, positions, key, schema, spec, required_history):
        target_observed = ~group[list(schema.target_cols)].isna().any(axis=1)
        observed_positions = np.flatnonzero(target_observed.to_numpy())
        if len(observed_positions) < spec.context_length:
            return None
        cutoff = int(observed_positions[-1])
        encoder_local = np.arange(cutoff - spec.context_length + 1, cutoff + 1)
        if encoder_local[0] < required_history:
            return None
        decoder_local = np.arange(cutoff + 1, min(len(group), cutoff + 1 + spec.prediction_length))
        combined = np.concatenate([encoder_local, decoder_local])
        if spec.require_regular and not _is_regular(group.iloc[combined][schema.time_col]):
            return None
        return Window(
            key,
            tuple(int(positions[value]) for value in encoder_local),
            tuple(int(positions[value]) for value in decoder_local),
        )


def _as_tuple(value) -> Tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _is_regular(values: Iterable[Any]) -> bool:
    values = pd.Series(values)
    if len(values) < 3:
        return True
    if pd.api.types.is_datetime64_any_dtype(values):
        numeric = values.astype("datetime64[ns]").astype("int64").to_numpy()
    else:
        numeric = values.to_numpy()
    differences = np.diff(numeric)
    return bool(len(differences) == 0 or np.all(differences == differences[0]))
