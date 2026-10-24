"""Validate and regularize time grids without fitting feature statistics."""

import numpy as np
import pandas as pd

from .schema import FeatureDType, FeatureRole


def groups(frame, columns):
    """Return stable group keys and positional row indices."""
    if not columns:
        return [((), np.arange(len(frame), dtype=np.int64))]
    return [
        (key if isinstance(key, tuple) else (key,), np.asarray(rows, dtype=np.int64))
        for key, rows in frame.groupby(list(columns), observed=True, sort=False).indices.items()
    ]


def validate_frame(frame, schema):
    """Validate all structural problems before transforming a copied frame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    schema.validate_frame(frame)
    if frame.empty:
        raise ValueError("History must contain at least one row")
    frame = frame.copy()
    time = schema.time_col
    if not pd.api.types.is_numeric_dtype(frame[time]):
        frame[time] = pd.to_datetime(frame[time], errors="raise")
    keys = list(schema.group_cols) + [time]
    problems = []
    if frame[keys].isna().any(axis=None):
        problems.append("Group and time columns cannot contain missing values")
    if pd.api.types.is_numeric_dtype(frame[time]) and not np.isfinite(frame[time]).all():
        problems.append("Timestamps must be finite")
    if frame.duplicated(keys).any():
        problems.append("Duplicate group/time keys")
    for spec in schema.features:
        if spec.dtype != FeatureDType.CATEGORICAL:
            try:
                if np.isinf(frame[spec.name].to_numpy(dtype=float)).any():
                    problems.append(f"{spec.name!r} contains infinite values")
            except (ValueError, TypeError):
                problems.append(f"{spec.name!r} must be numeric")
    for key, rows in groups(frame, schema.group_cols):
        for spec in schema.features:
            if spec.role == FeatureRole.STATIC and frame.iloc[rows][spec.name].nunique(dropna=True) > 1:
                problems.append(f"static feature {spec.name!r} varies in series {key!r}")
    if problems:
        raise ValueError("Invalid frame:\n- " + "\n- ".join(problems))
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def infer_frequency(frame, schema):
    """Infer only from the caller's training partition; gaps need explicit freq."""
    frequencies = []
    for key, rows in groups(frame, schema.group_cols):
        values = frame.iloc[rows][schema.time_col]
        if len(values) < 2:
            raise ValueError(f"Series {key!r} needs at least two timestamps")
        if pd.api.types.is_numeric_dtype(values):
            differences = np.diff(values)
            if not np.allclose(differences, differences[0]) or differences[0] <= 0:
                raise ValueError("Cannot infer frequency with gaps; pass freq explicitly")
            frequency = float(differences[0])
        else:
            frequency = pd.infer_freq(values) if len(values) >= 3 else None
            if frequency is None:
                raise ValueError("Cannot infer a regular frequency; pass freq explicitly")
        frequencies.append(frequency)
    if len(set(frequencies)) != 1:
        raise ValueError("Series have different frequencies")
    return frequencies[0]


def regularize(frame, schema, frequency):
    """Insert missing time steps; preserve NaNs until the tensor boundary."""
    result = []
    for key, rows in groups(frame, schema.group_cols):
        group = frame.iloc[rows]
        times = group[schema.time_col]
        if pd.api.types.is_numeric_dtype(times):
            if not isinstance(frequency, (float, int)) or frequency <= 0:
                raise ValueError("Numeric timestamps require a positive numeric freq")
            offsets = (times.to_numpy() - times.iloc[0]) / frequency
            if not np.allclose(offsets, np.rint(offsets)):
                raise ValueError(f"Series {key!r} has off-grid timestamps")
            grid = times.iloc[0] + frequency * np.arange(int(round(offsets[-1])) + 1)
            # Snap only within the accepted floating-point tolerance.
            group = group.copy()
            group[schema.time_col] = grid[np.rint(offsets).astype(int)]
        else:
            grid = pd.date_range(times.iloc[0], times.iloc[-1], freq=frequency)
            if not times.isin(grid).all():
                raise ValueError(f"Series {key!r} has off-grid timestamps")
        expanded = group.set_index(schema.time_col).reindex(grid)
        expanded.index.name = schema.time_col
        for name, value in zip(schema.group_cols, key):
            expanded[name] = value
        for spec in schema.features:
            if spec.role == FeatureRole.STATIC:
                observed = group[spec.name].dropna()
                if len(observed):
                    expanded[spec.name] = expanded[spec.name].fillna(observed.iloc[0])
        result.append(expanded.reset_index())
    return pd.concat(result, ignore_index=True)
