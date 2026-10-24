"""Dataframe preparation built on the shared schema and window materializer."""

import numpy as np
import pandas as pd

from tfts.features import (
    DatetimeTransform,
    FeatureDType,
    FeatureManifest,
    FeaturePipeline,
    FeatureRole,
    PreparedTimeSeries,
    TimeSeriesSchema,
    resolve_feature_plan,
)

from .materializers import SequenceMaterializer
from .preprocessing import inverse_output
from .scaling import ChannelScaler
from .windowing import Window, WindowIndex, WindowIndexer, WindowSpec


class FramePreprocessor:
    """One fitted column layout, vocabulary, and set of training-only scalers."""

    def __init__(self, schema, context, horizon, normalize="standard", calendar=(), freq=None):
        self.schema = schema
        self.context = context
        self.horizon = horizon
        self.normalize = normalize
        self.calendar = tuple(calendar)
        self.freq = freq

    def _groups(self, frame):
        return WindowIndexer._groups(frame, self.schema.group_cols)

    def _validate(self, frame):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")
        self.schema.validate_frame(frame)
        if frame.empty:
            raise ValueError("History must contain at least one row")
        frame = frame.copy()
        time = self.schema.time_col
        if not pd.api.types.is_numeric_dtype(frame[time]):
            frame[time] = pd.to_datetime(frame[time], errors="raise")
        elif not np.isfinite(frame[time].to_numpy()).all():
            raise ValueError("Timestamps must be finite")
        keys = list(self.schema.group_cols) + [time]
        if frame[keys].isna().any(axis=None):
            raise ValueError("Group and time columns cannot contain missing values")
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate group/time keys: {frame.loc[frame.duplicated(keys), keys].iloc[0].to_dict()}")
        frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        for spec in self.schema.features:
            if spec.dtype != FeatureDType.CATEGORICAL:
                values = frame[spec.name].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"{spec.name!r} contains missing or infinite values. "
                        "Supply finite history before fitting/predicting"
                    )
        FeaturePipeline._validate_static_features(frame, self.schema)
        return frame

    def _check_time(self, frame, fit=False):
        time = self.schema.time_col
        for key, positions in self._groups(frame):
            values = frame.iloc[positions][time]
            if len(values) < 2 and fit:
                raise ValueError(f"Series {key!r} needs at least two timestamps")
            if pd.api.types.is_numeric_dtype(values):
                step = float(values.iloc[1] - values.iloc[0]) if fit else self.frequency
                frequency = step
                regular = step > 0 and np.allclose(np.diff(values), step)
            else:
                frequency = (
                    self.frequency if not fit else self.freq or (pd.infer_freq(values) if len(values) >= 3 else None)
                )
                if frequency is None:
                    raise ValueError(f"Cannot infer a regular frequency for series {key!r}. Pass freq explicitly")
                expected = pd.date_range(values.iloc[0], periods=len(values), freq=frequency)
                regular = pd.DatetimeIndex(values).equals(expected)
            if not regular:
                raise ValueError(f"Series {key!r} has gaps or off-grid timestamps")
            if fit and not hasattr(self, "frequency"):
                self.frequency = frequency
            elif frequency != self.frequency:
                # Offset aliases may differ but describe exactly the same grid.
                if isinstance(frequency, str) and isinstance(self.frequency, str):
                    if pd.tseries.frequencies.to_offset(frequency) == pd.tseries.frequencies.to_offset(self.frequency):
                        continue
                raise ValueError(
                    f"Series {key!r} frequency {frequency!r} differs from fitted frequency {self.frequency!r}"
                )

    def _calendar_frame(self, frame):
        transforms = [DatetimeTransform(self.calendar, categorical=False)] if self.calendar else []
        return FeaturePipeline(transforms).fit_transform(frame, self.schema)

    def fit(self, frame, validation=0.2, input_spec=None):
        if validation is None:
            validation = 0.0
        if not 0 <= validation < 1:
            raise ValueError("validation must be None or a fraction in [0, 1)")
        frame = self._validate(frame)
        self._check_time(frame, fit=True)
        prepared = self._calendar_frame(frame)
        self.output_schema = prepared.schema
        self.plan = resolve_feature_plan(self.output_schema, input_spec=input_spec)
        cutoffs, training_rows = {}, []
        for key, positions in self._groups(frame):
            cutoff = len(positions) if not validation else int(len(positions) * (1 - validation))
            if cutoff < self.context + self.horizon:
                raise ValueError(f"Series {key!r} needs at least context + horizon training rows")
            if validation and len(positions) - cutoff < self.horizon:
                raise ValueError(f"Series {key!r} validation partition is shorter than horizon")
            cutoffs[key] = int(positions[0]) + cutoff
            training_rows.extend(positions[:cutoff])
        train = prepared.frame.iloc[training_rows]
        self.vocabularies = {}
        self.scalers = {}
        for spec in self.plan.selected:
            if spec.dtype == FeatureDType.CATEGORICAL:
                values = pd.unique(train[spec.name].dropna())
                self.vocabularies[spec.name] = [value.item() if hasattr(value, "item") else value for value in values]
            else:
                self.scalers[spec.name] = ChannelScaler(self.normalize).fit(train[[spec.name]].to_numpy())
        self.target_scaler = ChannelScaler(self.normalize).fit(train[list(self.schema.target_cols)].to_numpy())
        windows = WindowIndexer().build(prepared, WindowSpec(self.context, self.horizon))
        train_windows = tuple(
            window for window in windows.windows if window.decoder_indices[-1] < cutoffs[window.group_key]
        )
        valid_windows = tuple(
            window for window in windows.windows if window.decoder_indices[0] >= cutoffs[window.group_key]
        )
        self.fit_report = {
            "training_rows": len(training_rows),
            "training_windows": len(train_windows),
            "validation_windows": len(valid_windows),
            "dropped_boundary_windows": len(windows) - len(train_windows) - len(valid_windows),
        }
        encoded = self._encode(prepared)
        return (
            encoded,
            WindowIndex(windows.spec, train_windows),
            WindowIndex(windows.spec, valid_windows) if validation else None,
        )

    def _encode(self, prepared):
        frame = prepared.frame.copy()
        for name, vocabulary in self.vocabularies.items():
            codes = {value: index + 1 for index, value in enumerate(vocabulary)}
            frame[name] = frame[name].map(codes).fillna(0).astype(np.int32)
        for name, scaler in self.scalers.items():
            frame[name] = scaler.transform(frame[[name]].to_numpy())[:, 0]
        frame[list(self.schema.target_cols)] = self.target_scaler.transform(
            frame[list(self.schema.target_cols)].to_numpy()
        )
        return PreparedTimeSeries(frame, prepared.manifest)

    def dataset(self, prepared, windows, **kwargs):
        return SequenceMaterializer().dataset(prepared, windows, plan=self.plan, **kwargs)

    def prediction(self, history, future=None):
        history = self._validate(history)
        self._check_time(history)
        known = [spec.name for spec in self.schema.features if spec.role == FeatureRole.KNOWN_FUTURE]
        if future is not None:
            future = future.copy()
            if set(self.schema.target_cols) & set(future.columns):
                raise ValueError("future must contain known covariates and timestamps, without target columns")
            required = set(self.schema.group_cols) | {self.schema.time_col} | set(known)
            missing = required - set(future.columns)
            if missing:
                raise ValueError(f"future is missing columns: {sorted(missing)}")
            if not isinstance(self.frequency, (float, int)):
                future[self.schema.time_col] = pd.to_datetime(future[self.schema.time_col])
        elif known:
            raise ValueError(f"Provide future rows for known covariates: {known}")
        future_groups = {} if future is None else {key: future.iloc[pos] for key, pos in self._groups(future)}
        histories = self._groups(history)
        if future is not None and set(future_groups) != {key for key, _ in histories}:
            raise ValueError("future must contain exactly the groups in history")
        frames, windows, output_index = [], [], []
        offset = 0
        for key, positions in histories:
            group = history.iloc[positions]
            if len(group) < self.context:
                raise ValueError(f"Series {key!r} needs at least {self.context} history rows")
            group = group.tail(self.context)
            last = group[self.schema.time_col].iloc[-1]
            if isinstance(self.frequency, (float, int)):
                dates = last + self.frequency * np.arange(1, self.horizon + 1)
            else:
                dates = pd.date_range(last, periods=self.horizon + 1, freq=self.frequency)[1:]
            next_frame = pd.DataFrame({self.schema.time_col: dates})
            for name, value in zip(self.schema.group_cols, key):
                next_frame[name] = value
            supplied = future_groups.get(key)
            if supplied is not None:
                supplied = supplied.sort_values(self.schema.time_col)
                if len(supplied) != self.horizon or not np.array_equal(
                    supplied[self.schema.time_col].to_numpy(), np.asarray(dates)
                ):
                    raise ValueError(
                        f"future timestamps for series {key!r} must cover exactly the next {self.horizon} steps"
                    )
                for name in known:
                    if supplied[name].isna().any():
                        raise ValueError(f"Missing future covariate {name!r} for series {key!r}")
                    if (
                        self.schema.get(name).dtype != FeatureDType.CATEGORICAL
                        and not np.isfinite(supplied[name].to_numpy(dtype=float)).all()
                    ):
                        raise ValueError(f"Infinite future covariate {name!r} for series {key!r}")
                    next_frame[name] = supplied[name].to_numpy()
            for spec in self.schema.features:
                if spec.role == FeatureRole.STATIC:
                    next_frame[spec.name] = group[spec.name].iloc[-1]
                elif spec.name not in next_frame:
                    next_frame[spec.name] = np.nan
            frames.extend([group, next_frame])
            windows.append(
                Window(
                    key,
                    range(offset, offset + self.context),
                    range(offset + self.context, offset + self.context + self.horizon),
                )
            )
            offset += self.context + self.horizon
            output_index.extend((key, date) for date in dates)
        raw = pd.concat(frames, ignore_index=True)
        # Calendar features are deterministic and use timestamps only.
        transformed = raw
        output_schema = self.schema
        if self.calendar:
            transformed, additions = DatetimeTransform(self.calendar, categorical=False).transform(raw, self.schema)
            output_schema = self.schema.with_features(additions)
        prepared = self._encode(PreparedTimeSeries(transformed, FeatureManifest(output_schema)))
        return (
            prepared,
            WindowIndex(WindowSpec(self.context, self.horizon, mode="predict"), tuple(windows)),
            output_index,
        )

    def inverse_output(self, output):
        return inverse_output(output, self.target_scaler)

    def to_frame(self, output, index, steps=None):
        count = self.horizon if steps is None else steps
        predictions = np.asarray(output.predictions)
        quantiles = None if output.quantile_values is None else np.asarray(output.quantile_values)
        rows = []
        for row_index, (key, date) in enumerate(index):
            batch, step = divmod(row_index, self.horizon)
            if step >= count:
                continue
            for channel, target in enumerate(self.schema.target_cols):
                row = dict(zip(self.schema.group_cols, key))
                row.update(
                    {
                        self.schema.time_col: date,
                        "target": target,
                        "prediction": float(predictions[batch, step, channel]),
                    }
                )
                if quantiles is not None:
                    row.update(
                        {
                            f"q{level:g}": float(quantiles[batch, step, channel, i])
                            for i, level in enumerate(output.quantiles)
                        }
                    )
                rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self):
        return {
            "schema": self.schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "context": self.context,
            "horizon": self.horizon,
            "normalize": self.normalize,
            "calendar": list(self.calendar),
            "freq": self.freq,
            "frequency": self.frequency,
            "vocabularies": self.vocabularies,
            "scalers": {name: scaler.to_dict() for name, scaler in self.scalers.items()},
            "target_scaler": self.target_scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, state):
        result = cls(
            TimeSeriesSchema.from_dict(state["schema"]),
            state["context"],
            state["horizon"],
            state["normalize"],
            state["calendar"],
            state["freq"],
        )
        result.output_schema = TimeSeriesSchema.from_dict(state["output_schema"])
        result.plan = resolve_feature_plan(result.output_schema)
        result.frequency = state["frequency"]
        result.vocabularies = state["vocabularies"]
        result.scalers = {name: ChannelScaler.from_dict(value) for name, value in state["scalers"].items()}
        result.target_scaler = ChannelScaler.from_dict(state["target_scaler"])
        return result
