"""Dataframe preparation built on the shared schema and window materializer."""

import numpy as np
import pandas as pd

from tfts.data.features import (
    DatetimeTransform,
    FeatureDType,
    FeatureManifest,
    FeaturePipeline,
    FeatureRole,
    PreparedTimeSeries,
    Schema,
    resolve_feature_plan,
)

from .regularize import groups, infer_frequency, regularize, validate_frame
from .scaling import ChannelScaler
from .split import TemporalSplit
from .windows import Window, WindowIndex, WindowIndexer, WindowSpec


def index_frame(prepared, spec):
    """Translate a validated frame to the NumPy window index boundary."""
    frame, schema = prepared.frame, prepared.schema
    schema.validate_frame(frame)
    times = frame[schema.time_col]
    time_values = pd.DatetimeIndex(times).asi8 if pd.api.types.is_datetime64_any_dtype(times) else times.to_numpy()
    observed = frame[list(schema.target_cols)].notna().all(axis=1).to_numpy()
    return WindowIndexer().build(
        groups(frame, schema.group_cols),
        spec,
        time_values=time_values,
        target_observed=observed,
        required_history=prepared.manifest.required_history,
    )


class FramePipeline:
    """One fitted column layout, vocabulary, and set of training-only scalers."""

    def __init__(self, schema, context, horizon, normalize="standard", calendar=(), freq=None, transforms=()):
        self.schema = schema
        self.context = context
        self.horizon = horizon
        self.normalize = normalize
        self.calendar = tuple(calendar)
        self.freq = freq
        self.transforms = tuple(transforms)
        if not isinstance(context, int) or not isinstance(horizon, int) or min(context, horizon) < 1:
            raise ValueError("context and horizon must be positive integers")
        if normalize == "log1p":
            raise ValueError("FramePipeline requires affine target scaling; use log1p explicitly on values")

    def _groups(self, frame):
        return groups(frame, self.schema.group_cols)

    def validate(self, frame):
        return validate_frame(frame, self.schema)

    def transform(self, frame):
        """Apply the fitted layout without refitting scalers or vocabularies."""
        if not hasattr(self, "frequency"):
            raise RuntimeError("FramePipeline is not fitted")
        frame = regularize(self.validate(frame), self.schema, self.frequency)
        return self._encode(self.features_.transform(frame))

    def fit(self, frame, validation=0.2):
        if validation is None:
            validation = 0.0
        if not 0 <= validation < 1:
            raise ValueError("validation must be None or a fraction in [0, 1)")
        split = TemporalSplit(validation)
        frame = self.validate(frame)
        raw_train, _ = split.rows(frame, self.schema.group_cols)
        self.frequency = self.freq if self.freq is not None else infer_frequency(frame.iloc[raw_train], self.schema)
        frame = regularize(frame, self.schema, self.frequency)
        training_rows, cutoffs = split.rows(frame, self.schema.group_cols)
        import copy

        transforms = [DatetimeTransform(self.calendar, categorical=False)] if self.calendar else []
        self.features_ = FeaturePipeline(transforms + list(copy.deepcopy(self.transforms)))
        self.features_.fit(frame.iloc[training_rows], self.schema)
        prepared = self.features_.transform(frame)
        self.output_schema = prepared.schema
        self.plan = resolve_feature_plan(self.output_schema)
        for key, positions in self._groups(frame):
            cutoff = cutoffs[key] - int(positions[0])
            if cutoff < self.context + self.horizon:
                raise ValueError(f"Series {key!r} needs at least context + horizon training rows")
            if validation and len(positions) - cutoff < self.horizon:
                raise ValueError(f"Series {key!r} validation partition is shorter than horizon")
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
        windows = index_frame(prepared, WindowSpec(self.context, self.horizon, require_regular=False))
        train_windows, valid_windows = split.windows(windows, cutoffs)
        self.fit_report = {
            "training_rows": len(training_rows),
            "training_windows": len(train_windows),
            "validation_windows": (0 if valid_windows is None else len(valid_windows)),
            "dropped_boundary_windows": len(windows)
            - len(train_windows)
            - (0 if valid_windows is None else len(valid_windows)),
        }
        encoded = self._encode(prepared)
        return (
            encoded,
            train_windows,
            valid_windows,
        )

    def _encode(self, prepared):
        frame = prepared.frame.copy()
        for name, vocabulary in self.vocabularies.items():
            codes = {value: index + 1 for index, value in enumerate(vocabulary)}
            missing = frame[name].isna()
            frame[name] = frame[name].map(codes).fillna(0).where(~missing, np.nan)
        for name, scaler in self.scalers.items():
            frame[name] = scaler.transform(frame[[name]].to_numpy())[:, 0]
        frame[list(self.schema.target_cols)] = self.target_scaler.transform(
            frame[list(self.schema.target_cols)].to_numpy()
        )
        return PreparedTimeSeries(frame, prepared.manifest)

    def dataset(self, prepared, windows, **kwargs):
        from .tf_dataset import SequenceMaterializer

        return SequenceMaterializer().dataset(prepared, windows, plan=self.plan, **kwargs)

    def prediction(self, history, future=None):
        history = regularize(self.validate(history), self.schema, self.frequency)
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
            warmup = self.features_.manifest_.required_history
            if len(group) < self.context + warmup:
                raise ValueError(f"Series {key!r} needs context + {warmup} warm-up rows")
            group = group.tail(self.context + warmup)
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
                    range(offset + warmup, offset + warmup + self.context),
                    range(offset + warmup + self.context, offset + warmup + self.context + self.horizon),
                )
            )
            offset += warmup + self.context + self.horizon
            output_index.extend((key, date) for date in dates)
        raw = pd.concat(frames, ignore_index=True)
        prepared = self._encode(self.features_.transform(raw))
        return (
            prepared,
            WindowIndex(WindowSpec(self.context, self.horizon, mode="predict"), tuple(windows)),
            output_index,
        )

    def inverse_output(self, output):
        from .arrays import inverse_output

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
            "features": self.features_.to_dict(),
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
            Schema.from_dict(state["schema"]),
            state["context"],
            state["horizon"],
            state["normalize"],
            state["calendar"],
            state["freq"],
        )
        result.features_ = FeaturePipeline.from_dict(state["features"])
        result.transforms = result.features_.transforms[1 if result.calendar else 0 :]
        result.output_schema = Schema.from_dict(state["output_schema"])
        result.plan = resolve_feature_plan(result.output_schema)
        result.frequency = state["frequency"]
        result.vocabularies = state["vocabularies"]
        result.scalers = {name: ChannelScaler.from_dict(value) for name, value in state["scalers"].items()}
        result.target_scaler = ChannelScaler.from_dict(state["target_scaler"])
        return result
