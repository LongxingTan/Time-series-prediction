"""Composable, backend-neutral feature transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import FeatureDType, FeatureManifest, FeatureRole, FeatureSpec, TimeSeriesSchema


@dataclass(frozen=True)
class PreparedTimeSeries:
    """A sorted feature frame together with its fitted semantic manifest."""

    frame: pd.DataFrame
    manifest: FeatureManifest

    @property
    def schema(self) -> TimeSeriesSchema:
        return self.manifest.schema


class FeatureTransform(ABC):
    """Base contract for deterministic feature transformations."""

    @property
    def required_history(self) -> int:
        return 0

    def fit(self, frame: pd.DataFrame, schema: TimeSeriesSchema) -> Mapping[str, Any]:
        return {}

    def output_required_history(self, spec: FeatureSpec, source_history: Mapping[str, int]) -> int:
        """Return warm-up rows required by one output feature."""
        return self.required_history

    @abstractmethod
    def transform(self, frame: pd.DataFrame, schema: TimeSeriesSchema) -> Tuple[pd.DataFrame, Sequence[FeatureSpec]]:
        """Return a copied frame and the specs for newly generated columns."""


class LagTransform(FeatureTransform):
    """Create causal lagged values without filling unavailable history."""

    def __init__(self, columns, lags, tags: Iterable[str] = ("lag", "target_history")):
        self.columns = (columns,) if isinstance(columns, str) else tuple(columns)
        self.lags = (lags,) if isinstance(lags, int) else tuple(lags)
        self.tags = frozenset(tags)
        if not self.lags or any(not isinstance(lag, int) or lag <= 0 for lag in self.lags):
            raise ValueError("lags must contain positive integers")

    @property
    def required_history(self) -> int:
        return max(self.lags)

    def output_required_history(self, spec, source_history):
        return source_history.get(spec.source, 0) + int(spec.parameters["lag"])

    def transform(self, frame, schema):
        result = frame.copy()
        specs: List[FeatureSpec] = []
        grouped = result.groupby(list(schema.group_cols), observed=True, sort=False) if schema.group_cols else None
        for column in self.columns:
            source = schema.get(column)
            for lag in self.lags:
                name = f"{column}_lag_{lag}"
                result[name] = grouped[column].shift(lag) if grouped is not None else result[column].shift(lag)
                specs.append(
                    FeatureSpec(
                        name=name,
                        role=FeatureRole.OBSERVED_PAST,
                        dtype=source.dtype,
                        source=column,
                        transform="lag",
                        parameters={"lag": lag},
                        tags=self.tags,
                    )
                )
        return result, specs


class RollingTransform(FeatureTransform):
    """Create strictly causal rolling statistics."""

    _FUNCTIONS = {"mean", "std", "min", "max", "median"}

    def __init__(
        self, columns, windows, functions=("mean", "std"), min_periods=None, shift: int = 1, tags=("rolling",)
    ):
        self.columns = (columns,) if isinstance(columns, str) else tuple(columns)
        self.windows = (windows,) if isinstance(windows, int) else tuple(windows)
        self.functions = (functions,) if isinstance(functions, str) else tuple(functions)
        self.min_periods = min_periods
        self.shift = shift
        self.tags = frozenset(tags)
        if not self.windows or any(not isinstance(window, int) or window <= 0 for window in self.windows):
            raise ValueError("windows must contain positive integers")
        unknown = set(self.functions) - self._FUNCTIONS
        if unknown:
            raise ValueError(f"unsupported rolling functions: {sorted(unknown)}")
        if not isinstance(shift, int) or shift < 0:
            raise ValueError("shift must be a non-negative integer")

    @property
    def required_history(self) -> int:
        return max(self.windows) + self.shift - 1

    def output_required_history(self, spec, source_history):
        return source_history.get(spec.source, 0) + int(spec.parameters["window"]) + int(spec.parameters["shift"]) - 1

    def transform(self, frame, schema):
        result = frame.copy()
        specs: List[FeatureSpec] = []
        for column in self.columns:
            schema.get(column)
            if schema.group_cols:
                base = result.groupby(list(schema.group_cols), observed=True, sort=False)[column].shift(self.shift)
                keys = [result[group] for group in schema.group_cols]
            else:
                base = result[column].shift(self.shift)
                keys = None
            for window in self.windows:
                min_periods = self.min_periods if self.min_periods is not None else window
                for function in self.functions:
                    if keys is None:
                        rolling = base.rolling(window, min_periods=min_periods)
                        values = getattr(rolling, function)()
                    else:
                        values = base.groupby(keys, observed=True, sort=False).transform(
                            lambda values, fn=function: getattr(values.rolling(window, min_periods=min_periods), fn)()
                        )
                    name = f"{column}_roll_{window}_{function}"
                    result[name] = values
                    specs.append(
                        FeatureSpec(
                            name=name,
                            role=FeatureRole.OBSERVED_PAST,
                            dtype=FeatureDType.REAL,
                            source=column,
                            transform="rolling",
                            parameters={"window": window, "function": function, "shift": self.shift},
                            tags=self.tags,
                        )
                    )
        return result, specs


class DatetimeTransform(FeatureTransform):
    """Generate decoder-safe calendar attributes from the time column."""

    _ATTRIBUTES = {"year", "month", "day", "dayofweek", "dayofyear", "hour", "minute", "quarter"}

    def __init__(self, attributes=("month", "dayofweek"), categorical: bool = True, tags=("calendar",)):
        self.attributes = tuple(attributes)
        self.categorical = categorical
        self.tags = frozenset(tags)
        unknown = set(self.attributes) - self._ATTRIBUTES
        if unknown:
            raise ValueError(f"unsupported datetime attributes: {sorted(unknown)}")

    def transform(self, frame, schema):
        result = frame.copy()
        times = pd.to_datetime(result[schema.time_col])
        specs = []
        dtype = FeatureDType.CATEGORICAL if self.categorical else FeatureDType.REAL
        for attribute in self.attributes:
            name = f"{schema.time_col}_{attribute}"
            result[name] = getattr(times.dt, attribute).astype("int32")
            specs.append(
                FeatureSpec(
                    name,
                    FeatureRole.KNOWN_FUTURE,
                    dtype,
                    source=schema.time_col,
                    transform="datetime",
                    parameters={"attribute": attribute},
                    tags=self.tags,
                )
            )
        return result, specs


class FourierTransform(FeatureTransform):
    """Generate sine/cosine pairs for a calendar period."""

    def __init__(self, attribute: str, period: int, order: int = 1, tags=("calendar", "fourier")):
        if period <= 0 or order <= 0:
            raise ValueError("period and order must be positive")
        self.attribute = attribute
        self.period = period
        self.order = order
        self.tags = frozenset(tags)

    def transform(self, frame, schema):
        result = frame.copy()
        times = pd.to_datetime(result[schema.time_col])
        if not hasattr(times.dt, self.attribute):
            raise ValueError(f"unsupported datetime attribute {self.attribute!r}")
        values = getattr(times.dt, self.attribute).astype(float)
        specs = []
        for harmonic in range(1, self.order + 1):
            angle = 2 * np.pi * harmonic * values / self.period
            for function, generated in (("sin", np.sin(angle)), ("cos", np.cos(angle))):
                name = f"{schema.time_col}_{self.attribute}_{function}_{harmonic}"
                result[name] = generated
                specs.append(
                    FeatureSpec(
                        name,
                        FeatureRole.KNOWN_FUTURE,
                        FeatureDType.REAL,
                        source=schema.time_col,
                        transform="fourier",
                        parameters={"attribute": self.attribute, "period": self.period, "harmonic": harmonic},
                        tags=self.tags,
                    )
                )
        return result, specs


class CategoricalEncoderTransform(FeatureTransform):
    """Fit stable integer codes, reserving one value for missing/unseen data."""

    def __init__(self, columns, suffix: str = "_encoded", unknown_value: int = 0, tags=("encoded",)):
        self.columns = (columns,) if isinstance(columns, str) else tuple(columns)
        if not self.columns:
            raise ValueError("columns must contain at least one feature")
        if not suffix:
            raise ValueError("suffix must be non-empty")
        self.suffix = suffix
        self.unknown_value = int(unknown_value)
        if self.unknown_value < 0:
            raise ValueError("unknown_value must be non-negative for embedding compatibility")
        self.tags = frozenset(tags)
        self.categories_: Dict[str, Tuple[Any, ...]] = {}

    def fit(self, frame, schema):
        categories = {}
        for column in self.columns:
            spec = schema.get(column)
            if spec.role == FeatureRole.TARGET:
                raise ValueError("CategoricalEncoderTransform cannot encode a target column")
            values = frame[column].dropna().drop_duplicates().tolist()
            categories[column] = tuple(_python_scalar(value) for value in values)
        self.categories_ = categories
        return {column: [_json_scalar(value) for value in values] for column, values in categories.items()}

    def transform(self, frame, schema):
        if set(self.categories_) != set(self.columns):
            raise RuntimeError("CategoricalEncoderTransform is not fitted")
        result = frame.copy()
        specs = []
        for column in self.columns:
            source = schema.get(column)
            available_codes = (code for code in range(len(self.categories_[column]) + 1) if code != self.unknown_value)
            mapping = {value: code for value, code in zip(self.categories_[column], available_codes)}
            cardinality = max([self.unknown_value, *mapping.values()]) + 1
            name = f"{column}{self.suffix}"
            result[name] = result[column].map(mapping).fillna(self.unknown_value).astype("int32")
            specs.append(
                FeatureSpec(
                    name,
                    source.role,
                    FeatureDType.CATEGORICAL,
                    source=column,
                    transform="categorical_encode",
                    parameters={"unknown_value": self.unknown_value, "cardinality": cardinality},
                    tags=source.tags | self.tags,
                )
            )
        return result, specs


class FeaturePipeline:
    """Fit and apply an ordered collection of feature transformations."""

    def __init__(self, transforms: Optional[Sequence[FeatureTransform]] = None):
        self.transforms = tuple(transforms or ())
        self.manifest_: Optional[FeatureManifest] = None
        self.input_schema_: Optional[TimeSeriesSchema] = None

    def fit(self, frame: pd.DataFrame, schema: TimeSeriesSchema) -> "FeaturePipeline":
        prepared, states, final_schema, required_history = self._run(frame, schema, fit=True)
        del prepared
        self.manifest_ = FeatureManifest(
            final_schema,
            required_history=required_history,
            fitted_state=states,
        )
        self.input_schema_ = schema
        return self

    def transform(self, frame: pd.DataFrame) -> PreparedTimeSeries:
        if self.manifest_ is None:
            raise RuntimeError("FeaturePipeline is not fitted; call fit() or fit_transform() first")
        if self.input_schema_ is None:
            raise RuntimeError("FeaturePipeline input schema is unavailable")
        prepared, _, final_schema, _ = self._run(frame, self.input_schema_, fit=False)
        if final_schema.to_dict() != self.manifest_.schema.to_dict():
            raise RuntimeError("feature pipeline output schema changed after fitting")
        return PreparedTimeSeries(prepared, self.manifest_)

    def fit_transform(self, frame: pd.DataFrame, schema: TimeSeriesSchema) -> PreparedTimeSeries:
        prepared, states, final_schema, required_history = self._run(frame, schema, fit=True)
        self.manifest_ = FeatureManifest(
            final_schema,
            required_history=required_history,
            fitted_state=states,
        )
        self.input_schema_ = schema
        return PreparedTimeSeries(prepared, self.manifest_)

    def _run(self, frame, schema, fit):
        schema.validate_frame(frame)
        sort_columns = list(schema.group_cols) + [schema.time_col]
        result = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True).copy()
        current_schema = schema
        histories = {spec.name: 0 for spec in schema.features}
        states: Dict[str, Any] = {}
        for index, transform in enumerate(self.transforms):
            if fit:
                states[f"{index}:{type(transform).__name__}"] = dict(transform.fit(result, current_schema))
            result, specs = transform.transform(result, current_schema)
            histories.update({spec.name: transform.output_required_history(spec, histories) for spec in specs})
            current_schema = current_schema.with_features(specs)
        current_schema.validate_frame(result)
        self._validate_static_features(result, current_schema)
        return result, states, current_schema, max(histories.values(), default=0)

    @staticmethod
    def _validate_static_features(frame, schema):
        static = [spec.name for spec in schema.features if spec.role == FeatureRole.STATIC]
        if not static:
            return
        if schema.group_cols:
            counts = frame.groupby(list(schema.group_cols), observed=True, sort=False)[static].nunique(dropna=False)
            invalid = [column for column in static if (counts[column] > 1).any()]
        else:
            invalid = [column for column in static if frame[column].nunique(dropna=False) > 1]
        if invalid:
            raise ValueError(f"static features vary within a time series: {invalid}")


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _json_scalar(value):
    value = _python_scalar(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
