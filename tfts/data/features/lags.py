"""Causal lag and rolling features, shared by every data entry point."""

from typing import Iterable, List

from tfts.data.features.base import FeatureTransform
from tfts.data.schema import FeatureDType, FeatureRole, FeatureSpec
from tfts.registry import register_transform


@register_transform("lag")
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


@register_transform("rolling")
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
