"""Semantic contracts for backend-neutral time-series features."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple


class FeatureRole(str, Enum):
    """When a feature is available relative to a forecast cutoff."""

    TARGET = "target"
    OBSERVED_PAST = "observed_past"
    KNOWN_FUTURE = "known_future"
    STATIC = "static"


class FeatureDType(str, Enum):
    """Logical datatype used when a feature is materialized."""

    REAL = "real"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


def _as_enum(value, enum_class):
    return value if isinstance(value, enum_class) else enum_class(value)


def _freeze_value(value):
    """Recursively detach and freeze JSON-like contract values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _thaw_value(value):
    """Return ordinary JSON-compatible containers from frozen contract values."""
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_thaw_value(item) for item in value), key=str)
    return value


@dataclass(frozen=True)
class FeatureSpec:
    """Description and lineage for one model feature."""

    name: str
    role: FeatureRole
    dtype: FeatureDType = FeatureDType.REAL
    source: Optional[str] = None
    transform: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    tags: FrozenSet[str] = frozenset()
    enabled_by_default: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("feature name must be a non-empty string")
        object.__setattr__(self, "role", _as_enum(self.role, FeatureRole))
        object.__setattr__(self, "dtype", _as_enum(self.dtype, FeatureDType))
        object.__setattr__(self, "parameters", _freeze_value(self.parameters))
        object.__setattr__(self, "tags", frozenset(self.tags))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "dtype": self.dtype.value,
            "source": self.source,
            "transform": self.transform,
            "parameters": _thaw_value(self.parameters),
            "tags": sorted(self.tags),
            "enabled_by_default": self.enabled_by_default,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FeatureSpec":
        return cls(**dict(values))


@dataclass(frozen=True)
class TimeSeriesSchema:
    """Column roles for a collection of one or more time series."""

    time_col: str
    target_cols: Tuple[str, ...]
    features: Tuple[FeatureSpec, ...] = ()
    group_cols: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        targets = (self.target_cols,) if isinstance(self.target_cols, str) else tuple(self.target_cols)
        groups = (self.group_cols,) if isinstance(self.group_cols, str) else tuple(self.group_cols)
        features = tuple(self.features)
        if not self.time_col:
            raise ValueError("time_col must be a non-empty string")
        if not targets or any(not target for target in targets):
            raise ValueError("target_cols must contain at least one column")
        if len(set(targets)) != len(targets):
            raise ValueError("target_cols must be unique")

        by_name = {spec.name: spec for spec in features}
        if len(by_name) != len(features):
            raise ValueError("feature names must be unique")
        for target in targets:
            spec = by_name.get(target)
            if spec is None:
                features += (FeatureSpec(target, FeatureRole.TARGET),)
            elif spec.role != FeatureRole.TARGET:
                raise ValueError(f"target column {target!r} must have role='target'")
        unknown_targets = [
            spec.name for spec in features if spec.role == FeatureRole.TARGET and spec.name not in targets
        ]
        if unknown_targets:
            raise ValueError(f"target feature specs absent from target_cols: {unknown_targets}")

        object.__setattr__(self, "target_cols", targets)
        object.__setattr__(self, "group_cols", groups)
        object.__setattr__(self, "features", features)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in self.features)

    def get(self, name: str) -> FeatureSpec:
        for spec in self.features:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def with_features(self, specs: Iterable[FeatureSpec]) -> "TimeSeriesSchema":
        additions = tuple(specs)
        existing = {spec.name for spec in self.features}
        duplicates = [spec.name for spec in additions if spec.name in existing]
        if duplicates:
            raise ValueError(f"feature transforms produced existing columns: {duplicates}")
        return TimeSeriesSchema(self.time_col, self.target_cols, self.features + additions, self.group_cols)

    def validate_frame(self, frame) -> None:
        required = {self.time_col, *self.target_cols, *self.group_cols, *self.feature_names}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"data is missing schema columns: {missing}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_col": self.time_col,
            "target_cols": list(self.target_cols),
            "group_cols": list(self.group_cols),
            "features": [spec.to_dict() for spec in self.features],
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TimeSeriesSchema":
        values = dict(values)
        values["features"] = tuple(FeatureSpec.from_dict(spec) for spec in values.get("features", ()))
        return cls(**values)


@dataclass(frozen=True)
class FeatureManifest:
    """Serializable output of a fitted feature pipeline."""

    schema: TimeSeriesSchema
    required_history: int = 0
    fitted_state: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.required_history < 0:
            raise ValueError("required_history cannot be negative")
        object.__setattr__(self, "fitted_state", _freeze_value(self.fitted_state))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "required_history": self.required_history,
            "schema": self.schema.to_dict(),
            "fitted_state": _thaw_value(self.fitted_state),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FeatureManifest":
        values = dict(values)
        if values.get("schema_version") != 1:
            raise ValueError(f"unsupported feature manifest version {values.get('schema_version')!r}")
        values["schema"] = TimeSeriesSchema.from_dict(values["schema"])
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "FeatureManifest":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


@dataclass(frozen=True)
class FeatureSelection:
    """User-controlled include/exclude policy for model covariates."""

    include_names: FrozenSet[str] = frozenset()
    exclude_names: FrozenSet[str] = frozenset()
    include_tags: FrozenSet[str] = frozenset()
    exclude_tags: FrozenSet[str] = frozenset()
    include_roles: FrozenSet[FeatureRole] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "include_names", frozenset(self.include_names))
        object.__setattr__(self, "exclude_names", frozenset(self.exclude_names))
        object.__setattr__(self, "include_tags", frozenset(self.include_tags))
        object.__setattr__(self, "exclude_tags", frozenset(self.exclude_tags))
        object.__setattr__(self, "include_roles", frozenset(_as_enum(role, FeatureRole) for role in self.include_roles))
        overlap = self.include_names & self.exclude_names
        if overlap:
            raise ValueError(f"features cannot be both included and excluded: {sorted(overlap)}")

    @property
    def has_includes(self) -> bool:
        return bool(self.include_names or self.include_tags or self.include_roles)


@dataclass(frozen=True)
class FeaturePlan:
    """Resolved, ordered features for one model invocation."""

    selected: Tuple[FeatureSpec, ...]
    excluded: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected", tuple(self.selected))
        object.__setattr__(self, "excluded", MappingProxyType(dict(self.excluded)))

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in self.selected)


def resolve_feature_plan(
    schema: TimeSeriesSchema,
    selection: Optional[FeatureSelection] = None,
    input_spec: Optional[Any] = None,
    unsupported: str = "raise",
) -> FeaturePlan:
    """Resolve user selection against a model input specification."""

    if unsupported not in {"raise", "drop"}:
        raise ValueError("unsupported must be either 'raise' or 'drop'")
    if input_spec is not None and not input_spec.supports_multivariate_target and len(schema.target_cols) > 1:
        raise ValueError("model does not support multivariate targets")
    selection = selection or FeatureSelection()
    known_names = set(schema.feature_names)
    unknown = (selection.include_names | selection.exclude_names) - known_names
    if unknown:
        raise ValueError(f"unknown feature names: {sorted(unknown)}")

    selected = []
    excluded: Dict[str, str] = {}
    for spec in schema.features:
        if spec.role == FeatureRole.TARGET:
            continue
        included = spec.enabled_by_default
        if selection.has_includes:
            included = (
                spec.name in selection.include_names
                or bool(spec.tags & selection.include_tags)
                or spec.role in selection.include_roles
            )
        if spec.name in selection.exclude_names or spec.tags & selection.exclude_tags:
            included = False
            excluded[spec.name] = "excluded by selection"
        if not included:
            excluded.setdefault(spec.name, "not selected")
            continue

        incompatibility = _input_incompatibility(spec, input_spec)
        if incompatibility:
            explicitly_selected = spec.name in selection.include_names
            if unsupported == "raise" or explicitly_selected:
                raise ValueError(f"feature {spec.name!r} is not supported: {incompatibility}")
            excluded[spec.name] = incompatibility
            continue
        selected.append(spec)
    return FeaturePlan(tuple(selected), excluded)


def _input_incompatibility(spec: FeatureSpec, input_spec: Optional[Any]) -> Optional[str]:
    if input_spec is None:
        return None
    accepted_roles = {getattr(role, "value", role) for role in input_spec.accepted_roles}
    if spec.role.value not in accepted_roles:
        return f"model does not accept role {spec.role.value!r}"
    if spec.dtype == FeatureDType.CATEGORICAL and not input_spec.supports_categorical:
        return "model does not accept categorical features"
    if spec.role == FeatureRole.STATIC and not input_spec.supports_static:
        return "model does not accept static features"
    accepted_dtypes = input_spec.accepted_dtypes_by_role.get(spec.role.value)
    if accepted_dtypes is not None and spec.dtype.value not in accepted_dtypes:
        return f"model does not accept dtype {spec.dtype.value!r} for role {spec.role.value!r}"
    return None
