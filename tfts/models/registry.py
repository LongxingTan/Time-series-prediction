"""Declarative model registry and compatibility views.

Each model registers itself next to its implementation with :func:`register_model`.
The registry imports built-in model modules lazily, so adding a model requires no
edits to a central mapping.
"""

from collections import OrderedDict
from dataclasses import dataclass
import importlib
import pkgutil
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Tuple, Type

from tfts.contracts.capabilities import BackboneCapabilities

if TYPE_CHECKING:
    from tfts.features import FeaturePlan, FeatureSelection, TimeSeriesSchema


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable metadata stored for one model implementation."""

    name: str
    model_class: Type
    config_class: Type
    description: str = ""
    paper: str = ""
    tags: Tuple[str, ...] = ()
    tier: str = "experimental"
    capabilities: BackboneCapabilities = BackboneCapabilities()

    def as_dict(self) -> Dict[str, Any]:
        """Return the legacy dictionary representation."""
        capabilities = {
            "output_ports": sorted(port.value for port in self.capabilities.output_ports),
            "forecast_modes": sorted(mode.value for mode in self.capabilities.forecast_modes),
            "supports_future_covariates": self.capabilities.supports_future_covariates,
            "supports_missing_mask": self.capabilities.supports_missing_mask,
            "supports_variable_length": self.capabilities.supports_variable_length,
            "input_spec": {
                "layout": self.capabilities.input_spec.layout.value,
                "accepted_roles": sorted(self.capabilities.input_spec.accepted_roles),
                "supports_categorical": self.capabilities.input_spec.supports_categorical,
                "supports_static": self.capabilities.input_spec.supports_static,
                "supports_multivariate_target": self.capabilities.input_spec.supports_multivariate_target,
                "accepted_dtypes_by_role": {
                    role: sorted(dtypes)
                    for role, dtypes in self.capabilities.input_spec.accepted_dtypes_by_role.items()
                },
                "arrangement": self.capabilities.input_spec.arrangement.value,
                "accepted_topologies": sorted(
                    topology.value for topology in self.capabilities.input_spec.accepted_topologies
                ),
                "supports_dynamic_graph": self.capabilities.input_spec.supports_dynamic_graph,
                "supports_node_mask": self.capabilities.input_spec.supports_node_mask,
            },
        }
        return {
            "class_name": self.model_class.__name__,
            "config_class": self.config_class.__name__,
            "description": self.description,
            "paper": self.paper,
            "tags": list(self.tags),
            "tier": self.tier,
            "capabilities": capabilities,
        }


_ENTRIES: "OrderedDict[str, ModelMetadata]" = OrderedDict()
_BUILTINS_LOADED = False
_LOADING_BUILTINS = False
_NON_MODEL_MODULES = {"auto_config", "auto_model", "base", "registry"}


def register_model(
    name: str,
    *,
    config: Type,
    paper: str = "",
    tags: Tuple[str, ...] = (),
    tier: str = "experimental",
    description: str = "",
    capabilities: Optional[BackboneCapabilities] = None,
):
    """Register a model class where it is defined.

    Duplicate names are rejected unless the decorator is evaluated again for
    the same classes, which makes module reloads harmless.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Model name must be a non-empty string")
    if tier not in {"core", "experimental"}:
        raise ValueError("tier must be either 'core' or 'experimental'")

    def decorator(model_class: Type) -> Type:
        entry = ModelMetadata(
            name=name,
            model_class=model_class,
            config_class=config,
            description=description or (model_class.__doc__ or "").strip().split("\n", 1)[0],
            paper=paper,
            tags=tuple(tags),
            tier=tier,
            capabilities=capabilities or BackboneCapabilities(),
        )
        existing = _ENTRIES.get(name)
        if existing is not None and (
            existing.model_class.__module__ != model_class.__module__
            or existing.model_class.__name__ != model_class.__name__
            or existing.config_class.__name__ != config.__name__
        ):
            raise ValueError(f"Model name {name!r} is already registered by {existing.model_class.__name__}")
        _ENTRIES[name] = entry
        return model_class

    return decorator


def _load_builtin_models() -> None:
    """Import every implementation module once so decorators populate the registry."""
    global _BUILTINS_LOADED, _LOADING_BUILTINS
    if _BUILTINS_LOADED or _LOADING_BUILTINS:
        return

    _LOADING_BUILTINS = True
    try:
        package = importlib.import_module("tfts.models")
        module_names = sorted(
            info.name
            for info in pkgutil.iter_modules(package.__path__)
            if not info.ispkg and info.name not in _NON_MODEL_MODULES and not info.name.startswith("_")
        )
        for module_name in module_names:
            importlib.import_module(f"tfts.models.{module_name}")
        _BUILTINS_LOADED = True
    finally:
        _LOADING_BUILTINS = False


class _RegistryView(Mapping[str, Dict[str, Any]]):
    """Read-only mapping preserving the historical ``MODEL_REGISTRY`` API."""

    def __getitem__(self, key: str) -> Dict[str, Any]:
        _load_builtin_models()
        return _ENTRIES[key].as_dict()

    def __iter__(self) -> Iterator[str]:
        _load_builtin_models()
        return iter(_ENTRIES)

    def __len__(self) -> int:
        _load_builtin_models()
        return len(_ENTRIES)


MODEL_REGISTRY: Mapping[str, Dict[str, Any]] = _RegistryView()


class RegistryFieldView(Mapping[str, Any]):
    """Live view of one metadata field, used by legacy auto mappings."""

    def __init__(self, field: str):
        self.field = field

    def __getitem__(self, key: str) -> Any:
        return MODEL_REGISTRY[key][self.field]

    def __iter__(self) -> Iterator[str]:
        return iter(MODEL_REGISTRY)

    def __len__(self) -> int:
        return len(MODEL_REGISTRY)


def list_models(tag: Optional[str] = None, tier: Optional[str] = None) -> List[str]:
    """List registered models, optionally filtered by tag and stability tier."""
    _load_builtin_models()
    if tier is not None and tier not in {"core", "experimental"}:
        raise ValueError("tier must be either 'core' or 'experimental'")
    return sorted(
        name
        for name, entry in _ENTRIES.items()
        if (tag is None or tag in entry.tags) and (tier is None or tier == entry.tier)
    )


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Return metadata for a registered model."""
    _load_builtin_models()
    if model_name not in _ENTRIES:
        raise ValueError(f"Unknown model {model_name!r}. Available: {list_models()}")
    return _ENTRIES[model_name].as_dict()


def get_model_class(model_name: str) -> Type:
    """Resolve a registry name directly to its model class."""
    _load_builtin_models()
    try:
        return _ENTRIES[model_name].model_class
    except KeyError as error:
        raise ValueError(f"Unknown model {model_name!r}. Available: {list_models()}") from error


def get_model_capabilities(model_name: str) -> BackboneCapabilities:
    """Return the immutable capabilities declared by a backbone."""
    _load_builtin_models()
    try:
        return _ENTRIES[model_name].capabilities
    except KeyError as error:
        raise ValueError(f"Unknown model {model_name!r}. Available: {list_models()}") from error


def check_batch_support(model_name: str, batch, spec=None) -> None:
    """Raise a directive error when a backbone cannot consume a batch."""
    from tfts.contracts import SpatialArrangement, TopologyInput

    spec = spec or get_model_capabilities(model_name).input_spec
    arrangement = batch.arrangement
    if arrangement != spec.arrangement:
        message = (
            f"{model_name!r} requires the {spec.arrangement.value} arrangement, "
            f"but past_values has the {arrangement.value} arrangement."
        )
        if arrangement != SpatialArrangement.NONE and spec.arrangement == SpatialArrangement.NONE:
            message += " Use spatial_strategy='per_node' for independent spatial forecasts."
        raise ValueError(message)
    structure = batch.structure
    if structure is not None and getattr(structure, "is_dynamic", False) and not spec.supports_dynamic_graph:
        raise ValueError(f"{model_name!r} does not support time-varying adjacency")
    if structure is not None and getattr(structure, "node_mask", None) is not None:
        if not spec.supports_node_mask:
            raise ValueError(f"{model_name!r} does not support masked nodes")
    topologies = getattr(batch, "topology_inputs", frozenset())
    accepted = spec.accepted_topologies
    topology_optional = TopologyInput.NONE in accepted
    if not topology_optional and not (topologies & accepted):
        names = sorted(topology.value for topology in accepted)
        raise ValueError(f"{model_name!r} requires one of topology inputs {names}")


def resolve_model_features(
    model_name: str,
    schema: "TimeSeriesSchema",
    selection: Optional["FeatureSelection"] = None,
    unsupported: str = "raise",
) -> "FeaturePlan":
    """Resolve an ordered feature plan against a registered model's inputs."""
    from tfts.features import resolve_feature_plan

    capabilities = get_model_capabilities(model_name)
    return resolve_feature_plan(schema, selection, capabilities.input_spec, unsupported=unsupported)


def list_supported_tasks(model_name: str) -> List[str]:
    """Derive safe task support from the backbone's declared output ports."""
    from tfts.contracts import OutputPort

    capabilities = get_model_capabilities(model_name)
    tasks = ["forecasting"]
    if capabilities.has_port(OutputPort.SEQUENCE):
        tasks.append("classification")
    if capabilities.has_port(OutputPort.TEMPORAL_SEQUENCE):
        tasks.extend(["imputation", "anomaly_detection"])
    return tasks


def get_config_class(model_name: str) -> Type:
    """Resolve a registry name directly to its config class."""
    _load_builtin_models()
    try:
        return _ENTRIES[model_name].config_class
    except KeyError as error:
        raise ValueError(f"Unknown model {model_name!r}. Available: {list_models()}") from error


def get_model_class_name(model_name: str) -> str:
    """Resolve a registry name to its model class name."""
    return get_model_class(model_name).__name__


def get_config_class_name(model_name: str) -> str:
    """Resolve a registry name to its config class name."""
    return get_config_class(model_name).__name__
