"""Declarative model capabilities used for safe auto dispatch."""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Mapping

from .structure import SpatialArrangement, TopologyInput


class OutputPort(str, Enum):
    SEQUENCE = "sequence"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    POOLED = "pooled"
    NATIVE_FORECAST = "native_forecast"
    DISTRIBUTION = "distribution"


@dataclass(frozen=True)
class ModelInputSpec:
    """Feature roles and datatypes one model implementation can consume."""

    accepted_roles: FrozenSet[str] = frozenset({"observed_past"})
    supports_categorical: bool = False
    supports_static: bool = False
    supports_multivariate_target: bool = True
    requires_target_history: bool = False
    requires_matching_time_features: bool = False
    accepted_dtypes_by_role: Mapping[str, FrozenSet[str]] = field(default_factory=dict)
    arrangement: SpatialArrangement = SpatialArrangement.NONE
    accepted_topologies: FrozenSet[TopologyInput] = frozenset({TopologyInput.NONE})
    supports_dynamic_graph: bool = False
    supports_node_mask: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_roles",
            frozenset(getattr(role, "value", role) for role in self.accepted_roles),
        )
        unknown_roles = set(self.accepted_roles) - {"observed_past", "known_future", "static"}
        if unknown_roles:
            raise ValueError(f"unknown feature roles: {sorted(unknown_roles)}")
        if "static" in self.accepted_roles and not self.supports_static:
            raise ValueError("accepted static features require supports_static=True")
        accepted_dtypes = {
            getattr(role, "value", role): frozenset(getattr(dtype, "value", dtype) for dtype in dtypes)
            for role, dtypes in self.accepted_dtypes_by_role.items()
        }
        unknown_roles = set(accepted_dtypes) - set(self.accepted_roles)
        if unknown_roles:
            raise ValueError(f"dtype constraints reference unaccepted roles: {sorted(unknown_roles)}")
        unknown_dtypes = set().union(*accepted_dtypes.values()) - {"real", "categorical", "boolean"}
        if unknown_dtypes:
            raise ValueError(f"unknown feature dtypes: {sorted(unknown_dtypes)}")
        if not self.supports_categorical and any("categorical" in dtypes for dtypes in accepted_dtypes.values()):
            raise ValueError("categorical dtype constraints require supports_categorical=True")
        object.__setattr__(self, "accepted_dtypes_by_role", MappingProxyType(accepted_dtypes))
        arrangement = SpatialArrangement.normalize(self.arrangement)
        topologies = frozenset(TopologyInput.normalize(topology) for topology in self.accepted_topologies)
        if not topologies:
            raise ValueError("accepted_topologies cannot be empty")
        if (self.supports_dynamic_graph or self.supports_node_mask) and (arrangement != SpatialArrangement.SET):
            raise ValueError("graph capabilities require the set arrangement")
        object.__setattr__(self, "arrangement", arrangement)
        object.__setattr__(self, "accepted_topologies", topologies)


@dataclass(frozen=True)
class BackboneCapabilities:
    """Output ports and accepted inputs used by task composition."""

    output_ports: FrozenSet[OutputPort] = frozenset({OutputPort.NATIVE_FORECAST})
    input_spec: ModelInputSpec = ModelInputSpec()

    def has_port(self, port: OutputPort) -> bool:
        if port == OutputPort.SEQUENCE and OutputPort.TEMPORAL_SEQUENCE in self.output_ports:
            return True
        return port in self.output_ports
