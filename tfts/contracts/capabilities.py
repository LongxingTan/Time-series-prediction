"""Declarative model capabilities used for safe auto dispatch."""

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class OutputPort(str, Enum):
    SEQUENCE = "sequence"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    POOLED = "pooled"
    NATIVE_FORECAST = "native_forecast"
    DISTRIBUTION = "distribution"


class ForecastMode(str, Enum):
    DIRECT = "direct"
    RECURSIVE = "recursive"
    AUTOREGRESSIVE = "autoregressive"
    DIFFUSION = "diffusion"


class InputLayout(str, Enum):
    """Physical input layout expected at the final model boundary."""

    SEQUENCE = "sequence"
    TABULAR = "tabular"


@dataclass(frozen=True)
class ModelInputSpec:
    """Feature roles and datatypes one model implementation can consume."""

    layout: InputLayout = InputLayout.SEQUENCE
    accepted_roles: FrozenSet[str] = frozenset({"observed_past"})
    supports_categorical: bool = False
    supports_static: bool = False
    supports_multivariate_target: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", InputLayout(self.layout))
        object.__setattr__(
            self,
            "accepted_roles",
            frozenset(getattr(role, "value", role) for role in self.accepted_roles),
        )


@dataclass(frozen=True)
class BackboneCapabilities:
    """Properties that cannot be derived from the task-model registry."""

    output_ports: FrozenSet[OutputPort] = frozenset({OutputPort.NATIVE_FORECAST})
    forecast_modes: FrozenSet[ForecastMode] = frozenset({ForecastMode.DIRECT})
    supports_future_covariates: bool = False
    supports_missing_mask: bool = False
    supports_variable_length: bool = False
    input_spec: ModelInputSpec = ModelInputSpec()

    def has_port(self, port: OutputPort) -> bool:
        if port == OutputPort.SEQUENCE and OutputPort.TEMPORAL_SEQUENCE in self.output_ports:
            return True
        return port in self.output_ports
