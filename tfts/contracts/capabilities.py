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


@dataclass(frozen=True)
class BackboneCapabilities:
    """Properties that cannot be derived from the task-model registry."""

    output_ports: FrozenSet[OutputPort] = frozenset({OutputPort.NATIVE_FORECAST})
    forecast_modes: FrozenSet[ForecastMode] = frozenset({ForecastMode.DIRECT})
    supports_future_covariates: bool = False
    supports_missing_mask: bool = False
    supports_variable_length: bool = False

    def has_port(self, port: OutputPort) -> bool:
        if port == OutputPort.SEQUENCE and OutputPort.TEMPORAL_SEQUENCE in self.output_ports:
            return True
        return port in self.output_ports
