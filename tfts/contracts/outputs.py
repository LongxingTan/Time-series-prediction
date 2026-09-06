"""Structured outputs with stable names across architectures."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import tensorflow as tf


@dataclass(frozen=True)
class ModelOutput:
    """Immutable base for every task and backbone output."""

    loss: Optional[tf.Tensor] = None

    def replace(self, **changes):
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name) for field in fields(self) if getattr(self, field.name) is not None
        }


@dataclass(frozen=True)
class BackboneOutput(ModelOutput):
    sequence_output: Optional[tf.Tensor] = None
    pooled_output: Optional[tf.Tensor] = None
    native_forecast: Optional[tf.Tensor] = None
    distribution_params: Optional[Mapping[str, tf.Tensor]] = None
    state: Any = None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
    attentions: Optional[Tuple[tf.Tensor, ...]] = None


@dataclass(frozen=True)
class ForecastOutput(ModelOutput):
    predictions: Optional[tf.Tensor] = None
    distribution_params: Optional[Mapping[str, tf.Tensor]] = None
    quantile_values: Optional[tf.Tensor] = None
    quantiles: Optional[Tuple[float, ...]] = None
    samples: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None


@dataclass(frozen=True)
class ClassificationOutput(ModelOutput):
    logits: Optional[tf.Tensor] = None
    probabilities: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None


@dataclass(frozen=True)
class ImputationOutput(ModelOutput):
    reconstructed_values: Optional[tf.Tensor] = None
    imputed_values: Optional[tf.Tensor] = None
    mask: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None


@dataclass(frozen=True)
class AnomalyDetectionOutput(ModelOutput):
    reconstruction: Optional[tf.Tensor] = None
    scores: Optional[tf.Tensor] = None
    labels: Optional[tf.Tensor] = None
    threshold: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None
