"""Structured outputs with stable names across architectures."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Tuple

import tensorflow as tf


class ModelOutput(OrderedDict):
    """Dataclass mapping that supports named and positional access."""

    def __post_init__(self) -> None:
        OrderedDict.__init__(self)
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                self[field.name] = value

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self.to_tuple()[key]
        return super().__getitem__(key)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        # Keep attribute and mapping access coherent after dataclass construction.
        # ``OrderedDict`` is not initialized while dataclass ``__init__`` runs.
        try:
            dataclass_field = name in {field.name for field in fields(self)}
            if dataclass_field:
                if value is None:
                    OrderedDict.pop(self, name, None)
                else:
                    OrderedDict.__setitem__(self, name, value)
        except TypeError:
            pass

    def to_tuple(self) -> Tuple[Any, ...]:
        return tuple(self.values())


@dataclass
class BackboneOutput(ModelOutput):
    sequence_output: Optional[tf.Tensor] = None
    pooled_output: Optional[tf.Tensor] = None
    native_forecast: Optional[tf.Tensor] = None
    distribution_params: Optional[Mapping[str, tf.Tensor]] = None
    state: Any = None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
    attentions: Optional[Tuple[tf.Tensor, ...]] = None


@dataclass
class ForecastOutput(ModelOutput):
    predictions: Optional[tf.Tensor] = None
    distribution_params: Optional[Mapping[str, tf.Tensor]] = None
    quantile_values: Optional[tf.Tensor] = None
    quantiles: Optional[Tuple[float, ...]] = None
    samples: Optional[tf.Tensor] = None
    loss: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None


@dataclass
class ClassificationOutput(ModelOutput):
    logits: Optional[tf.Tensor] = None
    probabilities: Optional[tf.Tensor] = None
    loss: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None


@dataclass
class ImputationOutput(ModelOutput):
    reconstructed_values: Optional[tf.Tensor] = None
    imputed_values: Optional[tf.Tensor] = None
    mask: Optional[tf.Tensor] = None
    loss: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None


@dataclass
class AnomalyDetectionOutput(ModelOutput):
    reconstruction: Optional[tf.Tensor] = None
    scores: Optional[tf.Tensor] = None
    labels: Optional[tf.Tensor] = None
    threshold: Optional[tf.Tensor] = None
    loss: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None
