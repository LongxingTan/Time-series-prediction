"""Portable generation state. Decoder caches never enter user hooks."""

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

import tensorflow as tf


class Distribution(Protocol):
    def sample(self, parameters: Mapping[str, tf.Tensor], seed=None) -> tf.Tensor:
        """Draw a value using the supplied distribution parameters."""
        ...


@dataclass(frozen=True)
class GenStep:
    offset: tf.Tensor
    past_values: tf.Tensor
    generated: tf.Tensor
    prediction: tf.Tensor
    distribution: Optional[Distribution] = None
    parameters: Optional[Mapping[str, tf.Tensor]] = None
    value: Optional[tf.Tensor] = None
    quantile_values: Optional[tf.Tensor] = None
