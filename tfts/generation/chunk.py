"""Public per-emission data and private decoder state."""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import tensorflow as tf


@dataclass(frozen=True)
class Chunk:
    offset: tf.Tensor
    past: tf.Tensor
    generated: tf.Tensor
    prediction: tf.Tensor
    parameters: Optional[Mapping[str, tf.Tensor]] = None
    quantiles: Optional[tf.Tensor] = None
    value: Optional[tf.Tensor] = None
    feedback: Optional[tf.Tensor] = None
    seed: Optional[tf.Tensor] = None

    def replace(self, **changes) -> "Chunk":
        return replace(self, **changes)


@dataclass(frozen=True)
class State:
    """Opaque decoder-owned state. It is never exposed to processors."""

    value: Any
