"""Shared values exchanged by optional incremental backbone decoding.

Context and state are owned by the backbone. Predictions have shape
[B, 1, ..., target_dim]; parameters and quantiles retain their named axes.
These transient values are not model configuration or checkpoint state.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import tensorflow as tf


@dataclass(frozen=True)
class DecodeSession:
    context: Any
    state: Any
    previous: tf.Tensor


@dataclass(frozen=True)
class DecodeStepOutput:
    prediction: tf.Tensor
    state: Any = None
    distribution: Optional[Any] = None
    parameters: Optional[Mapping[str, tf.Tensor]] = None
    quantile_values: Optional[tf.Tensor] = None
