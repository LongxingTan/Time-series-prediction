"""Base class for probability distribution outputs."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import tensorflow as tf


class DistributionOutput(tf.keras.layers.Layer, ABC):
    """Contract for a parametric predictive distribution projected from RNN hidden states.

    Implementations own their trainable parameter layers (created in ``__init__``) so
    that a containing model can build once, serialize, and share weights with the
    autoregressive generator via ``decode_step``.
    """

    event_shape: tuple = ()
    """Shape of a single predicted value (``()`` for a scalar target)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def target_dim(self) -> int:
        """Number of scalar targets predicted jointly by this head."""

    @abstractmethod
    def parameters(self, hidden_states: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Project ``(batch, seq, hidden)`` states into distribution parameter tensors.

        Returns a dict (e.g. ``{"loc": ..., "scale": ...}``) whose tensors share the
        ``(batch, seq, target_dim)`` shape of ``hidden_states`` with the last dim
        replaced by ``target_dim``.
        """

    @abstractmethod
    def mean(self, parameters: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Deterministic point estimate of the distribution (used by greedy decoding)."""

    @abstractmethod
    def sample(self, parameters: Dict[str, tf.Tensor], seed: Optional[int] = None) -> tf.Tensor:
        """Draw one sample from the distribution (used by ancestral decoding)."""

    @abstractmethod
    def loss(self, y_true: tf.Tensor, parameters: Dict[str, tf.Tensor], reduction: str = "mean") -> tf.Tensor:
        """Negative log-likelihood of ``y_true`` under the distribution."""
