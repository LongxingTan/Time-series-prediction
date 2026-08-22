"""Univariate Normal distribution output head."""

import math
from typing import Dict, Optional

try:
    from keras import ops
except ImportError:  # Keras 2 bundled with TensorFlow < 2.16
    ops = None

import tensorflow as tf
from tensorflow.keras.layers import Dense

from .base import DistributionOutput


class NormalOutput(DistributionOutput):
    """Normal head with a softplus-constrained positive scale.

    Layer names ``loc`` / ``scale_param`` are intentional: they match the weight
    prefixes of the original Phase-4 DeepAR checkpoint so previously saved weights
    remain loadable after this refactor.
    """

    event_shape: tuple = ()

    def __init__(
        self,
        target_dim: int = 1,
        loc_name: str = "loc",
        scale_name: str = "scale_param",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(name="normal_output")
        self._target_dim = int(target_dim)
        self.epsilon = float(epsilon)
        # parameter layers are constructed once here so the model builds once and
        # can share weights with the autoregressive generator.
        self.loc_layer = Dense(self._target_dim, name=loc_name)
        self.scale_layer = Dense(self._target_dim, name=scale_name)

    @property
    def target_dim(self) -> int:
        return self._target_dim

    def parameters(self, hidden_states: tf.Tensor) -> Dict[str, tf.Tensor]:
        loc = self.loc_layer(hidden_states)
        scale_param = self.scale_layer(hidden_states)
        scale = (ops.softplus(scale_param) if ops is not None else tf.nn.softplus(scale_param)) + self.epsilon
        return {"loc": loc, "scale": scale}

    def mean(self, parameters: Dict[str, tf.Tensor]) -> tf.Tensor:
        return parameters["loc"]

    def sample(self, parameters: Dict[str, tf.Tensor], seed: Optional[int] = None) -> tf.Tensor:
        loc = parameters["loc"]
        scale = parameters["scale"]
        if seed is None:
            noise = tf.random.normal(tf.shape(loc))
        else:
            noise = tf.random.stateless_normal(tf.shape(loc), seed=[seed, 0])
        return loc + scale * noise

    def loss(self, y_true: tf.Tensor, parameters: Dict[str, tf.Tensor], reduction: str = "mean") -> tf.Tensor:
        """Gaussian negative log-likelihood aggregating the last two axes (batch x hidden x target)."""
        loc = parameters["loc"]
        scale = parameters["scale"]
        # log-space is numerically stable; softplus scale is already > epsilon.
        log_scale = tf.math.log(scale)
        nll = 0.5 * (math.log(2.0 * math.pi) + 2.0 * log_scale + tf.square(y_true - loc) / tf.square(scale))
        if reduction == "none":
            return nll
        if reduction == "sum":
            return tf.reduce_sum(nll)
        if reduction == "mean":
            return tf.reduce_mean(nll)
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    def get_config(self):
        config = super().get_config()
        config.update({"target_dim": self.target_dim, "epsilon": self.epsilon})
        return config
