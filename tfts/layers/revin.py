"""Reversible instance normalization with explicit, per-call statistics."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="tfts")
class RevIN(tf.keras.layers.Layer):
    """Normalize over time; statistics are never stored on the layer."""

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        scale = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True) + self.epsilon)
        return (x - mean) / scale, (mean, scale)

    def inverse(self, y, stats):
        mean, scale = stats
        return y * scale + mean

    def get_config(self):
        return dict(super().get_config(), epsilon=self.epsilon)
