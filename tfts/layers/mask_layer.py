"""Layer for :py:class:`~tfts.models.transformer`"""

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers


class CausalMask(tf.keras.layers.Layer):
    """Casual Mask is used for transformer decoder, used in first self-attention for decoder feature"""

    def __init__(self, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        mask_shape = [batch_size, seq_length, seq_length]  # for multi-heads split [B, 1, L, L]
        mask_a = tf.linalg.band_part(tf.ones(mask_shape), 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(tf.ones(mask_shape), 0, 0)  # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.float32)
        return mask

    def get_config(self):
        config = {
            "num_attention_heads": self.num_attention_heads,
        }
        base_config = super(CausalMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        return (batch_size, seq_length, seq_length)


class ProbMask:
    """ProbMask for informer"""

    def __init__(self, B, H, L, index, scores):
        # B: batch_size, H: num_attention_heads, L: seq_length
        mask = tf.ones([L, scores.shape[-1]], tf.float32)

        mask = 1 - tf.linalg.band_part(mask, -1, 0)
        mask_expanded = tf.broadcast_to(mask, [B, H, L, scores.shape[-1]])
        # mask specific q based on reduced Q
        mask_Q = tf.gather_nd(mask_expanded, index)
        self._mask = tf.cast(tf.reshape(mask_Q, tf.shape(scores)), tf.bool)

    @property
    def mask(self):
        return self._mask
