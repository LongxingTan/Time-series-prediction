# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.transformer`"""

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers


class CausalMask:
    """Casual Mask is used for transformer decoder, used in first self-attention for decoder feature"""

    def __init__(self, B, L):
        mask_shape = [B, L, L]  # for multi-heads split [B, 1, L, L]
        mask_a = tf.linalg.band_part(tf.ones(mask_shape), 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(tf.ones(mask_shape), 0, 0)  # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        self._mask = mask
        tf.stop_gradient(self._mask)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    """ProbMask for informer"""

    def __init__(self, B, H, L, index, scores):
        # B: batch_size, H: num_heads, L: seq_length
        mask = tf.ones([L, scores.shape[-1]], tf.float32)

        mask = 1 - tf.linalg.band_part(mask, -1, 0)
        mask_expanded = tf.broadcast_to(mask, [B, H, L, scores.shape[-1]])
        # mask specific q based on reduced Q
        mask_Q = tf.gather_nd(mask_expanded, index)
        self._mask = tf.cast(tf.reshape(mask_Q, scores.shape), tf.bool)

    @property
    def mask(self):
        return self._mask
