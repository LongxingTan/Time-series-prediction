"""Focused regression coverage for the model review."""

import unittest

import numpy as np
import tensorflow as tf

from tfts.layers.revin import RevIN


class RevINTest(unittest.TestCase):
    def test_revin_formula_inverse_and_independent_stats(self):
        layer = RevIN()
        x = tf.random.stateless_normal([2, 9, 3], seed=[3, 4])
        normalized, stats = layer(x)
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        scale = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True) + 1e-5)
        np.testing.assert_array_equal(normalized, (x - mean) / scale)
        layer(x * 10 + 100)
        np.testing.assert_allclose(layer.inverse(normalized, stats), x, atol=2e-7)
        self.assertEqual(RevIN.from_config(layer.get_config()).epsilon, layer.epsilon)
