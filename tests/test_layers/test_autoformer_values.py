"""Regression for aggregation over the value projection, independently of model ports."""

import unittest

import numpy as np
import tensorflow as tf

from tfts.layers.autoformer_layer import AutoCorrelation


class AutoCorrelationValuesTest(unittest.TestCase):
    def test_autocorrelation_aggregates_values_and_has_value_gradient(self):
        layer = AutoCorrelation(4, 2)
        q = tf.zeros([2, 2, 16, 2])
        v = tf.Variable(tf.ones_like(q) * 3)
        with tf.GradientTape() as tape:
            result = layer.time_delay_agg(q, q, v)
            loss = tf.reduce_sum(result)
        np.testing.assert_allclose(result, 3, atol=1e-6)
        np.testing.assert_allclose(tape.gradient(loss, v), 1, atol=1e-6)
        x = tf.random.normal([2, 16, 4])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(x, x, x))
        gradients = tape.gradient(loss, layer.wv.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))
        self.assertGreater(float(tf.linalg.global_norm(gradients)), 0)
