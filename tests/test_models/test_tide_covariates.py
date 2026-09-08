"""Focused regression coverage for the model review."""

import unittest

import numpy as np
import tensorflow as tf

from tfts.models.tide import Tide, TideConfig


class TideCovariatesTest(unittest.TestCase):
    def test_tide_uses_future_covariates(self):
        model = Tide(3, TideConfig(feature_dim=2, target_dim=2))
        future = tf.Variable(tf.random.normal([2, 3, 2]))
        inputs = dict(x=tf.random.normal([2, 12, 2]), encoder_feature=tf.ones([2, 12, 2]), decoder_feature=future)
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=False)
            loss = tf.reduce_sum(prediction)
        gradient = tape.gradient(loss, future)
        self.assertIsNotNone(gradient)
        self.assertGreater(float(tf.linalg.global_norm([gradient])), 0)
        changed = model(dict(inputs, decoder_feature=future + 2), training=False)
        self.assertGreater(float(tf.reduce_max(tf.abs(prediction - changed))), 1e-6)
