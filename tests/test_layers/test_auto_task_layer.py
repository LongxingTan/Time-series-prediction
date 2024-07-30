import unittest

import tensorflow as tf

from tfts.layers.auto_task import GaussianHead


class DeepARLayerTest(unittest.TestCase):
    def test_gaussian_layer(self):
        hidden_size = 32
        layer = GaussianHead(hidden_size)

        x = tf.random.normal([2, 10, 1])
        mu, sig = layer(x)
        self.assertEqual(mu.shape, (2, 10, hidden_size))
        self.assertEqual(sig.shape, (2, 10, hidden_size))
