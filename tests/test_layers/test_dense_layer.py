import unittest

import tensorflow as tf

from tfts.layers.dense_layer import DenseTemp, FeedForwardNetwork


class DenseLayerTest(unittest.TestCase):
    def test_dense_temp_layer(self):
        hidden_size = 32

        layer = DenseTemp(hidden_size)
        x = tf.random.normal([2, 10, 3])
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, hidden_size))

    def test_ffn(self):
        hidden_size = 64
        filter_size = 32
        relu_dropout = 0.5

        layer = FeedForwardNetwork(hidden_size, filter_size, relu_dropout)
        x = tf.random.normal([2, 128, hidden_size])
        y = layer(x, training=True)
        self.assertEqual(y.shape, (2, 128, hidden_size))
