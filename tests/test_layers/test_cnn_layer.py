import unittest

import tensorflow as tf

from tfts.layers.cnn_layer import ConvAttTemp, ConvTemp


class CNNLayerTest(unittest.TestCase):
    def test_conv_layer(self):
        train_length = 10
        filters = 64
        kernel_size = 2

        layer = ConvTemp(filters, kernel_size=kernel_size, dilation_rate=2)
        x = tf.random.normal([2, train_length, 1])
        y = layer(x)
        self.assertEqual(y.shape, (2, train_length, filters))
