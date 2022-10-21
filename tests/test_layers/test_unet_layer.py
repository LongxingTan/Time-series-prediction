import unittest

import tensorflow as tf

from tfts.layers.unet_layer import ConvbrLayer, ReBlock, SeBlock, conv_br, re_block, se_block


class UNetLayerTest(unittest.TestCase):
    def test_conv_br_layer(self):
        x = tf.random.normal([2, 16, 1])
        units = 4
        kernel_size = 3
        strides = 1
        dilation = 1
        y = conv_br(x, units, kernel_size, strides, dilation)
        self.assertEqual(y.shape, (2, 16, units))

    def test_se_block_layer(self):
        x = tf.random.normal([2, 16, 1])
        units = 4
        y = se_block(x, units)
        self.assertEqual(y.shape, (2, 16, units))

    def test_re_block_layer(self):
        x = tf.random.normal([2, 16, 1])
        units = 4
        kernel_size = 3
        y = re_block(x, units, kernel_size)
        self.assertEqual(y.shape, (2, 16, units))
