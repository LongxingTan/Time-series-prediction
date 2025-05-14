import unittest

import tensorflow as tf

from tfts.layers.unet_layer import ConvbrLayer, ReBlock, SeBlock


class UNetLayerTest(unittest.TestCase):
    def test_conv_br_layer(self):
        x = tf.random.normal([2, 16, 1])
        units = 4
        kernel_size = 3
        strides = 1
        dilation = 1
        layer = ConvbrLayer(units=units, kernel_size=kernel_size, strides=strides, dilation=dilation)
        y = layer(x)
        self.assertEqual(y.shape, (2, 16, units))

    def test_se_block_layer(self):
        x = tf.random.normal([2, 16, 4])  # Input channels should match units
        units = 4
        layer = SeBlock(units=units)
        y = layer(x)
        self.assertEqual(y.shape, (2, 16, units))

    def test_re_block_layer(self):
        x = tf.random.normal([2, 16, 4])  # Input channels should match units
        units = 4
        kernel_size = 3
        strides = 1
        dilation = 1
        use_se = True
        layer = ReBlock(units=units, kernel_size=kernel_size, strides=strides, dilation=dilation, use_se=use_se)
        y = layer(x)
        self.assertEqual(y.shape, (2, 16, units))
