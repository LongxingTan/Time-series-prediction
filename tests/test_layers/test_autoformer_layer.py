import unittest

import tensorflow as tf

from tfts.layers.autoformer_layer import AutoCorrelation, MovingAvg, SeriesDecomp


class AutoFormerLayerTest(unittest.TestCase):
    def test_moving_avg(self):
        # Define the kernel size and stride
        kernel_size = 25
        stride = 1
        x_input = tf.random.normal([2, 100, 3])
        moving_avg_layer = MovingAvg(kernel_size=kernel_size, stride=stride)

        output = moving_avg_layer(x_input)
        self.assertEqual(output.shape, x_input.shape)

    def test_series_decomp(self):
        kernel_size = 3
        layer = SeriesDecomp(kernel_size)

        x = tf.random.normal([2, 100, 3])
        y1, y2 = layer(x)
        self.assertEqual(y1.shape, x.shape)
        self.assertEqual(y2.shape, x.shape)
        self.assertEqual(layer.kernel_size, kernel_size)

    def test_auto_correlation(self):
        d_model = 64
        num_attention_heads = 4
        layer = AutoCorrelation(d_model=d_model, num_attention_heads=num_attention_heads)

        x = tf.random.normal([2, 100, d_model])
        y = layer(x, x, x)
        self.assertEqual(y.shape, (2, 100, d_model))
