import unittest

import tensorflow as tf

from tfts.layers.autoformer_layer import AutoCorrelation, SeriesDecomp


class AutoFormerLayerTest(unittest.TestCase):
    def test_series_decomp(self):
        kernel_size = 3
        layer = SeriesDecomp(kernel_size)

        x = tf.random.normal([2, 100, 1])
        y1, y2 = layer(x)
        self.assertEqual(y1.shape, (2, 100, 1))
        self.assertEqual(y2.shape, (2, 100, 1))

    def test_auto_correlation(self):
        d_model = 64
        num_heads = 4
        layer = AutoCorrelation(d_model=d_model, num_heads=num_heads)

        x = tf.random.normal([2, 100, d_model])
        y = layer(x, x, x)
        self.assertEqual(y.shape, (2, 100, d_model))
