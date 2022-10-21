import unittest

import tensorflow as tf

from tfts.layers.nbeats_layer import GenericBlock, NBeatsLayer, SeasonalityBlock, TrendBlock


class NBeatsLayerTest(unittest.TestCase):
    def test_nbeats_layer(self):
        units = 8
        thetas_dim = 1
        share_thetas = 1
        layer = NBeatsLayer(units, thetas_dim, share_thetas)

        x = tf.random.normal([2, 16, 1])
        y = layer(x)
        self.assertEqual(y.shape, (2, 16, units))

    def test_generic_block(self):
        units = 8
        thetas_dim = 1
        backcast_length = 16
        forecast_length = 8
        layer = GenericBlock(units, thetas_dim, backcast_length, forecast_length)

        x = tf.random.normal([2, 15, 1])
        y1, y2 = layer(x)
        self.assertEqual(y1.shape, (2, 15, backcast_length))
        self.assertEqual(y2.shape, (2, 15, forecast_length))

    def test_trend_block(self):
        units = 8
        thetas_dim = 1
        backcast_length = 16
        forecast_length = 8
        layer = TrendBlock(units, thetas_dim, backcast_length, forecast_length)

        x = tf.random.normal([2, 15, 1])
        y1, y2 = layer(x)
        self.assertEqual(y1.shape, (2, 15, backcast_length))
        self.assertEqual(y2.shape, (2, 15, forecast_length))
