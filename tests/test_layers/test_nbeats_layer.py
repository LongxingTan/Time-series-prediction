import unittest

import tensorflow as tf

from tfts.layers.nbeats_layer import GenericBlock, SeasonalityBlock, TrendBlock


class NBeatsLayerTest(unittest.TestCase):
    def test_generic_block(self):
        train_sequence_length = 16
        predict_sequence_length = 8
        hidden_size = 8
        n_block_layers = 4
        layer = GenericBlock(train_sequence_length, predict_sequence_length, hidden_size, n_block_layers)

        x = tf.random.normal([2, train_sequence_length])
        y1, y2 = layer(x)
        self.assertEqual(y1.shape, (2, train_sequence_length))
        self.assertEqual(y2.shape, (2, predict_sequence_length))

    def test_trend_block(self):
        train_sequence_length = 16
        predict_sequence_length = 8
        hidden_size = 8
        n_block_layers = 4

        layer = TrendBlock(train_sequence_length, predict_sequence_length, hidden_size, n_block_layers)

        x = tf.random.normal([2, train_sequence_length])
        y1, y2 = layer(x)
        self.assertEqual(y1.shape, (2, train_sequence_length))
        self.assertEqual(y2.shape, (2, predict_sequence_length))

    def test_seasonality_block(self):
        train_sequence_length = 16
        predict_sequence_length = 8
        hidden_size = 8
        n_block_layers = 4

        layer = SeasonalityBlock(train_sequence_length, predict_sequence_length, hidden_size, n_block_layers)

        x = tf.random.normal([2, train_sequence_length])
        y1, y2 = layer(x)
        self.assertEqual(y1.shape, (2, train_sequence_length))
        self.assertEqual(y2.shape, (2, predict_sequence_length))
