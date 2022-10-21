import unittest

import tensorflow as tf

from tfts.layers.embed_layer import DataEmbedding, PositionalEmbedding, PositionalEncoding, TokenEmbedding


class EmbedLayerTest(unittest.TestCase):
    def test_token_embedding(self):
        hidden_size = 64
        layer = TokenEmbedding(hidden_size)

        x = tf.ones([2, 128, 1], dtype=tf.float32)
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, hidden_size))

        config = layer.get_config()
        self.assertEqual(config["embed_size"], hidden_size)

    def test_positional_embedding(self):
        x = tf.ones([2, 128, 10])
        layer = PositionalEmbedding()
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, 10))

        config = layer.get_config()
        self.assertEqual(config["max_len"], 5000)

    def test_positional_encoding(self):
        x = tf.ones([2, 128, 10])
        layer = PositionalEncoding()
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, 10))

        config = layer.get_config()
        self.assertEqual(config["max_len"], 5000)

    def test_data_embedding(self):
        embed_size = 64
        layer = DataEmbedding(embed_size)

        x = tf.ones([2, 128, 10])
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, embed_size))

        config = layer.get_config()
        self.assertEqual(config["embed_size"], embed_size)
