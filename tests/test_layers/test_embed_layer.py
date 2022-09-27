
import unittest
from tfts.layers.embed_layer import *


class EmbedLayerTest(unittest.TestCase):
    def test_token_embedding(self):
        vocab_size = 50
        hidden_size = 64
        layer = TokenEmbedding(hidden_size)

        x = tf.ones([2, 128, 1], dtype=tf.float32)
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, hidden_size))

    def test_positional_embedding(self):
        x = tf.ones([2, 128, 10])
        layer = PositionalEmbedding()
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, 10))

    def test_positional_encoding(self):
        x = tf.ones([2, 128, 10])
        layer = PositionalEncoding()
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, 10))

    def test_data_embedding(self):
        embed_size = 64
        layer = DataEmbedding(embed_size)

        x = tf.ones([2, 128, 10])
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, embed_size))

