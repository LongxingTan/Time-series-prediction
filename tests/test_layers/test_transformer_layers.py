# # https://github.com/tensorflow/models/blob/r2.1.0/official/transformer/v2/transformer_layers_test.py

import unittest
import tensorflow as tf
from tfts.layers.attention_layer import *


class AttentionLayerTest(unittest.TestCase):
    def test_full_attention_layer(self):
        hidden_size = 64
        num_heads = 4
        attention_dropout = 0.1
        dim_per_head = hidden_size // num_heads

        layer = FullAttention(hidden_size, num_heads, attention_dropout)
        self.assertDictEqual(layer.get_config(), {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'attention_dropout': dropout,
        })

        x = tf.random.normal([2, 128, 16])
        cache = {
            'k': tf.zeros([2, 0, num_heads, dim_per_head]),
            'v': tf.zeros([2, 0, num_heads, dim_per_head])
        }
        y = layer(x, training=True, cache=cache)
        self.assertEqual(y.shape, (2, 128, hidden_size))
        self.assertEqual(cache['k'].shaoe, (2, 128, num_heads, dim_per_head))

    def test_self_attention_layer(self):
        hidden_size = 64
        num_heads = 4
        attention_dropout = 0.1
        dim_per_head = hidden_size // num_heads

    def test_sparse_attention_layer(self):
        hidden_size = 64

    def test_prob_attention_layer(self):
        pass

    def test_fast_attention_layer(self):
        pass


if __name__ == "__main__":
    unittest.main()
