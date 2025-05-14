import unittest

import numpy as np
import tensorflow as tf

from tfts.layers.attention_layer import Attention, ProbAttention, SelfAttention
from tfts.layers.mask_layer import CausalMask


class AttentionLayerTest(unittest.TestCase):
    def test_full_attention_layer(self):
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        layer = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob)

        q = tf.random.normal([2, 128, 16])
        k = tf.random.normal([2, 128, 4])
        v = tf.random.normal([2, 128, 4])
        y = layer(q, k, v, training=True)
        self.assertEqual(y.shape, (2, 128, hidden_size))
        config = layer.get_config()
        self.assertEqual(config["hidden_size"], hidden_size)

        batch, seq_len = 2, 128
        dummy = tf.zeros((batch, seq_len, 1))
        mask_layer = CausalMask(num_attention_heads=1)
        mask = mask_layer(dummy)
        assert mask.shape == (batch, seq_len, seq_len)

    def test_self_attention_layer(self):
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        layer = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)

        x = tf.random.normal([2, 128, 16])
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, hidden_size))

    def test_sparse_attention_layer(self):
        pass

    def test_prob_attention_layer(self):
        hidden_size = 128
        num_attention_heads = 4
        attention_probs_dropout_prob = 0
        layer = ProbAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)

        q = tf.random.normal([2, 128, 16])
        k = tf.random.normal([2, 128, 4])
        v = tf.random.normal([2, 128, 4])
        y = layer(q, k, v)
        self.assertEqual(y.shape, (2, 128, hidden_size))
        config = layer.get_config()
        self.assertEqual(config["hidden_size"], hidden_size)

    def test_fast_attention_layer(self):
        pass


if __name__ == "__main__":
    unittest.main()
