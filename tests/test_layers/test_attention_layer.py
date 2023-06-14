import unittest

import tensorflow as tf

from tfts.layers.attention_layer import FullAttention, ProbAttention, SelfAttention
from tfts.layers.mask_layer import CausalMask


class AttentionLayerTest(unittest.TestCase):
    def test_full_attention_layer(self):
        hidden_size = 64
        num_heads = 4
        attention_dropout = 0.1
        layer = FullAttention(hidden_size, num_heads, attention_dropout)

        q = tf.random.normal([2, 128, 16])
        k = tf.random.normal([2, 128, 4])
        v = tf.random.normal([2, 128, 4])
        y = layer(q, k, v, training=True)
        self.assertEqual(y.shape, (2, 128, hidden_size))
        config = layer.get_config()
        self.assertEqual(config["hidden_size"], hidden_size)

        mask = CausalMask(2 * num_heads, 128).mask
        y2 = layer(q, k, v, mask=mask)
        self.assertEqual(y2.shape, (2, 128, hidden_size))

    def test_self_attention_layer(self):
        hidden_size = 64
        num_heads = 4
        attention_dropout = 0.1
        layer = SelfAttention(hidden_size, num_heads, attention_dropout)

        x = tf.random.normal([2, 128, 16])
        y = layer(x)
        self.assertEqual(y.shape, (2, 128, hidden_size))

    def test_sparse_attention_layer(self):
        pass

    def test_prob_attention_layer(self):
        hidden_size = 128
        num_heads = 4
        attention_dropout = 0
        layer = ProbAttention(hidden_size, num_heads, attention_dropout)

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
