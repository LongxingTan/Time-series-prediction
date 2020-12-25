
# the transformer test case mainly refer to the official:
# https://github.com/tensorflow/models/blob/r2.1.0/official/transformer/v2/transformer_layers_test.py

import tensorflow as tf
from deepts.layers.attention_layer import *


class TransformerLayerTest(tf.test.TestCase):
    def test_attention_layer(self):
        hidden_size = 64
        num_heads = 4
        dropout = 0.5
        dim_per_head = hidden_size//num_heads

        layer = Attention()
        self.assertDictEqual(layer.get_config(), {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'attention_dropout': dropout,
        })

        length = 2
        x = tf.ones([1, length, hidden_size])
        bias = tf.ones([1])

    def test_ffn_layer(self):
        hidden_szie = 64
        filter_size = 32
        relu_dropout = 0.5


if __name__ == '__main__':
    tf.test.main()

