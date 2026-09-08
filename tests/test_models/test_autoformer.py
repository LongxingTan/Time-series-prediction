import unittest

import tensorflow as tf

from tfts.models.autoformer import AutoFormer, DecoderLayer, EncoderLayer


class AutoFormerTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        model = AutoFormer(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 64])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_encoder(self):
        kernel_size = 25
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        layer = EncoderLayer(
            d_model=hidden_size,
            d_ff=hidden_size,
            num_heads=num_attention_heads,
            moving_avg=kernel_size,
            dropout=attention_probs_dropout_prob,
        )

        x = tf.random.normal([2, 100, hidden_size])  # after embedding
        y = layer(x)
        self.assertEqual(y.shape, (2, 100, hidden_size))

    def test_decoder_layer(self):
        kernel_size = 25
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        layer = DecoderLayer(
            d_model=hidden_size,
            c_out=hidden_size,
            d_ff=hidden_size,
            num_heads=num_attention_heads,
            moving_avg=kernel_size,
            dropout=attention_probs_dropout_prob,
        )

        x = tf.random.normal([2, 50, hidden_size])  # after embedding
        memory = tf.random.normal([2, 100, hidden_size])
        # init_trend = tf.random.normal([2, 50, hidden_size])
        y1, trend = layer(x, memory)
        self.assertEqual(trend.shape, x.shape)

        self.assertEqual(y1.shape, (2, 50, hidden_size))
