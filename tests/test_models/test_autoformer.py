import functools
import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.autoformer import AutoFormer, DecoderLayer, EncoderLayer


class AutoFormerTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        model = AutoFormer(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 32])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_encoder(self):
        kernel_size = 32
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        layer = EncoderLayer(kernel_size, hidden_size, num_attention_heads, attention_probs_dropout_prob)

        x = tf.random.normal([2, 100, hidden_size])  # after embedding
        y = layer(x)
        self.assertEqual(y.shape, (2, 100, hidden_size))

    def test_decoder_layer(self):
        kernel_size = 32
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        layer = DecoderLayer(kernel_size, hidden_size, num_attention_heads, attention_probs_dropout_prob)

        x = tf.random.normal([2, 50, hidden_size])  # after embedding
        memory = tf.random.normal([2, 100, hidden_size])
        init_trend = tf.random.normal([2, 50, hidden_size])
        y1, y2 = layer(x, memory, init_trend)

        self.assertEqual(y1.shape, (2, 50, hidden_size))
        self.assertEqual(y2.shape, (2, 50, hidden_size))
