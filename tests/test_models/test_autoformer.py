import functools
import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.autoformer import AutoFormer, DecoderLayer, EncoderLayer


class AutoFormerTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        custom_model_params = {"attention_hidden_sizes": 32}
        model = AutoFormer(predict_sequence_length=predict_sequence_length, custom_model_params=custom_model_params)

        x = tf.random.normal([2, 16, 32])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_encoder(self):
        kernel_size = 32
        attention_hidden_sizes = 64
        num_heads = 4
        attention_dropout = 0.1
        layer = EncoderLayer(kernel_size, attention_hidden_sizes, num_heads, attention_dropout)

        x = tf.random.normal([2, 100, attention_hidden_sizes])  # after embedding
        y = layer(x)
        self.assertEqual(y.shape, (2, 100, attention_hidden_sizes))

    def test_decoder_layer(self):
        kernel_size = 32
        attention_hidden_sizes = 64
        num_heads = 4
        attention_dropout = 0.1
        layer = DecoderLayer(kernel_size, attention_hidden_sizes, num_heads, attention_dropout)

        x = tf.random.normal([2, 50, attention_hidden_sizes])  # after embedding
        memory = tf.random.normal([2, 100, attention_hidden_sizes])
        init_trend = tf.random.normal([2, 50, attention_hidden_sizes])
        y1, y2 = layer(x, memory, init_trend)

        self.assertEqual(y1.shape, (2, 50, attention_hidden_sizes))
        self.assertEqual(y2.shape, (2, 50, attention_hidden_sizes))
