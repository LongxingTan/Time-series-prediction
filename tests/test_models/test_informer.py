"""
python -m unittest -v tests/test_models/test_informer.py
"""

import unittest

import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.layers.attention_layer import FullAttention, ProbAttention
from tfts.models.informer import Decoder, DecoderLayer, DistilConv, Encoder, EncoderLayer, Informer


class InformerTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        custom_model_params = {"skip_connect_mean": True}
        model = Informer(predict_sequence_length=predict_sequence_length, custom_model_params=custom_model_params)

        x = tf.random.normal([2, 16, 5])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_encoder_layer(self):
        attention_hidden_sizes = 64
        num_heads = 4
        attention_dropout = 0.1
        ffn_hidden_sizes = 64
        ffn_dropout = 0.1

        attn_layer = ProbAttention(attention_hidden_sizes, num_heads, attention_dropout)

        layer = EncoderLayer(attn_layer, attention_hidden_sizes, ffn_hidden_sizes, ffn_dropout)
        x = tf.random.normal([2, 100, attention_hidden_sizes])  # after embedding
        y = layer(x)
        self.assertEqual(y.shape, (2, 100, attention_hidden_sizes))

        config = layer.get_config()
        self.assertEqual(config["attention_hidden_sizes"], attention_hidden_sizes)

    def test_encoder(self):
        attention_hidden_sizes = 64
        num_heads = 4
        attention_dropout = 0.1
        ffn_hidden_sizes = 64
        ffn_dropout = 0.1
        n_encoder_layers = 4
        attn_layer = ProbAttention(attention_hidden_sizes, num_heads, attention_dropout)

        layers = [
            EncoderLayer(attn_layer, attention_hidden_sizes, ffn_hidden_sizes, ffn_dropout)
            for _ in range(n_encoder_layers)
        ]
        conv_layers = [DistilConv(attention_hidden_sizes) for _ in range(n_encoder_layers - 1)]
        norm_layer = LayerNormalization()

        layer = Encoder(layers, conv_layers=conv_layers, norm_layer=norm_layer)
        x = tf.random.normal([2, 100, attention_hidden_sizes])  # after embedding
        y = layer(x)
        self.assertEqual(y.shape, (2, 100, attention_hidden_sizes))

    def test_decoder_layer(self):
        attention_hidden_sizes = 64
        num_heads = 4
        attention_dropout = 0.1
        ffn_hidden_sizes = 64
        ffn_dropout = 0.1

        attn_layer1 = ProbAttention(attention_hidden_sizes, num_heads, attention_dropout)
        attn_layer2 = FullAttention(attention_hidden_sizes, num_heads, attention_dropout)
        layer = DecoderLayer(attn_layer1, attn_layer2, attention_hidden_sizes, ffn_hidden_sizes, ffn_dropout)
        x = tf.random.normal([2, 50, attention_hidden_sizes])  # after embedding
        memory = tf.random.normal([2, 100, attention_hidden_sizes])
        y = layer(x, memory=memory)
        self.assertEqual(y.shape, (2, 50, attention_hidden_sizes))

        config = layer.get_config()
        self.assertEqual(config["attention_hidden_sizes"], attention_hidden_sizes)

    def test_decoder(self):
        attention_hidden_sizes = 64
        num_heads = 4
        attention_dropout = 0.1
        ffn_hidden_sizes = 64
        ffn_dropout = 0.1
        n_decoder_layers = 4
        attn_layer1 = FullAttention(attention_hidden_sizes, num_heads, attention_dropout)
        attn_layer2 = FullAttention(attention_hidden_sizes, num_heads, attention_dropout)

        layers = [
            DecoderLayer(attn_layer1, attn_layer2, attention_hidden_sizes, ffn_hidden_sizes, ffn_dropout)
            for _ in range(n_decoder_layers)
        ]
        norm_layer = LayerNormalization()

        decoder = Decoder(layers, norm_layer)
        x = tf.random.normal([2, 50, attention_hidden_sizes])  # after embedding
        memory = tf.random.normal([2, 100, attention_hidden_sizes])
        y = decoder(x, memory=memory)

        self.assertEqual(y.shape, (2, 50, attention_hidden_sizes))
