"""
python -m unittest -v tests/test_models/test_informer.py
"""

from typing import Any, Dict
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer, Trainer
from tfts.layers.attention_layer import Attention, ProbAttention
from tfts.models.informer import Decoder, DecoderLayer, DistilConv, Encoder, EncoderLayer, Informer

tf.config.run_functions_eagerly(True)


class InformerTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        model = Informer(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 5])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_encoder_layer(self):
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        ffn_intermediate_size = 64
        hidden_dropout_prob = 0.1

        attn_layer = ProbAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)

        layer = EncoderLayer(attn_layer, hidden_size, ffn_intermediate_size, hidden_dropout_prob)
        x = tf.random.normal([2, 100, hidden_size])  # after embedding
        y = layer(x)
        self.assertEqual(y.shape, (2, 100, hidden_size))

        config = layer.get_config()
        self.assertEqual(config["hidden_size"], hidden_size)

    def test_encoder(self):
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        ffn_intermediate_size = 64
        hidden_dropout_prob = 0.1
        num_hidden_layers = 4
        attn_layer = ProbAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)

        layers = [
            EncoderLayer(attn_layer, hidden_size, ffn_intermediate_size, hidden_dropout_prob)
            for _ in range(num_hidden_layers)
        ]
        conv_layers = [DistilConv(hidden_size) for _ in range(num_hidden_layers - 1)]
        norm_layer = LayerNormalization()

        layer = Encoder(layers, conv_layers=conv_layers, norm_layer=norm_layer)
        x = tf.random.normal([2, 100, hidden_size])  # after embedding
        y = layer(x)
        self.assertEqual(y.shape, (2, 100, hidden_size))

    def test_decoder_layer(self):
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        ffn_intermediate_size = 64
        hidden_dropout_prob = 0.1

        attn_layer1 = ProbAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        attn_layer2 = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        layer = DecoderLayer(attn_layer1, attn_layer2, hidden_size, ffn_intermediate_size, hidden_dropout_prob)
        x = tf.random.normal([2, 50, hidden_size])  # after embedding
        memory = tf.random.normal([2, 100, hidden_size])
        y = layer(x, memory=memory)
        self.assertEqual(y.shape, (2, 50, hidden_size))

        config = layer.get_config()
        self.assertEqual(config["hidden_size"], hidden_size)

    def test_decoder(self):
        hidden_size = 64
        num_attention_heads = 4
        attention_probs_dropout_prob = 0.1
        ffn_intermediate_size = 64
        hidden_dropout_prob = 0.1
        n_decoder_layers = 4
        attn_layer1 = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        attn_layer2 = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob)

        layers = [
            DecoderLayer(attn_layer1, attn_layer2, hidden_size, ffn_intermediate_size, hidden_dropout_prob)
            for _ in range(n_decoder_layers)
        ]
        norm_layer = LayerNormalization()

        decoder = Decoder(layers, norm_layer)
        x = tf.random.normal([2, 50, hidden_size])  # after embedding
        memory = tf.random.normal([2, 100, hidden_size])
        y = decoder(x, memory=memory)

        self.assertEqual(y.shape, (2, 50, hidden_size))

    def test_train(self):
        config: Dict[str, Any] = {
            "num_hidden_layers": 1,
            "n_decoder_layers": 1,
            "hidden_size": 32 * 1,
            "num_attention_heads": 1,
            "attention_probs_dropout_prob": 0.0,
            "ffn_intermediate_size": 32 * 1,
            "hidden_dropout_prob": 0.0,
            "skip_connect_circle": False,
            "skip_connect_mean": False,
            "prob_attention": False,
            "distil_conv": False,
        }

        custom_config = config.copy()
        custom_config["prob_attention"] = True

        train_length = 49
        predict_length = 10
        n_encoder_feature = 2
        n_decoder_feature = 3
        batch_size = 1

        x_train = (
            np.random.rand(batch_size, train_length, 1),
            np.random.rand(batch_size, train_length, n_encoder_feature),
            np.random.rand(batch_size, predict_length, n_decoder_feature),
        )
        y_train = np.random.rand(batch_size, predict_length, 1)  # target: (batch, predict_length, 1)

        x_valid = (
            np.random.rand(batch_size, train_length, 1),
            np.random.rand(batch_size, train_length, n_encoder_feature),
            np.random.rand(batch_size, predict_length, n_decoder_feature),
        )
        y_valid = np.random.rand(batch_size, predict_length, 1)

        config = AutoConfig.for_model("informer")
        model = AutoModel.from_config(config, predict_length)
        trainer = KerasTrainer(model)
        trainer.train((x_train, y_train), (x_valid, y_valid), epochs=1)
