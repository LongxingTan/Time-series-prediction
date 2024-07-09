import functools
import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.transformer import Decoder, Decoder2, Encoder, Transformer


class TransformerTest(unittest.TestCase):
    def test_encoder(self):
        n_encoder_layers = 2
        attention_hidden_sizes = 32
        num_heads = 2
        attention_dropout = 0.0
        ffn_hidden_sizes = 32
        ffn_dropout = 0.0
        layer = Encoder(
            n_encoder_layers,
            attention_hidden_sizes,
            num_heads,
            attention_dropout,
            ffn_hidden_sizes,
            ffn_dropout,
        )
        x = tf.random.normal([2, 16, attention_hidden_sizes])
        y = layer(x)
        self.assertEqual(y.shape, (2, 16, attention_hidden_sizes))

        config = layer.get_config()
        self.assertEqual(config["attention_hidden_sizes"], attention_hidden_sizes)

    def test_decoder(self):
        predict_sequence_length = 2
        n_decoder_layers = 2
        attention_hidden_sizes = 32
        num_heads = 1
        attention_dropout = 0
        ffn_hidden_sizes = 32
        ffn_dropout = 0
        layer = Decoder(
            predict_sequence_length,
            n_decoder_layers,
            attention_hidden_sizes,
            num_heads,
            attention_dropout,
            ffn_hidden_sizes,
            ffn_dropout,
        )

        x = tf.random.normal([2, 16, attention_hidden_sizes])
        init = tf.random.normal([2, 1, 1])
        memory = tf.random.normal([2, 16, attention_hidden_sizes])
        y = layer(x, init, memory)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1))

    def test_decoder2(self):
        pass

    def test_model(self):
        predict_sequence_length = 8
        custom_model_config = {}
        model = Transformer(predict_sequence_length, custom_model_config)
        x = tf.random.normal([16, 160, 36])
        y = model(x)
        self.assertEqual(y.shape, (16, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        train, valid = tfts.get_data("sine", test_size=0.1)
        model = AutoModel("rnn", predict_length=8)
        trainer = KerasTrainer(model)
        trainer.train(train, valid, n_epochs=3)
        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)
