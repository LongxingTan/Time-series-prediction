import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.seq2seq import Decoder1, Decoder2, Decoder3, Encoder, Seq2seq


class Seq2seqTest(unittest.TestCase):
    def test_encoder(self):
        pass

    def test_decoder1(self):
        predict_sequence_length = 5
        rnn_size = 32
        layer = Decoder1(rnn_size=rnn_size, predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 11, 1])
        init_input = tf.random.normal([2, 1])
        init_state = tf.random.normal([2, rnn_size])
        y = layer(x, init_input, init_state)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1))

    def test_decoder2(self):
        predict_sequence_length = 5
        rnn_size = 32
        layer = Decoder2(rnn_size=rnn_size, predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 11, 1])
        init_input = tf.random.normal([2, 1])
        init_state = tf.random.normal([2, rnn_size])
        y = layer(x, init_input, init_state)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1))

    def test_decoder3(self):
        rnn_type = "gru"
        rnn_size = 32
        layer = Decoder3(rnn_type=rnn_type, rnn_size=rnn_size, predict_sequence_length=1)
        x = tf.random.normal([2, 11, 1])
        init_input = tf.random.normal([2, 1])
        init_state = tf.random.normal([2, rnn_size])
        y = layer(x, init_input, init_state)
        self.assertEqual(y.shape, (2, 11, 1))

    def test_model(self):
        predict_sequence_length = 8

        model = Seq2seq(predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_model_gru_attn(self):
        predict_sequence_length = 8
        model = Seq2seq(predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_model_lstm(self):
        predict_sequence_length = 8

        model = Seq2seq(predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_model_lstm_gru(self):
        predict_sequence_length = 8

        model = Seq2seq(predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        pass
