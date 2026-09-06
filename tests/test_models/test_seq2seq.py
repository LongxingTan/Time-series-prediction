import unittest

import tensorflow as tf

from tfts.contracts import BackboneOutput, TimeSeriesBatch
from tfts.models.seq2seq import Decoder, Encoder, Seq2seq


class Seq2seqTest(unittest.TestCase):
    def test_encoder(self):
        pass

    def test_decoder_step(self):
        predict_sequence_length = 5
        rnn_size = 32
        layer = Decoder(rnn_size=rnn_size, predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 11, 1])
        init_input = tf.random.normal([2, 1])
        init_state = tf.random.normal([2, rnn_size])
        layer.build(x.shape)
        output = layer.step(init_input[:, None, :], (init_state,), x[:, 0, :])
        self.assertEqual(output.prediction.shape, (2, 1, 1))

    def test_model(self):
        predict_sequence_length = 8
        model = Seq2seq(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(batch=TimeSeriesBatch(x))
        self.assertIsInstance(y, BackboneOutput)
        self.assertEqual(y.native_forecast.shape, (2, predict_sequence_length, 1))

    def test_model_gru_attn(self):
        predict_sequence_length = 8
        model = Seq2seq(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(batch=TimeSeriesBatch(x))
        self.assertEqual(y.native_forecast.shape, (2, predict_sequence_length, 1))

    def test_model_lstm(self):
        predict_sequence_length = 8

        model = Seq2seq(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(batch=TimeSeriesBatch(x))
        self.assertEqual(y.native_forecast.shape, (2, predict_sequence_length, 1))

    def test_model_lstm_gru(self):
        predict_sequence_length = 8

        model = Seq2seq(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(batch=TimeSeriesBatch(x))
        self.assertEqual(y.native_forecast.shape, (2, predict_sequence_length, 1))

    def test_train(self):
        pass
