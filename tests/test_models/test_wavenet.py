import unittest

import tensorflow as tf

from tfts.contracts import TimeSeriesBatch
from tfts.models.wavenet import Decoder, Encoder, WaveNet, WaveNetConfig


class WaveNetTest(unittest.TestCase):
    def test_config(self):
        config = WaveNetConfig()
        print(config.dense_hidden_size)
        print(config.kernel_sizes)

    def test_encoder(self):
        kernel_sizes = [2]
        filters = 32
        dilation_rates = [4]
        dense_hidden_size = 32
        layer = Encoder(kernel_sizes, filters, dilation_rates, dense_hidden_size)

        x = tf.random.normal([2, 7, 1])
        y1, y2 = layer(x)
        self.assertEqual(y1.shape, (2, 7, 1))
        self.assertEqual(y2[0].shape, (2, 7, filters))

    def test_decoder_step(self):
        filters = 32
        dilation_rates = [2]
        dense_hidden_size = 64
        predict_sequence_length = 3
        layer = Decoder(filters, dilation_rates, dense_hidden_size, predict_sequence_length)

        x = tf.random.normal([2, 7, 1])
        init = tf.random.normal([2, 1])
        memory = [tf.random.normal([2, 7, 32])]

        layer.build(x.shape)
        output = layer.step(init[:, None, :], tuple(memory), x[:, 0, :])
        self.assertEqual(output.prediction.shape, (2, 1, 1))

    def test_decoder3(self):
        pass

    def test_model(self):
        predict_sequence_length = 8
        model = WaveNet(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(batch=TimeSeriesBatch(x))
        self.assertEqual(y.native_forecast.shape, (2, predict_sequence_length, 1))

    def test_train(self):
        pass
