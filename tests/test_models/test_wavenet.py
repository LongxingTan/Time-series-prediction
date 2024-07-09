import unittest

import tensorflow as tf

from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.wavenet import Decoder1, Decoder2, Decoder3, Encoder, WaveNet


class WaveNetTest(unittest.TestCase):
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

    def test_decoder1(self):
        filters = 32
        dilation_rates = [2]
        dense_hidden_size = 32
        predict_sequence_length = 3
        layer = Decoder1(filters, dilation_rates, dense_hidden_size, predict_sequence_length)

        x = tf.random.normal([2, 7, 1])
        init = tf.random.normal([2, 1])
        memory = [tf.random.normal([2, 7, 32]), tf.random.normal([2, 7, 32])]

        y = layer(x, init, memory)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1))

    def test_decoder2(self):
        filters = 32
        dilation_rates = [2]
        dense_hidden_size = 32
        predict_sequence_length = 3
        layer = Decoder2(filters, dilation_rates, dense_hidden_size, predict_sequence_length)

        x = tf.random.normal([2, 7, 1])
        init = tf.random.normal([2, 1])
        memory = [tf.random.normal([2, 7, 32]), tf.random.normal([2, 7, 32])]

        y = layer(x, init, memory)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1))

    def test_decoder3(self):
        pass

    def test_model(self):
        predict_sequence_length = 8
        model = WaveNet(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        pass
