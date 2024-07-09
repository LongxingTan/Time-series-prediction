"""Test the U-net model"""

import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.unet import Unet


class UnetTest(unittest.TestCase):
    def test_encoder(self):
        pass

    def test_decoder(self):
        pass

    def test_model(self):
        predict_sequence_length = 4
        model = Unet(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        pass
