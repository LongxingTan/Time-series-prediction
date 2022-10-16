import unittest

import tensorflow as tf

from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.wavenet import WaveNet


class WaveNetTest(unittest.TestCase):
    def test_encoder(self):
        pass

    def test_decoder1(self):
        pass

    def test_decoder2(self):
        pass

    def test_decoder3(self):
        pass

    def test_model(self):
        predict_sequence_length = 8
        custom_model_params = {}
        model = WaveNet(predict_sequence_length=predict_sequence_length, custom_model_params=custom_model_params)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        pass
