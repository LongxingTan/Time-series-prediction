import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.deepar import DeepAR


class DeepARTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        model = DeepAR(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, predict_sequence_length, 3])
        loc, scale = model(x)
        self.assertEqual(loc.shape, (2, predict_sequence_length, 1))
        self.assertEqual(scale.shape, (2, predict_sequence_length, 1))
