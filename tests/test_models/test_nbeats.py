"""Test the N-beats model"""

import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.nbeats import NBeats

# class NBeatsTest(unittest.TestCase):
#     def test_model(self):
#         predict_sequence_length = 8
#         custom_model_params = {}
#         model = NBeats(predict_sequence_length=predict_sequence_length, custom_model_params=custom_model_params)
#
#         x = tf.random.normal([2, 16, 3])
#         y = model(x)
#         self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")
