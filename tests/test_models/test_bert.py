import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.bert import Bert


class AutoFormerTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        custom_model_config = {"attention_hidden_sizes": 32}
        model = Bert(predict_sequence_length=predict_sequence_length, custom_model_config=custom_model_config)

        x = tf.random.normal([2, 16, 32])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")
