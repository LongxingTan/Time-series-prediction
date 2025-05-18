import unittest

import tensorflow as tf

from tfts.models.bert import Bert, BertConfig


class AutoFormerTest(unittest.TestCase):
    def test_config(self):
        config = BertConfig()
        print(config)

    def test_model(self):
        predict_sequence_length = 8

        model = Bert(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 32])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")
