import unittest

import tensorflow as tf

from examples.make_custom_model import build_model


class CustomModelTest(unittest.TestCase):
    def test_model(self):
        model = build_model()
        x = tf.random.normal([2, 24, 15])
        y = model(x)
        print(model.summary)
        self.assertEqual(y.shape, (2, 16, 1))
