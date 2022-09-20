
import unittest
import tensorflow as tf
from tfts.models.transformer import Transformer


class TransformerTest(unittest.TestCase):
    def test_model(self):
        custom_model_params = {}
        model = Transformer(custom_model_params)


