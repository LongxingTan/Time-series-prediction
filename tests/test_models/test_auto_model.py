import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf

from tfts.models.auto_model import (
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForClassification,
    AutoModelForPrediction,
    AutoModelForSegmentation,
)


class MockModel:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, return_dict=None):
        return tf.convert_to_tensor([[1.0, 2.0, 3.0]])


class TestAutoModel(unittest.TestCase):
    def test_auto_model_init(
        self,
    ):

        auto_model = AutoModel("seq2seq", predict_length=5)
        input_data = np.random.rand(1, 10, 3)
        output = auto_model(input_data)

        self.assertEqual(output.shape, (1, 5, 1))
