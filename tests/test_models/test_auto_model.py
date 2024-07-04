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


class TestAutoModel(unittest.TestCase):
    def test_auto_model_init(
        self,
    ):

        auto_model = AutoModel("seq2seq", predict_length=5)
        input_data = np.random.rand(1, 10, 3)
        output = auto_model(input_data)

        self.assertEqual(output.shape, (1, 5, 1))
