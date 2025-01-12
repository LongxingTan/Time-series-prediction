import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf

from tfts import (
    AutoConfig,
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
        config = AutoConfig.for_model("seq2seq")

        auto_model = AutoModel.from_config(config, predict_sequence_length=5)
        input_data = np.random.rand(1, 10, 3)
        output = auto_model(input_data)

        self.assertEqual(output.shape, (1, 5, 1))
