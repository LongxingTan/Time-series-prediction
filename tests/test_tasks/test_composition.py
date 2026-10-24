import unittest

import tensorflow as tf

from tfts.models.auto_config import AutoConfig
from tfts.tasks.auto import AutoModel
from tfts.tasks.classification import ClassificationOutput


class TestTaskComposition(unittest.TestCase):
    def test_registry_builds_classification_model(self):
        model = AutoModel.from_config(AutoConfig.for_model("bert"), task="classification", num_labels=3)
        output = model(tf.random.normal([2, 8, 2]))
        self.assertIsInstance(output, ClassificationOutput)
        self.assertEqual(output.logits.shape, (2, 3))

    def test_forecast_model_exposes_recursive_generation(self):
        model = AutoModel.from_config(AutoConfig.for_model("dlinear"), output_chunk_length=2)
        output = model.generate(tf.random.normal([2, 8, 1]), {"horizon": 4, "mode": "recursive"})
        self.assertEqual(output.predictions.shape, (2, 4, 1))
