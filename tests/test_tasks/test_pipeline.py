import unittest

import tensorflow as tf

from tfts.contracts import ClassificationOutput
from tfts.tasks.pipeline import TaskPipeline


class TestTaskPipeline(unittest.TestCase):
    def test_pipeline_builds_task_model_from_registry_name(self):
        pipeline = TaskPipeline("classification", "bert", num_labels=3)
        output = pipeline(tf.random.normal([2, 8, 2]))

        self.assertIsInstance(output, ClassificationOutput)
        self.assertEqual(output.logits.shape, (2, 3))

    def test_generation_is_only_exposed_for_forecasting(self):
        pipeline = TaskPipeline("forecasting", "dlinear", prediction_length=2)
        output = pipeline(
            tf.random.normal([2, 8, 1]),
            generation_config={"prediction_length": 4, "strategy": "recursive"},
        )
        self.assertEqual(output.predictions.shape, (2, 4, 1))

        classifier = TaskPipeline("classification", "bert", num_labels=2)
        with self.assertRaisesRegex(ValueError, "only valid for forecasting"):
            classifier(tf.random.normal([2, 8, 1]), generation_config={})


if __name__ == "__main__":
    unittest.main()
