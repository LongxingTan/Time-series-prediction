import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting
from tfts.generation import ForecastGenerationConfig, RemoveInvalidValuesProcessor, ValueClipProcessor


class GenerationStrategyTest(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForForecasting.from_config(AutoConfig.for_model("dlinear"), prediction_length=2)
        self.inputs = tf.random.normal([2, 8, 1])

    def test_direct_and_recursive_rollout_share_one_entry_point(self):
        direct = self.model.generate(self.inputs)
        recursive = self.model.generate(self.inputs, {"prediction_length": 5, "strategy": "recursive"})

        self.assertEqual(direct.predictions.shape, (2, 2, 1))
        self.assertEqual(recursive.predictions.shape, (2, 5, 1))

    def test_auto_selects_recursive_for_a_longer_requested_horizon(self):
        output = self.model.generate(self.inputs, prediction_length=4)
        self.assertEqual(output.predictions.shape, (2, 4, 1))

    def test_processors_are_applied_after_value_selection(self):
        output = self.model.generate(
            self.inputs,
            processors=[
                RemoveInvalidValuesProcessor(fallback=0.0),
                ValueClipProcessor(minimum=-0.25, maximum=0.25),
            ],
        )

        self.assertTrue(bool(tf.reduce_all(output.predictions <= 0.25)))
        self.assertTrue(bool(tf.reduce_all(output.predictions >= -0.25)))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(output.predictions))))

    def test_generation_config_is_serializable_and_rejects_runtime_objects(self):
        config = ForecastGenerationConfig(prediction_length=7, seed=9)
        restored = ForecastGenerationConfig.from_args(config.to_dict())
        self.assertEqual(restored, config)

        with self.assertRaisesRegex(ValueError, "Unknown generation config fields"):
            ForecastGenerationConfig.from_args({"processor": object()})


if __name__ == "__main__":
    unittest.main()
