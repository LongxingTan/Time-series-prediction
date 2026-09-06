import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting
from tfts.generation import (
    DistributionSampler,
    ForecastGenerationConfig,
    RemoveInvalidValuesProcessor,
    ValueClipProcessor,
)


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

    def test_direct_honors_custom_sampler_before_processors(self):
        output = self.model.generate(
            self.inputs,
            sampler=lambda output, **kwargs: tf.ones_like(output.prediction) * 5.0,
            processors=ValueClipProcessor(maximum=2.0),
            return_samples=True,
        )
        np.testing.assert_array_equal(output.predictions, np.full((2, 2, 1), 2.0))
        self.assertEqual(output.samples.shape, (2, 1, 2, 1))

    def test_distribution_sampling_requires_a_distribution(self):
        for strategy in ("direct", "recursive"):
            for sampler in ("sample", DistributionSampler()):
                with self.subTest(strategy=strategy, sampler=sampler):
                    with self.assertRaisesRegex(ValueError, "requires a model output distribution"):
                        self.model.generate(self.inputs, strategy=strategy, sampler=sampler)

    def test_probabilistic_strategies_share_auto_sampling_and_seeded_trajectories(self):
        model = AutoModelForForecasting.from_config(AutoConfig.for_model("deep_ar"), prediction_length=2)
        for strategy in ("direct", "recursive"):
            with self.subTest(strategy=strategy):
                config = ForecastGenerationConfig(strategy=strategy, num_samples=4, return_samples=True, seed=7)
                output = model.generate(self.inputs, config)
                explicit = model.generate(self.inputs, config, sampler="sample")
                traced = tf.function(lambda x: model.generate(x, config).samples)(self.inputs)
                self.assertEqual(output.samples.shape, (2, 4, 2, 1))
                np.testing.assert_array_equal(output.samples, explicit.samples)
                np.testing.assert_allclose(output.samples, traced, atol=1e-6)
                np.testing.assert_allclose(output.predictions, tf.reduce_mean(output.samples, axis=1))
                self.assertGreater(float(tf.reduce_max(tf.abs(output.samples[:, 0] - output.samples[:, 1]))), 0.0)

    def test_direct_distribution_parameters_match_requested_horizon(self):
        model = AutoModelForForecasting.from_config(AutoConfig.for_model("deep_ar"), prediction_length=2)
        output = model.generate(self.inputs, strategy="direct", prediction_length=1, sampler="mean")
        self.assertEqual(output.predictions.shape, (2, 1, 1))
        for parameter in output.distribution_params.values():
            self.assertEqual(parameter.shape, (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
