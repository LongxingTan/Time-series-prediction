import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting
from tfts.generation import Trajectory


class TrajectoryTest(unittest.TestCase):
    def test_interpolated_quantile_preserves_batch_time_and_target_axes(self):
        result = Trajectory(
            predictions=tf.zeros([2, 3, 1]),
            quantile_values=tf.broadcast_to([1.0, 5.0, 9.0], [2, 3, 1, 3]),
            quantiles=(0.1, 0.5, 0.9),
        )
        np.testing.assert_allclose(result.quantile(0.3), np.full([2, 3, 1], 3.0))

    def test_single_quantile_level_remains_constant(self):
        result = Trajectory(tf.ones([2, 3, 1]), quantile_values=tf.ones([2, 3, 1, 1]), quantiles=(0.5,))
        np.testing.assert_allclose(result.quantile(0.2), np.ones([2, 3, 1]))

    def test_generation_retains_head_quantile_levels(self):
        model = AutoModelForForecasting.from_config(
            AutoConfig.for_model("bert"), output_chunk_length=2, head="quantile", quantiles=(0.1, 0.5, 0.9)
        )
        result = model.generate(tf.ones([2, 8, 1]))
        self.assertEqual(result.quantiles, (0.1, 0.5, 0.9))
        np.testing.assert_allclose(result.quantile(0.1), result.quantile_values[..., 0])

    def test_sampled_parameters_retain_batch_and_sample_axes(self):
        model = AutoModelForForecasting.from_config(
            AutoConfig.for_model("bert"), output_chunk_length=2, head="distribution"
        )
        result = model.generate(tf.ones([2, 8, 1]), num_samples=3, seed=7)
        self.assertEqual(result.samples.shape, (2, 3, 2, 1))
        for parameter in result.distribution_params.values():
            self.assertEqual(parameter.shape, result.samples.shape)
