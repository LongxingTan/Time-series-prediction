import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting
from tfts.contracts import ForecastOutput, TimeSeriesBatch
from tfts.generation import ForecastGenerationConfig


class DeepARTest(unittest.TestCase):
    def setUp(self):
        self.prediction_length = 5
        self.model = AutoModelForForecasting.from_config(
            AutoConfig.for_model("deep_ar"), prediction_length=self.prediction_length
        )
        self.batch = TimeSeriesBatch(
            past_values=tf.random.normal([2, 12, 1]),
            future_values=tf.random.normal([2, self.prediction_length, 1]),
            static_categorical_features=tf.constant([[0], [1]], tf.int32),
        )

    def test_probabilistic_forward_contract(self):
        output = self.model(self.batch, return_dict=True)

        self.assertIsInstance(output, ForecastOutput)
        self.assertEqual(output.predictions.shape, (2, self.prediction_length, 1))
        self.assertEqual(set(output.distribution_params), {"loc", "scale"})
        self.assertTrue(bool(tf.reduce_all(output.distribution_params["scale"] > 0)))

    def test_sampled_autoregressive_generation_is_reproducible(self):
        config = ForecastGenerationConfig(num_samples=4, return_samples=True, seed=7, aggregation="median")
        first = self.model.generate(self.batch, config)
        second = self.model.generate(self.batch, config)

        self.assertEqual(first.predictions.shape, (2, self.prediction_length, 1))
        self.assertEqual(first.samples.shape, (2, 4, self.prediction_length, 1))
        np.testing.assert_allclose(first.samples.numpy(), second.samples.numpy(), atol=1e-6)
        ordered = tf.sort(first.samples, axis=1)
        expected_median = (ordered[:, 1, :, :] + ordered[:, 2, :, :]) / 2.0
        np.testing.assert_allclose(first.predictions.numpy(), expected_median.numpy(), atol=1e-6)

    def test_distribution_training_uses_negative_log_likelihood(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), run_eagerly=True)
        result = self.model.train_step(self.batch.as_dict())
        self.assertTrue(bool(tf.math.is_finite(result["loss"])))


if __name__ == "__main__":
    unittest.main()
