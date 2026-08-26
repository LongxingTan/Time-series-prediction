import tempfile
import unittest

import numpy as np
import tensorflow as tf

from tfts import (
    AutoConfig,
    AutoModel,
    AutoModelForAnomalyDetection,
    AutoModelForForecasting,
    AutoModelForImputation,
    AutoModelForTimeSeriesClassification,
)
from tfts.contracts import (
    AnomalyDetectionOutput,
    ClassificationOutput,
    ForecastOutput,
    ImputationOutput,
    TimeSeriesBatch,
)


class TestAutoModel(unittest.TestCase):
    def test_factory_returns_a_task_model_with_structured_output(self):
        config = AutoConfig.for_model("dlinear")
        model = AutoModelForForecasting.from_config(config, prediction_length=3)
        batch = TimeSeriesBatch(tf.random.normal([2, 12, 2]))

        output = model(batch, return_dict=True)

        self.assertIsInstance(output, ForecastOutput)
        self.assertEqual(output.predictions.shape, (2, 3, 1))
        self.assertIs(model.config, model.backbone.config)

    def test_task_dispatch_and_capability_validation(self):
        config = AutoConfig.for_model("bert")
        classifier = AutoModel.from_config(config, task="classification", num_labels=3)
        output = classifier(tf.random.normal([2, 10, 4]), return_dict=True)

        self.assertIsInstance(classifier, AutoModelForTimeSeriesClassification.model_class)
        self.assertIsInstance(output, ClassificationOutput)
        self.assertEqual(output.logits.shape, (2, 3))
        np.testing.assert_allclose(tf.reduce_sum(output.probabilities, axis=-1).numpy(), np.ones(2), atol=1e-6)

        with self.assertRaisesRegex(ValueError, "does not support imputation"):
            AutoModelForImputation.from_config(AutoConfig.for_model("dlinear"))

    def test_imputation_preserves_observed_values(self):
        model = AutoModelForImputation.from_config(AutoConfig.for_model("bert"), target_dim=2)
        values = tf.random.normal([2, 8, 2])
        mask = tf.constant([[[1.0, 0.0]] * 8] * 2)
        output = model(TimeSeriesBatch(past_values=values, past_observed_mask=mask), return_dict=True)

        self.assertIsInstance(output, ImputationOutput)
        np.testing.assert_allclose((output.imputed_values * mask).numpy(), (values * mask).numpy(), atol=1e-6)

    def test_anomaly_calibration_is_a_separate_fitted_stage(self):
        model = AutoModelForAnomalyDetection.from_config(AutoConfig.for_model("bert"))
        batch = TimeSeriesBatch(tf.random.normal([2, 8, 1]))
        model.calibrate(batch)
        output = model.detect(batch)

        self.assertIsInstance(output, AnomalyDetectionOutput)
        self.assertEqual(output.labels.shape, (2, 8))
        self.assertTrue(bool(tf.math.is_finite(output.threshold)))

    def test_task_artifact_round_trip(self):
        config = AutoConfig.for_model("rnn")
        model = AutoModelForForecasting.from_config(config, prediction_length=3)
        sample = tf.random.normal([2, 8, 2])
        expected = model(sample)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = AutoModel.from_pretrained(directory)
            actual = restored(sample)

        self.assertEqual(restored.task_config.prediction_length, 3)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
