import unittest

import numpy as np
import tensorflow as tf

from tfts.contracts import ForecastOutput
from tfts.distributions import NormalOutput
from tfts.tasks.anomaly import QuantileCalibrator, SquaredErrorScorer
from tfts.tasks.auto_task import (
    ClassificationHead,
    DistributionForecastHead,
    PointForecastHead,
    QuantileForecastHead,
    ReconstructionHead,
)
from tfts.tasks.base import BaseHead


class TestTaskHeads(unittest.TestCase):
    def test_heads_have_one_narrow_contract(self):
        heads = [
            PointForecastHead(3, 2),
            QuantileForecastHead(3, (0.1, 0.5, 0.9), 2),
            DistributionForecastHead(NormalOutput(2), 3),
            ClassificationHead(4),
            ReconstructionHead(2),
        ]
        self.assertTrue(all(isinstance(head, BaseHead) for head in heads))

    def test_forecast_head_shapes(self):
        hidden = tf.random.normal([4, 8, 16])
        point = PointForecastHead(3, 2)(hidden)
        quantiles = QuantileForecastHead(3, (0.1, 0.5, 0.9), 2)(hidden)
        params = DistributionForecastHead(NormalOutput(2), 3)(hidden)

        self.assertEqual(point.shape, (4, 3, 2))
        self.assertEqual(quantiles.shape, (4, 3, 2, 3))
        self.assertEqual(params["loc"].shape, (4, 3, 2))
        self.assertTrue(bool(tf.reduce_all(params["scale"] > 0)))

    def test_classification_pooling_ignores_padding(self):
        head = ClassificationHead(3, hidden_units=())
        valid = tf.random.normal([2, 3, 4])
        first = tf.concat([valid, tf.zeros([2, 2, 4])], axis=1)
        second = tf.concat([valid, tf.fill([2, 2, 4], 1000.0)], axis=1)
        mask = tf.constant([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])

        first_logits = head(first, padding_mask=mask)
        second_logits = head(second, padding_mask=mask)
        np.testing.assert_allclose(first_logits.numpy(), second_logits.numpy(), atol=1e-6)

    def test_model_output_attribute_and_mapping_views_stay_coherent(self):
        output = ForecastOutput(predictions=tf.ones([1, 2, 1]))
        output.samples = tf.zeros([1, 3, 2, 1])
        self.assertIs(output.samples, output["samples"])
        self.assertIs(output[0], output.predictions)


class TestAnomalyServices(unittest.TestCase):
    def test_scorer_and_calibrator_are_tensorflow_native(self):
        observed = tf.constant([[[0.0], [2.0], [5.0]]])
        reconstructed = tf.constant([[[0.0], [1.0], [2.0]]])
        scores = SquaredErrorScorer()(observed, reconstructed)
        calibrator = QuantileCalibrator(0.5)

        threshold = calibrator.fit(scores)
        labels = calibrator.predict(scores)

        self.assertAlmostEqual(float(threshold), 1.0)
        np.testing.assert_array_equal(labels.numpy(), [[0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
