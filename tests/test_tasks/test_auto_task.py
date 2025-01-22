import unittest

import numpy as np
import tensorflow as tf

from tfts.tasks.auto_task import AnomalyHead, GaussianHead


class TestAnomalyHead(unittest.TestCase):
    def setUp(self):
        self.train_sequence_length = 5
        self.layer = AnomalyHead(train_sequence_length=self.train_sequence_length)

    def test_call(self):
        y_pred = np.array([[0.5, 0.2], [0.7, 0.3], [0.1, 0.1], [0.9, 0.4], [0.2, 0.3]])
        y_test = np.array([[0.6, 0.1], [0.8, 0.2], [0.1, 0.0], [1.0, 0.5], [0.3, 0.4]])

        m_dist = self.layer(y_pred, y_test)

        # self.assertEqual(len(m_dist), self.train_sequence_length)

        # Test that Mahalanobis distance is calculated for each error (it's non-negative)
        for dist in m_dist:
            self.assertGreaterEqual(dist, 0)

    def test_mahala_distance(self):
        x = np.array([0.5, 0.2])
        mean = np.array([0.6, 0.1])
        cov = np.array([[0.01, 0.001], [0.001, 0.02]])

        # Calculate Mahalanobis distance using the layer's method
        m_dist = self.layer.mahala_distantce(x, mean, cov)

        # The Mahalanobis distance should be a scalar value (float)
        self.assertIsInstance(m_dist, np.float64)


class DeepARLayerTest(unittest.TestCase):
    def test_gaussian_layer(self):
        hidden_size = 32
        layer = GaussianHead(hidden_size)

        x = tf.random.normal([2, 10, 1])
        mu, sig = layer(x)
        self.assertEqual(mu.shape, (2, 10, hidden_size))
        self.assertEqual(sig.shape, (2, 10, hidden_size))
