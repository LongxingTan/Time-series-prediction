import unittest

import numpy as np
import tensorflow as tf

from tfts.distributions import NormalOutput
from tfts.tasks.auto_task import (
    AnomalyHead,
    AnomalyOutput,
    ClassificationHead,
    ClassificationOutput,
    DistributionHead,
    GaussianHead,
    PredictionHead,
    PredictionOutput,
    QuantileHead,
    SegmentationHead,
)
from tfts.tasks.base import BaseHead, BaseTask, ModelOutput


class TestPredictionOutput(unittest.TestCase):
    """Test PredictionOutput dataclass"""

    def test_initialization(self):
        """Test PredictionOutput can be initialized"""
        logits = tf.constant([[0.1, 0.9]])
        hidden = tf.constant([[[0.5]]])
        output = PredictionOutput(prediction_logits=logits, last_hidden_state=hidden)

        self.assertTrue(tf.reduce_all(output.prediction_logits == logits))
        self.assertTrue(tf.reduce_all(output.last_hidden_state == hidden))

    def test_all_fields(self):
        """Test all fields can be set"""
        logits = tf.constant([[0.1, 0.9]])
        hidden = tf.constant([[[0.5]]])
        attentions = (tf.constant([0.1]), tf.constant([0.2]))
        loss = tf.constant(0.5)

        output = PredictionOutput(
            prediction_logits=logits,
            last_hidden_state=hidden,
            hidden_states=(hidden,),
            attentions=attentions,
            loss=loss,
        )

        self.assertIsNotNone(output.prediction_logits)
        self.assertIsNotNone(output.last_hidden_state)
        self.assertIsNotNone(output.hidden_states)
        self.assertIsNotNone(output.attentions)
        self.assertIsNotNone(output.loss)


class TestPredictionHead(unittest.TestCase):
    """Test PredictionHead layer"""

    def test_initialization(self):
        """Test PredictionHead can be initialized"""
        head = PredictionHead()
        self.assertIsInstance(head, tf.keras.layers.Layer)
        self.assertIsInstance(head, BaseTask)

    def test_inheritance(self):
        """Test PredictionHead inherits from correct classes"""
        head = PredictionHead()
        self.assertTrue(isinstance(head, tf.keras.layers.Layer))
        self.assertTrue(isinstance(head, BaseTask))

    def test_call_projects_hidden_states_to_forecast_contract(self):
        head = PredictionHead(predict_sequence_length=4, num_labels=2)
        outputs = head(tf.random.normal([3, 9, 16]))
        self.assertEqual(outputs.shape, (3, 4, 2))

    def test_residual_last_value_is_applied_after_projection(self):
        head = PredictionHead(predict_sequence_length=3, num_labels=1, residual="last_value")
        hidden = tf.zeros([2, 5, 4])
        inputs = tf.reshape(tf.range(10, dtype=tf.float32), [2, 5, 1])
        head(hidden, inputs=inputs)
        head.projection.kernel.assign(tf.zeros_like(head.projection.kernel))
        head.projection.bias.assign(tf.zeros_like(head.projection.bias))

        outputs = head(hidden, inputs=inputs)
        expected = tf.tile(inputs[:, -1:, :], [1, 3, 1])
        self.assertTrue(tf.reduce_all(outputs == expected))


class TestComposableHeads(unittest.TestCase):
    def test_quantile_head_contract(self):
        outputs = QuantileHead((0.1, 0.5, 0.9), num_labels=2)(tf.random.normal([4, 7, 8]))
        self.assertEqual(outputs.shape, (4, 7, 2, 3))

    def test_distribution_head_connects_distribution_output(self):
        head = DistributionHead(NormalOutput(target_dim=2))
        outputs = head(tf.random.normal([4, 7, 8]))
        self.assertEqual(set(outputs), {"loc", "scale"})
        self.assertEqual(outputs["loc"].shape, (4, 7, 2))
        self.assertTrue(tf.reduce_all(outputs["scale"] > 0))

    def test_all_layer_heads_share_base_contract(self):
        self.assertIsInstance(PredictionHead(), BaseHead)
        self.assertIsInstance(ClassificationHead(), BaseHead)
        self.assertIsInstance(QuantileHead((0.5,)), BaseHead)


class TestClassificationHead(unittest.TestCase):
    """Test ClassificationHead layer"""

    def setUp(self):
        self.num_labels = 3
        self.dense_units = (128, 64)
        self.head = ClassificationHead(num_labels=self.num_labels, dense_units=self.dense_units)

    def test_initialization(self):
        """Test ClassificationHead initialization"""
        self.assertIsInstance(self.head, tf.keras.layers.Layer)
        self.assertEqual(len(self.head.intermediate_dense_layers), 2)

    def test_default_initialization(self):
        """Test default parameters"""
        head = ClassificationHead()
        self.assertEqual(len(head.intermediate_dense_layers), 1)

    def test_call_shape(self):
        """Test output shape is correct"""
        batch_size = 4
        seq_length = 10
        hidden_size = 64
        inputs = tf.random.normal([batch_size, seq_length, hidden_size])

        outputs = self.head(inputs)

        self.assertEqual(outputs.shape, (batch_size, self.num_labels))

    def test_call_output_range(self):
        """Test softmax outputs sum to 1"""
        inputs = tf.random.normal([2, 10, 64])
        outputs = self.head(inputs)

        # Softmax outputs should sum to 1
        sums = tf.reduce_sum(outputs, axis=-1)
        self.assertTrue(tf.reduce_all(tf.abs(sums - 1.0) < 1e-5))

    def test_pooling_layer(self):
        """Test pooling layer is configured correctly"""
        self.assertIsInstance(self.head.pooling, tf.keras.layers.GlobalAveragePooling1D)

    def test_single_dense_unit(self):
        """Test with single intermediate dense layer"""
        head = ClassificationHead(num_labels=2, dense_units=(64,))
        inputs = tf.random.normal([2, 10, 32])
        outputs = head(inputs)
        self.assertEqual(outputs.shape, (2, 2))

    def test_no_intermediate_layers(self):
        """Test with no intermediate dense layers"""
        head = ClassificationHead(num_labels=5, dense_units=())
        inputs = tf.random.normal([2, 10, 32])
        outputs = head(inputs)
        self.assertEqual(outputs.shape, (2, 5))


class TestClassificationOutput(unittest.TestCase):
    """Test ClassificationOutput dataclass"""

    def test_initialization(self):
        """Test ClassificationOutput initialization"""
        logits = tf.constant([[0.1, 0.9]])
        hidden = (tf.constant([[[0.5]]]),)
        loss = tf.constant(0.5)

        output = ClassificationOutput(logits=logits, hidden_states=hidden, loss=loss)

        self.assertTrue(tf.reduce_all(output.logits == logits))
        self.assertEqual(output.hidden_states, hidden)
        self.assertTrue(tf.reduce_all(output.loss == loss))


class TestAnomalyHead(unittest.TestCase):
    """Test AnomalyHead layer"""

    def setUp(self):
        self.train_sequence_length = 5
        self.head = AnomalyHead(train_sequence_length=self.train_sequence_length)

    def test_initialization(self):
        """Test AnomalyHead initialization"""
        self.assertEqual(self.head.train_sequence_length, self.train_sequence_length)

    def test_call_with_numpy(self):
        """Test call with numpy arrays"""
        y_pred = np.array([[0.5, 0.2], [0.7, 0.3], [0.1, 0.1], [0.9, 0.4], [0.2, 0.3]])
        y_test = np.array([[0.6, 0.1], [0.8, 0.2], [0.1, 0.0], [1.0, 0.5], [0.3, 0.4]])

        m_dist = self.head(y_pred, y_test)

        self.assertEqual(len(m_dist), len(y_pred) + self.train_sequence_length)
        # First train_sequence_length elements should be 0
        for i in range(self.train_sequence_length):
            self.assertEqual(m_dist[i], 0)
        # Remaining should be non-negative
        for dist in m_dist[self.train_sequence_length :]:
            self.assertGreaterEqual(dist, 0)

    def test_call_with_tensors(self):
        """Test call with TensorFlow tensors"""
        y_pred = tf.constant([[0.5, 0.2], [0.7, 0.3], [0.1, 0.1]])
        y_test = tf.constant([[0.6, 0.1], [0.8, 0.2], [0.1, 0.0]])

        m_dist = self.head(y_pred, y_test)

        self.assertEqual(len(m_dist), 3 + self.train_sequence_length)

    def test_call_with_3d_input(self):
        """Test call with 3D input (batch dimension)"""
        y_pred = np.array([[[0.5], [0.7], [0.1]]])
        y_test = np.array([[[0.6], [0.8], [0.1]]])

        m_dist = self.head(y_pred, y_test)

        self.assertEqual(len(m_dist), 3 + self.train_sequence_length)

    def test_mahala_distance(self):
        """Test Mahalanobis distance calculation"""
        x = np.array([0.5, 0.2])
        mean = np.array([0.6, 0.1])
        cov = np.array([[0.01, 0.001], [0.001, 0.02]])

        m_dist = AnomalyHead.mahala_distantce(x, mean, cov)

        self.assertIsInstance(m_dist, (np.floating, float))
        self.assertGreaterEqual(m_dist, 0)

    def test_mahala_distance_with_zero_covariance(self):
        """Test Mahalanobis distance handles zero covariance"""
        x = np.array([0.5, 0.2])
        mean = np.array([0.6, 0.1])
        cov = np.zeros((2, 2))  # Zero covariance matrix

        # Should not raise an error due to epsilon regularization
        m_dist = AnomalyHead.mahala_distantce(x, mean, cov)
        self.assertIsInstance(m_dist, (np.floating, float))

    def test_mahala_distance_epsilon_parameter(self):
        """Test custom epsilon parameter"""
        x = np.array([1.0, 2.0])
        mean = np.array([1.5, 2.5])
        cov = np.eye(2) * 0.01

        m_dist1 = AnomalyHead.mahala_distantce(x, mean, cov, epsilon=1e-8)
        m_dist2 = AnomalyHead.mahala_distantce(x, mean, cov, epsilon=1e-6)

        # Different epsilon values should produce different results
        self.assertNotEqual(m_dist1, m_dist2)


class TestAnomalyOutput(unittest.TestCase):
    """Test AnomalyOutput dataclass"""

    def test_initialization(self):
        """Test AnomalyOutput initialization"""
        scores = tf.constant([0.1, 0.2, 0.3])
        logits = tf.constant([[0.5]])
        loss = tf.constant(0.1)

        output = AnomalyOutput(anomaly_scores=scores, reconstruction_logits=logits, loss=loss)

        self.assertTrue(tf.reduce_all(output.anomaly_scores == scores))
        self.assertTrue(tf.reduce_all(output.reconstruction_logits == logits))
        self.assertTrue(tf.reduce_all(output.loss == loss))


class TestGaussianHead(unittest.TestCase):
    """Test GaussianHead layer"""

    def setUp(self):
        self.units = 32
        self.head = GaussianHead(units=self.units)

    def test_initialization(self):
        """Test GaussianHead initialization"""
        self.assertEqual(self.head.units, self.units)

    def test_build(self):
        """Test build method creates correct weights"""
        input_shape = (None, 10, 16)
        self.head.build(input_shape)

        self.assertEqual(self.head.weight1.shape, (16, self.units))
        self.assertEqual(self.head.weight2.shape, (16, self.units))
        self.assertEqual(self.head.bias1.shape, (self.units,))
        self.assertEqual(self.head.bias2.shape, (self.units,))

    def test_call_output_shape(self):
        """Test call returns correct shapes"""
        x = tf.random.normal([2, 10, 16])
        mu, sig = self.head(x)

        self.assertEqual(mu.shape, (2, 10, self.units))
        self.assertEqual(sig.shape, (2, 10, self.units))

    def test_call_output_values(self):
        """Test sigma is positive"""
        x = tf.random.normal([2, 10, 16])
        mu, sig = self.head(x)

        # Sigma should be positive due to log1p(exp(x)) + epsilon
        self.assertTrue(tf.reduce_all(sig > 0))

    def test_get_config(self):
        """Test get_config returns correct configuration"""
        config = self.head.get_config()

        self.assertIn("units", config)
        self.assertEqual(config["units"], self.units)

    def test_different_input_channels(self):
        """Test with different input channel sizes"""
        head = GaussianHead(units=64)
        x = tf.random.normal([4, 20, 8])
        mu, sig = head(x)

        self.assertEqual(mu.shape, (4, 20, 64))
        self.assertEqual(sig.shape, (4, 20, 64))

    def test_sigma_minimum_value(self):
        """Test sigma has minimum value due to epsilon"""
        x = tf.zeros([1, 1, 4])
        mu, sig = self.head(x)

        # Even with zero input, sigma should be at least epsilon (1e-7)
        self.assertTrue(tf.reduce_all(sig >= 1e-7))


class TestSegmentationHead(unittest.TestCase):
    """Test SegmentationHead layer"""

    def test_initialization(self):
        """Test SegmentationHead can be initialized"""
        head = SegmentationHead()
        self.assertIsInstance(head, tf.keras.layers.Layer)
        self.assertIsInstance(head, BaseTask)

    def test_inheritance(self):
        """Test SegmentationHead inherits from correct classes"""
        head = SegmentationHead()
        self.assertTrue(isinstance(head, tf.keras.layers.Layer))
        self.assertTrue(isinstance(head, BaseTask))


if __name__ == "__main__":
    unittest.main()
