import unittest
from unittest.mock import MagicMock, Mock, patch

import tensorflow as tf

from tfts.tasks.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    """Test Pipeline class from the second document"""

    def setUp(self):
        # Create a mock config
        self.mock_cfg = Mock()
        self.mock_cfg.model.name = "test_model"
        self.mock_cfg.model.train_sequence_length = 10
        self.mock_cfg.model.predict_sequence_length = 5
        self.mock_cfg.model.n_features = 3
        self.mock_cfg.model.n_outputs = 1
        self.mock_cfg.training.loss = "MeanSquaredError"
        self.mock_cfg.training.optimizer = "Adam"
        self.mock_cfg.training.learning_rate = 0.001
        self.mock_cfg.training.epochs = 10

    @patch("tensorflow.config.list_physical_devices")
    def test_setup_strategy_multi_gpu(self, mock_list_devices):
        """Test strategy setup with multiple GPUs"""
        mock_list_devices.return_value = ["GPU:0", "GPU:1"]

        pipeline = Pipeline(self.mock_cfg)

        self.assertIsInstance(pipeline.strategy, tf.distribute.Strategy)

    @patch("tensorflow.config.list_physical_devices")
    def test_setup_strategy_single_gpu(self, mock_list_devices):
        """Test strategy setup with single GPU"""
        mock_list_devices.return_value = ["GPU:0"]

        pipeline = Pipeline(self.mock_cfg)

        self.assertIsInstance(pipeline.strategy, tf.distribute.Strategy)

    @patch("tensorflow.config.list_physical_devices")
    def test_setup_strategy_cpu(self, mock_list_devices):
        """Test strategy setup with CPU only"""
        mock_list_devices.return_value = []

        pipeline = Pipeline(self.mock_cfg)

        self.assertIsInstance(pipeline.strategy, tf.distribute.Strategy)
