from typing import Any, Dict
import unittest

import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer
from tfts.models.gpt import GPT, GPTConfig


class GPTTest(unittest.TestCase):
    def test_config(self):
        """Test configuration initialization."""
        config = GPTConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            max_position_embeddings=256,
        )
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.max_position_embeddings, 256)

    def test_model_output_shape(self):
        """Test model output shape."""
        train_sequence_length = 14
        predict_sequence_length = 7
        config = GPTConfig(hidden_size=32, num_layers=1)
        model = GPT(predict_sequence_length=predict_sequence_length, config=config)

        x = tf.random.normal([2, train_sequence_length, 3])
        y = model(x)

        # Check output shape
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[1], predict_sequence_length)

    def test_model_direct_instantiation(self):
        """Test model direct instantiation."""
        config = GPTConfig(hidden_size=32, num_layers=1)
        model = GPT(predict_sequence_length=8, config=config)
        self.assertIsNotNone(model)

        # Test forward pass
        x = tf.random.normal([2, 10, 3])
        y = model(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 8)


if __name__ == "__main__":
    unittest.main()
