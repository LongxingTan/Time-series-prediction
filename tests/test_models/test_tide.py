from typing import Any, Dict
import unittest

import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer
from tfts.models.tide import Tide, TideConfig


class TideTest(unittest.TestCase):
    def test_config(self):
        """Test configuration initialization."""
        config = TideConfig(
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
        )
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_attention_heads, 4)

    def test_model_output_shape(self):
        """Test model output shape."""
        train_sequence_length = 14
        predict_sequence_length = 7
        config = TideConfig(hidden_size=64, num_layers=1)
        model = Tide(predict_sequence_length=predict_sequence_length, config=config)

        x = tf.random.normal([2, train_sequence_length, 3])
        y = model(x)

        # Check output shape
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[1], predict_sequence_length)

    def test_model_direct_instantiation(self):
        """Test model direct instantiation."""
        config = TideConfig(hidden_size=64, num_layers=1)
        model = Tide(predict_sequence_length=8, config=config)
        self.assertIsNotNone(model)

        # Test forward pass
        x = tf.random.normal([2, 10, 3])
        y = model(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 8)

    # def test_train(self):
    #     """Test training loop."""
    #     train, valid = tfts.get_data("sine", test_size=0.1)
    #     config = TideConfig(hidden_size=64, num_layers=1)
    #     model = Tide(predict_sequence_length=8, config=config)

    #     # Build the model
    #     model.build_model(train[0].shape)
    #     model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss="mse")
    #     model.fit(train[0], train[1], validation_data=valid, epochs=1, verbose=0)

    #     y_test = model.predict(valid[0])
    #     self.assertEqual(y_test.shape[0], valid[1].shape[0])


if __name__ == "__main__":
    unittest.main()
