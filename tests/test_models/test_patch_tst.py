from typing import Any, Dict
import unittest

import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer
from tfts.models.patch_tst import PatchTST, PatchTSTConfig


class PatchTSTTest(unittest.TestCase):
    def test_config(self):
        """Test configuration initialization."""
        config = PatchTSTConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            patch_size=16,
        )
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.patch_size, 16)

    def test_model_output_shape(self):
        """Test model output shape."""
        train_sequence_length = 32
        predict_sequence_length = 8
        config = AutoConfig.for_model("patch_tst")
        config.hidden_size = 32
        config.num_layers = 1
        config.patch_size = 8
        model = PatchTST(predict_sequence_length=predict_sequence_length, config=config)

        x = tf.random.normal([2, train_sequence_length, 3])
        y = model(x)

        # Check output shape
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[1], predict_sequence_length)

    def test_model_with_autoconfig(self):
        """Test model initialization from AutoConfig."""
        config = AutoConfig.for_model("patch_tst")
        config.hidden_size = 32
        config.num_layers = 1
        config.patch_size = 8

        model = AutoModel.from_config(config, predict_sequence_length=8)
        self.assertIsNotNone(model)

        # Test forward pass
        x = tf.random.normal([2, 32, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, 8, 1))

    # def test_train(self):
    #     """Test training loop."""
    #     train, valid = tfts.get_data("sine", test_size=0.1)
    #     config = AutoConfig.for_model("patch_tst")
    #     config.hidden_size = 32
    #     config.num_layers = 1
    #     config.patch_size = 8

    #     model = AutoModel.from_config(config, predict_sequence_length=8)
    #     trainer = KerasTrainer(model)

    #     trainer.train(train, valid, optimizer=tf.keras.optimizers.Adam(0.003), epochs=1)

    #     y_test = trainer.predict(valid[0])
    #     self.assertEqual(y_test.shape, valid[1].shape)


if __name__ == "__main__":
    unittest.main()
