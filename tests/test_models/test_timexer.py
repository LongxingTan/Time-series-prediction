import unittest

import tensorflow as tf

import tfts
from tfts.models.timexer import TimeXer, TimeXerConfig


class TimeXerTest(unittest.TestCase):
    def test_config(self):
        """Test configuration initialization."""
        config = TimeXerConfig(
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_attention_heads=4,
            patch_len=4,
        )
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(config.intermediate_size, 32)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.patch_len, 4)

    def test_model_output_shape(self):
        """Test model output shape."""
        train_sequence_length = 16
        predict_sequence_length = 8
        config = TimeXerConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_attention_heads=4, patch_len=4)
        model = TimeXer(predict_sequence_length=predict_sequence_length, config=config)

        x = tf.random.normal([2, train_sequence_length, 3])
        y = model(x)

        # Check output shape
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[1], predict_sequence_length)

    def test_model_direct_instantiation(self):
        """Test model direct instantiation."""
        config = TimeXerConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_attention_heads=4, patch_len=4)
        model = TimeXer(predict_sequence_length=8, config=config)
        self.assertIsNotNone(model)

        # Test forward pass
        x = tf.random.normal([2, 12, 4])
        y = model(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 8)

    def test_use_norm_both_paths(self):
        """TimeXer runs with instance normalization on or off."""
        config_on = TimeXerConfig(hidden_size=16, num_layers=1, num_attention_heads=4, patch_len=4, use_norm=True)
        config_off = TimeXerConfig(hidden_size=16, num_layers=1, num_attention_heads=4, patch_len=4, use_norm=False)
        x = tf.random.normal([2, 16, 3])
        y_on = TimeXer(predict_sequence_length=6, config=config_on)(x)
        y_off = TimeXer(predict_sequence_length=6, config=config_off)(x)
        self.assertEqual(y_on.shape, (2, 6, 3))
        self.assertEqual(y_off.shape, (2, 6, 3))

    def test_hidden_states_contract(self):
        config = TimeXerConfig(hidden_size=16, num_layers=1, num_attention_heads=4, patch_len=4)
        hidden = TimeXer(predict_sequence_length=6, config=config)(
            tf.random.normal([2, 16, 3]), output_hidden_states=True
        )
        self.assertEqual(hidden.shape[-1], config.hidden_size)

    # def test_train(self):
    #     """Test training loop."""
    #     train, valid = tfts.get_data("sine", test_size=0.1)
    #     config = TimeXerConfig(hidden_size=16, intermediate_size=32, num_layers=1, num_attention_heads=4, patch_len=4)
    #     model = TimeXer(predict_sequence_length=8, config=config)
    #
    #     model.build_model(train[0].shape)
    #     model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss="mse")
    #     model.fit(train[0], train[1], validation_data=valid, epochs=1, verbose=0)
    #
    #     y_test = model.predict(valid[0])
    #     self.assertEqual(y_test.shape[0], valid[1].shape[0])


if __name__ == "__main__":
    unittest.main()
