import unittest

import tensorflow as tf

import tfts
from tfts.models.timesnet import TimesNet, TimesNetConfig


class TimesNetTest(unittest.TestCase):
    def test_config(self):
        """Test configuration initialization."""
        config = TimesNetConfig(
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            top_k=5,
            num_kernels=6,
        )
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(config.intermediate_size, 32)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.num_kernels, 6)

    def test_model_output_shape(self):
        """Test model output shape."""
        train_sequence_length = 48
        predict_sequence_length = 12
        config = TimesNetConfig(hidden_size=16, intermediate_size=32, num_layers=1)
        model = TimesNet(predict_sequence_length=predict_sequence_length, config=config)

        x = tf.random.normal([2, train_sequence_length, 3])
        y = model(x)

        # Check output shape
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[1], predict_sequence_length)

    def test_model_varied_input_length(self):
        """TimesNet handles arbitrary sequence lengths thanks to FFT period discovery."""
        config = TimesNetConfig(hidden_size=16, intermediate_size=32, num_layers=1)
        model = TimesNet(predict_sequence_length=8, config=config)

        for length in (16, 40, 63):
            x = tf.random.normal([2, length, 4])
            y = model(x)
            self.assertEqual(y.shape, (2, 8, 4), f"failed for length={length}")

    def test_model_direct_instantiation(self):
        """Test model direct instantiation."""
        config = TimesNetConfig(hidden_size=16, num_layers=1)
        model = TimesNet(predict_sequence_length=8, config=config)
        self.assertIsNotNone(model)

        # Test forward pass
        x = tf.random.normal([2, 32, 3])
        y = model(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 8)

    # def test_train(self):
    #     """Test training loop."""
    #     train, valid = tfts.get_data("sine", test_size=0.1)
    #     config = TimesNetConfig(hidden_size=16, intermediate_size=32, num_layers=1)
    #     model = TimesNet(predict_sequence_length=8, config=config)
    #
    #     model.build_model(train[0].shape)
    #     model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss="mse")
    #     model.fit(train[0], train[1], validation_data=valid, epochs=1, verbose=0)
    #
    #     y_test = model.predict(valid[0])
    #     self.assertEqual(y_test.shape[0], valid[1].shape[0])


if __name__ == "__main__":
    unittest.main()
