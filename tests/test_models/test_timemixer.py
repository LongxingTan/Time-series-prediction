import unittest

import tensorflow as tf

import tfts
from tfts.models.timemixer import TimeMixer, TimeMixerConfig


class TimeMixerTest(unittest.TestCase):
    def test_config(self):
        """Test configuration initialization."""
        config = TimeMixerConfig(
            d_model=32,
            d_ff=64,
            e_layers=1,
            moving_avg=7,
            down_sampling_window=2,
            down_sampling_layers=1,
            down_sampling_method="avg",
        )
        self.assertEqual(config.d_model, 32)
        self.assertEqual(config.d_ff, 64)
        self.assertEqual(config.e_layers, 1)
        self.assertEqual(config.moving_avg, 7)
        self.assertEqual(config.down_sampling_window, 2)
        self.assertEqual(config.down_sampling_layers, 1)

    def test_model_output_shape(self):
        """Test model output shape."""
        train_sequence_length = 32
        predict_sequence_length = 8
        config = TimeMixerConfig(d_model=16, d_ff=32, e_layers=1, moving_avg=7)
        model = TimeMixer(predict_sequence_length=predict_sequence_length, config=config)

        x = tf.random.normal([2, train_sequence_length, 3])
        y = model(x)

        # Check output shape
        self.assertEqual(y.shape[0], 2)  # batch size
        self.assertEqual(y.shape[1], predict_sequence_length)
        self.assertEqual(y.shape[2], 3)  # all input channels predicted

    def test_model_direct_instantiation(self):
        """Test model direct instantiation."""
        config = TimeMixerConfig(d_model=16, d_ff=32, e_layers=1, moving_avg=7)
        model = TimeMixer(predict_sequence_length=6, config=config)
        self.assertIsNotNone(model)

        # Test forward pass
        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 6)

    def test_encoder_depth_uses_distinct_blocks(self):
        config = TimeMixerConfig(d_model=16, d_ff=32, e_layers=3, moving_avg=7)
        model = TimeMixer(predict_sequence_length=6, config=config)
        hidden = model(tf.random.normal([2, 16, 3]), output_hidden_states=True)
        self.assertEqual(hidden.shape, (2, 16, 16))
        self.assertEqual(len(model.pdm_blocks), 3)
        self.assertEqual(len({id(block) for block in model.pdm_blocks}), 3)

    # def test_train(self):
    #     """Test training loop."""
    #     train, valid = tfts.get_data("sine", test_size=0.1)
    #     config = TimeMixerConfig(d_model=16, d_ff=32, e_layers=1, moving_avg=7)
    #     model = TimeMixer(predict_sequence_length=8, config=config)
    #
    #     model.build_model(train[0].shape)
    #     model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss="mse")
    #     model.fit(train[0], train[1], validation_data=valid, epochs=1, verbose=0)
    #
    #     y_test = model.predict(valid[0])
    #     self.assertEqual(y_test.shape[0], valid[1].shape[0])


if __name__ == "__main__":
    unittest.main()
