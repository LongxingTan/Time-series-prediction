import unittest

import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer, Trainer
from tfts.models.rnn import RNN, RNN2, Encoder, RNNConfig
from tfts.training import TrainingArguments

# Smoke test pinning a single-device strategy so it runs identically on CI and
# any multi-GPU host.
_SINGLE_DEVICE_ARGS = TrainingArguments(output_dir="./weights", strategy="default")


class RNNTest(unittest.TestCase):
    def test_configured_encoder_depth_and_direction(self):
        # Given an explicitly stacked, bidirectional RNN configuration.
        config = RNNConfig(rnn_hidden_size=4, num_stacked_layers=2, bi_direction=True)
        model = RNN(predict_sequence_length=3, config=config)
        # When the configured model predicts a sequence.
        prediction = model(tf.ones([2, 8, 5]))
        # Then the configured architecture actually executes.
        self.assertEqual(prediction.shape, (2, 3, 1))
        self.assertEqual(len(model.encoder.rnn_layers), 2)
        self.assertTrue(all(isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.encoder.rnn_layers))

    def test_encoder_stack_and_bidirectional_states(self):
        # Given GRU and LSTM encoders with multiple recurrent layers.
        for rnn_type in ("gru", "lstm"):
            for bidirectional in (False, True):
                with self.subTest(rnn_type=rnn_type, bidirectional=bidirectional):
                    encoder = Encoder(4, rnn_type=rnn_type, num_stacked_layers=2, bi_direction=bidirectional)
                    # When the final sequence and state are requested.
                    sequence, state = encoder(tf.ones([2, 8, 5]))
                    # Then deeper layers accept the preceding output width,
                    # and both directions contribute their hidden states.
                    width = 8 if bidirectional else 4
                    self.assertEqual(sequence.shape, (2, 8, width))
                    self.assertEqual(state.shape, (2, width * (2 if rnn_type == "lstm" else 1)))

    def test_aligned_predictions_train_and_use_future_context(self):
        # Given a projected bidirectional model for aligned sequence regression.
        tf.keras.utils.set_random_seed(42)
        config = RNNConfig(
            rnn_hidden_size=4,
            dense_hidden_size=8,
            bi_direction=True,
            num_stacked_layers=2,
            return_sequences=True,
            input_projection=True,
        )
        model = RNN(predict_sequence_length=8, config=config)
        x = tf.random.normal([2, 8, 5])
        # When per-step predictions and training gradients are computed.
        with tf.GradientTape() as tape:
            prediction = model(x)
            loss = tf.reduce_mean(tf.square(prediction - 1))
        gradients = tape.gradient(loss, model.trainable_variables)
        changed = tf.concat([x[:, :1], x[:, 1:] + 5], axis=1)
        changed_prediction = model(changed)
        # Then all layers train, and the first prediction uses later context.
        self.assertEqual(prediction.shape, (2, 8, 1))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertGreater(float(tf.reduce_max(tf.abs(prediction[:, :1] - changed_prediction[:, :1]))), 1e-8)

    def test_encoder_shape_and_config_contract(self):
        # Given stacked encoders covering both cell types and state modes.
        for rnn_type in ("gru", "lstm"):
            for bidirectional in (False, True):
                for return_state in (False, True):
                    with self.subTest(cell=rnn_type, bidirectional=bidirectional, states=return_state):
                        encoder = Encoder(
                            4,
                            rnn_type=rnn_type,
                            num_stacked_layers=2,
                            bi_direction=bidirectional,
                            return_state=return_state,
                        )
                        x = tf.ones([2, 8, 5])
                        expected = encoder(x)
                        # When the layer is reconstructed from its config.
                        restored = Encoder.from_config(encoder.get_config())
                        restored(x)
                        restored.set_weights(encoder.get_weights())
                        actual = restored(x)
                        # Then declared shapes and predictions survive.
                        shapes = tf.nest.map_structure(lambda value: tuple(value.shape), actual)
                        self.assertEqual(shapes, encoder.compute_output_shape(x.shape))
                        for before, after in zip(tf.nest.flatten(expected), tf.nest.flatten(actual)):
                            tf.debugging.assert_near(before, after)

    def test_model(self):
        predict_sequence_length = 8
        model = RNN(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_aligned_mode_rejects_a_different_forecast_horizon(self):
        # Given aligned output with a declared length of eight steps.
        model = RNN(8, RNNConfig(rnn_hidden_size=4, return_sequences=True))
        # When the input has a different length, then it cannot silently
        # satisfy a forecasting contract with the wrong output horizon.
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model(tf.ones([2, 7, 3]))

    def test_aligned_mode_ignores_padded_values(self):
        # Given a stacked bidirectional model and two padded timesteps.
        model = RNN(8, RNNConfig(rnn_hidden_size=4, bi_direction=True, num_stacked_layers=2, return_sequences=True))
        x = tf.random.normal([2, 8, 3])
        mask = tf.constant([[True] * 6 + [False] * 2] * 2)
        inputs = {"x": x, "encoder_feature": tf.zeros([2, 8, 0]), "padding_mask": mask}
        # When only the padded values change.
        original = model(inputs)
        changed = model({**inputs, "x": tf.concat([x[:, :6], x[:, 6:] + 100], axis=1)})
        # Then predictions at real timesteps are unchanged.
        tf.debugging.assert_near(original[:, :6], changed[:, :6])

    def test_aligned_model_config_roundtrip(self):
        # Given an aligned model with optional architecture settings.
        config = RNNConfig(
            rnn_hidden_size=4,
            dense_hidden_size=8,
            rnn_type="lstm",
            bi_direction=True,
            num_stacked_layers=2,
            return_sequences=True,
            input_projection=True,
        )
        model = RNN(8, config)
        x = tf.random.normal([2, 8, 3])
        expected = model(x)
        # When Keras reconstructs the model configuration and weights.
        restored = RNN.from_config(model.get_config())
        restored(x)
        restored.set_weights(model.get_weights())
        # Then the same configuration and weights preserve predictions.
        tf.debugging.assert_near(expected, restored(x))

    def test_train(self):
        train, valid = tfts.get_data("sine", test_size=0.1)
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, output_chunk_length=8)
        trainer = KerasTrainer(model, args=_SINGLE_DEVICE_ARGS)
        trainer.train(train, valid, optimizer=tf.keras.optimizers.Adam(0.003), epochs=1)
        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)

    def test_model2(self):
        predict_sequence_length = 8
        model = RNN2(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")


if __name__ == "__main__":
    unittest.main()
