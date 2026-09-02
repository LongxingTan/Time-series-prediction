import tempfile
import unittest

import numpy as np
import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer, Trainer
from tfts.models.tcn import TCN, TCNConfig
from tfts.training import TrainingArguments

# Smoke test pinning a single-device strategy so it runs identically on CI and
# any multi-GPU host.
_SINGLE_DEVICE_ARGS = TrainingArguments(output_dir="./weights", strategy="default")


class TCNTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        model = TCN(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([16, 160, 36])
        y = model(x)
        self.assertEqual(y.shape, (16, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        train, valid = tfts.get_data("sine", test_size=0.1)
        config = AutoConfig.for_model("tcn")
        model = AutoModel.from_config(config=config, predict_sequence_length=8)
        trainer = KerasTrainer(model, args=_SINGLE_DEVICE_ARGS)
        trainer.train(train, valid, optimizer=tf.keras.optimizers.Adam(0.003), epochs=1)
        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)

    def test_from_pretrained_can_continue_training(self):
        config = TCNConfig(
            kernel_sizes=[2, 2],
            dilation_rates=[1, 2],
            filters=4,
            dense_hidden_size=3,
        )
        model = TCN(predict_sequence_length=2, config=config)
        model.build_model(tf.keras.Input(shape=(10, 1)))
        sample = np.random.default_rng(1).normal(size=(2, 10, 1)).astype(np.float32)
        expected = model.predict(sample, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            restored = TCN.from_pretrained(tmpdir)
            actual = restored.predict(sample, verbose=0)

            restored.compile(optimizer="adam", loss="mse")
            restored.train_on_batch(sample, np.zeros_like(actual))

        self.assertEqual(restored.predict_sequence_length, 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
