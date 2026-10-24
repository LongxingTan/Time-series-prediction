"""End-to-end contracts for beginner array workflows."""

import tempfile
import unittest

import numpy as np
import tensorflow as tf

import tfts
from tfts.contracts import ForecastOutput, TimeSeriesBatch
from tfts.tasks.objectives import SequenceMeanSquaredError


class ArrayEstimatorTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        rng = np.random.default_rng(17)
        self.x = rng.normal(size=(12, 8, 3)).astype("float32")
        self.y = np.stack((2 * self.x[..., 1] + 10, self.x[..., 2] - 20), axis=-1)

    def test_aligned_training_and_saved_preprocessing(self):
        pipe = tfts.Estimator("sequence_regression", bi_direction=True, epochs=2, batch_size=4)
        weight = np.ones((8, 8), dtype="float32")
        weight[:, -1] = 0
        labels = self.y[:8].copy()
        labels[:, -1] = 100000
        pipe.fit(self.x[:8], labels, sample_weight=weight, validation_data=(self.x[8:], self.y[8:]), verbose=0)
        self.assertTrue(pipe.model_.config.bi_direction)
        self.assertEqual(pipe.model_.config.target_dim, 2)
        np.testing.assert_allclose(
            pipe.preprocessor_.target_scaler.offset, self.y[:8, :-1].mean(axis=(0, 1)), rtol=1e-6
        )
        np.testing.assert_allclose(
            pipe.preprocessor_.scalers["past_values"].offset, self.x[:8].mean(axis=(0, 1)), atol=1e-7
        )
        prediction = pipe.predict(self.x[8:])
        self.assertEqual(prediction.shape, self.y[8:].shape)
        self.assertTrue(np.isfinite(prediction).all())
        with tempfile.TemporaryDirectory() as directory:
            pipe.save(directory)
            restored = tfts.Estimator.load(directory)
            np.testing.assert_allclose(restored.predict(self.x[8:]), prediction, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "window shapes"):
            pipe.predict(self.x[8:, :, :2])

    def test_forecast_dimensions_inferred(self):
        pipe = tfts.Estimator("forecasting", epochs=1)
        y = self.y[:, :3]
        pipe.fit(self.x[:8], y[:8], validation_data=(self.x[8:], y[8:]), verbose=0)
        self.assertEqual(pipe.predict(self.x[8:]).shape, (4, 3, 2))
        with tempfile.TemporaryDirectory() as directory:
            pipe.save(directory)
            restored = tfts.Estimator.load(directory)
            np.testing.assert_allclose(restored.predict(self.x[8:]), pipe.predict(self.x[8:]), atol=1e-6)

    def test_no_implicit_validation_split_or_shape_broadcast(self):
        pipe = tfts.Estimator("sequence_regression")
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "one target timestep"):
            pipe.fit(self.x[:8], self.y[:8, :2], validation_data=(self.x[8:], self.y[8:, :2]))
        with self.assertRaisesRegex(ValueError, "sample_weight"):
            pipe.fit(self.x[:8], self.y[:8], sample_weight=np.ones((8, 1)), validation_data=(self.x[8:], self.y[8:]))

    def test_unsupported_models_and_silent_channel_dropping_rejected(self):
        for name in ("dlinear", "patch_tst", "tide"):
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, "sequence_regression"):
                    tfts.Estimator("sequence_regression", model=name).fit(self.x, self.x)
                model = tfts.AutoModel.from_config(tfts.AutoConfig.for_model(name), output_chunk_length=2)
                with self.assertRaisesRegex((ValueError, tf.errors.InvalidArgumentError), "target_dim"):
                    model(self.x)
        for name in ("dlinear", "patch_tst"):
            model = tfts.AutoModel.from_config(tfts.AutoConfig.for_model(name), output_chunk_length=2)
            with self.assertRaisesRegex(ValueError, "does not support covariates"):
                model(dict(past_values=self.x[..., :1], past_time_features=self.x[..., 1:]))

    def test_tide_infers_covariates_and_restores_them(self):
        x = dict(
            past_values=self.x[..., :1], past_time_features=self.x[..., 1:], future_time_features=self.x[:, :3, 1:]
        )
        y = self.y[:, :3, :1]
        pipe = tfts.Estimator("forecasting", model="tide", epochs=1)
        pipe.fit(
            {k: v[:8] for k, v in x.items()},
            y[:8],
            validation_data=({k: v[8:] for k, v in x.items()}, y[8:]),
            verbose=0,
        )
        self.assertEqual(pipe.model_.config.feature_dim, 2)
        prediction = pipe.predict(x)
        with tempfile.TemporaryDirectory() as directory:
            pipe.save(directory)
            restored = tfts.Estimator.load(directory)
            np.testing.assert_allclose(restored.predict(x), prediction, atol=1e-5)

    def test_masked_objective_and_exact_target_shape(self):
        target = tf.constant([[[1.0, 3.0], [100.0, 100.0]]])
        batch = TimeSeriesBatch(past_values=tf.zeros([1, 2, 3]), labels=target, sample_weight=tf.constant([[1.0, 0.0]]))
        loss = SequenceMeanSquaredError()(batch, ForecastOutput(predictions=tf.zeros_like(target)))
        self.assertAlmostEqual(float(loss), 5.0)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            SequenceMeanSquaredError()(batch, ForecastOutput(predictions=tf.zeros([1, 2, 1])))

    def test_multitarget_native_forecasts_use_all_target_channels(self):
        for name in ("dlinear", "patch_tst"):
            with self.subTest(model=name):
                config = tfts.AutoConfig.for_model(name)
                if name == "patch_tst":
                    config.patch_size = 4
                    config.patch_stride = 2
                model = tfts.AutoModel.from_config(config, output_chunk_length=2, target_dim=3)
                self.assertEqual(model(self.x).predictions.shape, (12, 2, 3))

    def test_causal_sequence_regression_and_epoch_weighting(self):
        model = tfts.AutoModel.from_config(
            tfts.AutoConfig.for_model("rnn"), task="sequence_regression", output_chunk_length=8, target_dim=2
        )
        expected = model(self.x).predictions.numpy()
        changed = self.x.copy()
        changed[:, 4:] += 100
        np.testing.assert_allclose(model(changed).predictions[:, :4], expected[:, :4], atol=1e-6)
        # Unequal batch sizes and masks must produce the global masked MSE.
        weights = np.ones((12, 8), dtype="float32")
        weights[:5, 2:] = 0
        model.compile(optimizer="adam")
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y, weights)).batch(5)
        result = model.evaluate(dataset, verbose=0, return_dict=True)["loss"]
        expected_loss = np.sum((expected - self.y) ** 2 * weights[..., None]) / (weights.sum() * 2)
        self.assertAlmostEqual(result, float(expected_loss), places=4)

    def test_default_learns_target_absent_from_inputs(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(128, 8, 3)).astype("float32")
        y = 3 * x[..., 1:2] - 2 * x[..., 2:3] + 10
        pipe = tfts.Estimator("sequence_regression", input_projection=True, epochs=12, batch_size=16)
        pipe.fit(x[:96], y[:96], validation_data=(x[96:], y[96:]), verbose=0)
        mse = np.mean((pipe.predict(x[96:]) - y[96:]) ** 2)
        baseline = np.mean((y[96:] - y[:96].mean()) ** 2)
        self.assertLess(mse, baseline * 0.3)
