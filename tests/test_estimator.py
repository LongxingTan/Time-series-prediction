"""Behavioral contracts for the shared estimator and local model extensions."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import tensorflow as tf

import tfts
from tfts.contracts import BackboneCapabilities, ForecastOutput, OutputPort, TimeSeriesBatch
from tfts.data.frame import FramePreprocessor
from tfts.features import FeatureRole, FeatureSpec, TimeSeriesSchema
from tfts.models.base import BaseModel, CommonConfig
from tfts.models.registry import register_model
from tfts.tasks.objectives import MeanSquaredError, QuantileObjective


class EstimatorTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        random = np.random.default_rng(7)
        self.x = random.normal(size=(12, 8, 3)).astype(np.float32)
        self.y = (2 * self.x[:, :2, 1:2] - self.x[:, :2, 2:3]).astype(np.float32)
        self.frame = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=32), "value": np.arange(32, dtype=float)}
        )
        self.schema = TimeSeriesSchema("date", ("value",))

    def test_constructor_is_inert_and_fitted_parameters_are_separate(self):
        config = tfts.AutoConfig.for_model("rnn")
        original = copy.deepcopy(config.to_dict())
        with mock.patch("tfts.estimator.tf.keras.utils.set_random_seed") as seed:
            estimator = tfts.Estimator(config=config, epochs=1)
            self.assertFalse(hasattr(estimator, "model_"))
            seed.assert_not_called()
        with self.assertRaises(tfts.NotFittedError):
            estimator.predict(self.x)
        self.assertIs(estimator.fit(self.x, self.y, verbose=0), estimator)
        self.assertIs(estimator.config, config)
        self.assertEqual(config.to_dict(), original)
        self.assertEqual(estimator.horizon, 24)
        self.assertEqual(estimator.horizon_, 2)
        self.assertIsNotNone(estimator.history_)
        self.assertEqual(estimator.predict(self.x).shape, self.y.shape)
        estimator.set_params(epochs=2)
        with self.assertRaises(tfts.NotFittedError):
            estimator.predict(self.x)

    def test_refitting_replaces_preprocessing(self):
        estimator = tfts.Estimator(epochs=1).fit(self.x, self.y, verbose=0)
        first_model = estimator.model_
        estimator.fit(self.x + 50, self.y + 10, verbose=0)
        self.assertIsNot(first_model, estimator.model_)
        np.testing.assert_allclose(
            estimator.preprocessor_.target_scaler.offset, (self.y + 10).mean(axis=(0, 1)), atol=1e-6
        )

    def test_early_stopping_override_with_and_without_validation(self):
        for validation in (None, (self.x, self.y)):
            with self.subTest(validation=validation is not None):
                estimator = tfts.Forecaster(epochs=1)
                estimator.fit(self.x, self.y, validation_data=validation, early_stopping_patience=2, verbose=0)
                self.assertEqual(len(estimator.history_.history["loss"]), 1)
                self.assertEqual("val_loss" in estimator.history_.history, validation is not None)

    def test_dataset_classification_requires_declared_class_count(self):
        # The first batch deliberately omits the last class.
        labels = np.repeat(np.arange(3), 4)
        dataset = tf.data.Dataset.from_tensor_slices((self.x, labels)).batch(4)
        with self.assertRaisesRegex(ValueError, "explicit num_labels"):
            tfts.Estimator("classification", epochs=1).fit(dataset, verbose=0)
        for options in ({"num_labels": 3}, {"task_config": tfts.ClassificationTaskConfig(num_labels=3)}):
            with self.subTest(options=options):
                estimator = tfts.Estimator("classification", epochs=1, **options).fit(dataset, verbose=0)
                self.assertEqual(estimator.predict(dataset).shape, (12, 3))
                self.assertTrue(np.isfinite(estimator.evaluate(dataset, verbose=0)["loss"]))

    def test_dataset_and_sequence_infer_anomaly_target_width_without_labels(self):
        class Batches(tf.keras.utils.Sequence):
            def __init__(self, values):
                super().__init__()
                self.values = values

            def __len__(self):
                return 3

            def __getitem__(self, index):
                return {"past_values": self.values[index * 4 : (index + 1) * 4]}

        for inputs in (
            tf.data.Dataset.from_tensor_slices({"past_values": self.x}).batch(4),
            Batches(self.x),
        ):
            with self.subTest(input_type=type(inputs).__name__):
                estimator = tfts.Estimator("anomaly_detection", epochs=1).fit(inputs, verbose=0)
                self.assertEqual(estimator.model_.task_config.target_dim, 3)
                output = estimator.predict_output(inputs)
                self.assertEqual(output.reconstruction.shape, self.x.shape)
                self.assertTrue(np.isfinite(estimator.evaluate(inputs, verbose=0)["loss"]))
                with tempfile.TemporaryDirectory() as directory:
                    estimator.save(directory)
                    restored = tfts.Estimator.load(directory)
                    np.testing.assert_allclose(restored.predict(inputs), estimator.predict(inputs), atol=1e-6)

    def test_grid_forecasting_dataset_infers_dimensions(self):
        values = self.x.reshape(12, 8, 1, 3, 1)
        targets = values[:, :2]
        dataset = tf.data.Dataset.from_tensor_slices((values, targets)).batch(4)
        estimator = tfts.Forecaster("dlinear", spatial_strategy="per_node", epochs=1)
        estimator.fit(dataset, verbose=0)
        self.assertEqual(estimator.horizon_, 2)
        self.assertEqual(estimator.predict(dataset).shape, targets.shape)
        self.assertTrue(np.isfinite(estimator.evaluate(dataset, verbose=0)["loss"]))

    def test_array_anomaly_evaluation_uses_the_same_default_targets_as_fit(self):
        estimator = tfts.Estimator("anomaly_detection", epochs=1).fit(self.x, verbose=0)
        implicit = estimator.evaluate(self.x, verbose=0)["loss"]
        explicit = estimator.evaluate(self.x, self.x, verbose=0)["loss"]
        self.assertAlmostEqual(implicit, explicit)

    def test_arrays_and_datasets_validate_the_same_target_contract(self):
        for labels in (np.ones(12) * 0.5, np.full(12, -1), np.ones((12, 1)), np.ones((12, 2))):
            for inputs, targets in (
                (self.x, labels),
                (tf.data.Dataset.from_tensor_slices((self.x, labels)).batch(4), None),
            ):
                with self.subTest(shape=labels.shape, dataset=targets is None):
                    with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
                        tfts.Estimator("classification", num_labels=3, epochs=1).fit(inputs, targets, verbose=0)

    def test_array_and_classification_roundtrip(self):
        for task, target in (("forecasting", self.y), ("classification", np.arange(12) % 3)):
            with self.subTest(task=task):
                estimator = tfts.Estimator(task, epochs=1, batch_size=5).fit(self.x, target, verbose=0)
                expected = estimator.predict(self.x)
                with tempfile.TemporaryDirectory() as directory:
                    estimator.save(directory)
                    restored = tfts.Estimator.load(directory)
                    np.testing.assert_allclose(restored.predict(self.x), expected, atol=1e-6)
                self.assertEqual(expected.shape[0], len(self.x))

    def test_dataframe_uses_training_rows_and_forecasts_after_last_history_time(self):
        estimator = tfts.Forecaster("dlinear", context=4, horizon=2, schema=self.schema, epochs=1, batch_size=7)
        estimator.fit(self.frame.sample(frac=1, random_state=7), verbose=0)
        np.testing.assert_allclose(estimator.preprocessor_.target_scaler.offset, [12.0])
        result = estimator.predict(self.frame.tail(4))
        self.assertEqual(list(result.columns), ["date", "target", "prediction"])
        self.assertEqual(result["date"].tolist(), list(pd.date_range("2024-02-02", periods=2)))
        with tempfile.TemporaryDirectory() as directory:
            estimator.save(directory)
            restored = tfts.Forecaster.load(directory)
            pd.testing.assert_frame_equal(restored.predict(self.frame.tail(4)), result)
            path = Path(directory) / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["format_version"] = 999
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "upgrade"):
                tfts.Forecaster.load(directory)

    def test_temporal_partition_uses_context_before_validation_cutoff(self):
        processor = FramePreprocessor(self.schema, 4, 3)
        _, train, valid = processor.fit(self.frame, validation=0.25)
        self.assertEqual(max(window.decoder_indices[-1] for window in train.windows), 23)
        self.assertEqual(valid.windows[0].encoder_indices[0], 20)
        self.assertEqual(valid.windows[0].decoder_indices[0], 24)
        self.assertIsInstance(train.windows[0].encoder_indices, range)

    def test_known_future_covariates_are_explicit_and_saved(self):
        frame = self.frame.assign(promotion=np.arange(32) % 2)
        schema = TimeSeriesSchema("date", ("value",), (FeatureSpec("promotion", FeatureRole.KNOWN_FUTURE),))
        estimator = tfts.Forecaster("tide", context=4, horizon=2, schema=schema, epochs=1, hidden_size=8, num_layers=1)
        estimator.fit(frame, verbose=0)
        with self.assertRaisesRegex(ValueError, "future"):
            estimator.predict(frame.tail(4))
        future = pd.DataFrame({"date": pd.date_range("2024-02-02", periods=2), "promotion": [1, 0]})
        expected = estimator.predict(frame.tail(4), future=future)
        with tempfile.TemporaryDirectory() as directory:
            estimator.save(directory)
            pd.testing.assert_frame_equal(
                tfts.Forecaster.load(directory).predict(frame.tail(4), future=future), expected
            )
        future.loc[0, "promotion"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing future"):
            estimator.predict(frame.tail(4), future=future)

    def test_bad_data_and_unknown_options_fail_before_training(self):
        with self.assertRaisesRegex(ValueError, "Unknown options"):
            tfts.Estimator(hidden_sise=4).fit(self.x, self.y)
        for invalid, message in (
            (pd.concat([self.frame, self.frame.iloc[:1]]), "Duplicate"),
            (self.frame.drop(5), "regular|gaps"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                tfts.Forecaster(schema=self.schema, context=4, horizon=2).fit(invalid)

    def test_short_prediction_history_reuses_fitted_frequency(self):
        for context in (1, 2):
            processor = FramePreprocessor(self.schema, context, 2)
            processor.fit(self.frame)
            _, windows, dates = processor.prediction(self.frame.tail(context))
            self.assertEqual(len(windows), 1)
            self.assertEqual(dates[0][1], pd.Timestamp("2024-02-02"))

    def test_individual_dlinear_infers_channel_count(self):
        estimator = tfts.Forecaster("dlinear", individual=True, epochs=1)
        estimator.fit(self.x[..., :2], self.y.repeat(2, axis=-1), verbose=0)
        self.assertEqual(estimator.model_.backbone.channels, 2)
        with tempfile.TemporaryDirectory() as directory:
            estimator.save(directory)
            np.testing.assert_allclose(
                tfts.Forecaster.load(directory).predict(self.x[..., :2]),
                estimator.predict(self.x[..., :2]),
                atol=1e-6,
            )

    def test_custom_objective_survives_roundtrip(self):
        estimator = tfts.Forecaster(objective="mae", epochs=1).fit(self.x, self.y, verbose=0)
        expected = estimator.evaluate(self.x, self.y, verbose=0)
        with tempfile.TemporaryDirectory() as directory:
            estimator.save(directory)
            restored = tfts.Forecaster.load(directory)
            self.assertEqual(type(restored.model_.objective), type(estimator.model_.objective))
            self.assertAlmostEqual(restored.evaluate(self.x, self.y, verbose=0)["loss"], expected["loss"], places=6)

    def test_registered_head_owns_its_default_objective(self):
        from tfts.registry import _REGISTRY, register_head
        from tfts.tasks.auto_task import PointForecastHead

        @register_head("estimator_test_absolute")
        class AbsoluteHead(PointForecastHead):
            default_objective = "mae"

        self.addCleanup(_REGISTRY["head"].pop, "estimator_test_absolute")
        estimator = tfts.Forecaster(head="estimator_test_absolute", epochs=1).fit(self.x, self.y, verbose=0)
        self.assertIsInstance(estimator.model_.head, AbsoluteHead)
        self.assertEqual(estimator.model_._objective_spec.name, "mae")
        with tempfile.TemporaryDirectory() as directory:
            estimator.save(directory)
            restored = tfts.Forecaster.load(directory)
            self.assertIsInstance(restored.model_.head, AbsoluteHead)
            np.testing.assert_allclose(restored.predict(self.x), estimator.predict(self.x), atol=1e-6)

    def test_lazy_windows_match_eager_materialization(self):
        from tfts.data.materializers import SequenceMaterializer

        processor = FramePreprocessor(self.schema, 4, 2)
        prepared, windows, _ = processor.fit(self.frame)
        eager = SequenceMaterializer().materialize(prepared, windows, plan=processor.plan)
        batches = list(processor.dataset(prepared, windows, batch_size=7))
        np.testing.assert_array_equal(tf.concat([x["past_values"] for x, _ in batches], 0), eager.past_values)
        np.testing.assert_array_equal(tf.concat([y for _, y in batches], 0), eager.future_values)

    def test_researcher_adds_one_registered_keras_model(self):
        class TinyConfig(CommonConfig):
            model_type = "estimator_test_tiny"

        @register_model(
            TinyConfig.model_type,
            config=TinyConfig,
            capabilities=BackboneCapabilities(
                output_ports=frozenset({OutputPort.NATIVE_FORECAST, OutputPort.TEMPORAL_SEQUENCE})
            ),
        )
        class TinyModel(BaseModel):
            def __init__(self, predict_sequence_length=1, config=None):
                super().__init__(predict_sequence_length, config)
                self.encoder = tf.keras.layers.Dense(4, activation="tanh")
                self.projection = tf.keras.layers.Dense(predict_sequence_length * config.target_dim)

            def call(self, inputs, training=None):
                sequence = self.encoder(inputs)
                forecast = tf.reshape(
                    self.projection(tf.reduce_mean(sequence, axis=1)),
                    [-1, self.predict_sequence_length, self.config.target_dim],
                )
                return {"sequence_output": sequence, "native_forecast": forecast}

        from tfts.models import registry

        self.addCleanup(registry._ENTRIES.pop, TinyConfig.model_type)
        for task, target in (("forecasting", self.y), ("sequence_regression", self.x[..., :2])):
            with self.subTest(task=task):
                estimator = tfts.Estimator(task, TinyConfig.model_type, epochs=1)
                estimator.fit(self.x, target, verbose=0)
                expected = estimator.predict(self.x)
                self.assertEqual(expected.shape, target.shape)
                with tempfile.TemporaryDirectory() as directory:
                    estimator.save(directory)
                    np.testing.assert_allclose(tfts.Estimator.load(directory).predict(self.x), expected, atol=1e-6)


class ReductionTest(unittest.TestCase):
    def test_custom_objective_only_needs_reduction_method(self):
        from tfts.tasks.objectives import LossResult

        class SquaredError:
            def reduce(self, batch, output):
                errors = tf.square(output.predictions - batch.future_values)
                return LossResult(tf.reduce_sum(errors), tf.cast(tf.size(errors), tf.float32))

        model = tfts.AutoModel.from_config(tfts.AutoConfig.for_model("dlinear"), output_chunk_length=2)
        model.configure_objective(SquaredError())
        batch = TimeSeriesBatch(tf.ones([2, 4, 1]), future_values=tf.zeros([2, 2, 1]))
        self.assertAlmostEqual(float(model(batch).loss), 1.0)

    def test_mixed_precision_graph_training(self):
        policy = tf.keras.mixed_precision.global_policy()
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, policy)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = tfts.AutoModel.from_config(tfts.AutoConfig.for_model("rnn"), output_chunk_length=2, head="quantile")
        model.compile(optimizer="adam", run_eagerly=False)
        dataset = tf.data.Dataset.from_tensor_slices((tf.ones([4, 3, 1]), tf.ones([4, 2, 1]))).batch(2)
        history = model.fit(dataset, epochs=1, verbose=0)
        self.assertTrue(np.isfinite(history.history["loss"]).all())
        self.assertTrue(all(np.isfinite(variable.numpy()).all() for variable in model.trainable_variables))

    def test_learned_heads_emit_full_horizon_and_global_weighted_loss(self):
        random = np.random.default_rng(7)
        x = random.normal(size=(7, 3, 1)).astype(np.float32)
        y = random.normal(size=(7, 5, 2)).astype(np.float32)
        weights = np.ones_like(y)
        weights[:3, :4] = 0
        for head in ("quantile", "distribution"):
            with self.subTest(head=head):
                model = tfts.AutoModel.from_config(
                    tfts.AutoConfig.for_model("rnn"),
                    output_chunk_length=5,
                    target_dim=2,
                    head=head,
                    quantiles=(0.1, 0.5, 0.9),
                )
                output = model(x)
                self.assertEqual(output.predictions.shape, y.shape)
                if head == "quantile":
                    self.assertTrue(np.all(np.diff(output.quantile_values, axis=-1) >= 0))
                batch = TimeSeriesBatch(x, future_values=y, sample_weight=weights)
                expected = model.objective.reduce(batch, output).value
                model.compile(optimizer="adam", run_eagerly=False)
                dataset = tf.data.Dataset.from_tensor_slices((x, y, weights)).batch(3)
                self.assertAlmostEqual(model.evaluate(dataset, verbose=0), float(expected), places=5)
                with tempfile.TemporaryDirectory() as directory:
                    path = str(Path(directory) / "model.keras")
                    model.save(path)
                    restored = tf.keras.models.load_model(path, compile=False)
                    np.testing.assert_allclose(restored(x).predictions, output.predictions, atol=1e-6)

    def test_quantile_mask_preserves_scale_and_weight(self):
        target = tf.ones([3, 2, 2])
        values = tf.zeros([3, 2, 2, 3])
        output = ForecastOutput(predictions=values[..., 1], quantile_values=values)
        objective = QuantileObjective()
        batch = TimeSeriesBatch(tf.zeros([3, 4, 2]), future_values=target)
        expected = objective.reduce(batch, output)
        masked = objective.reduce(
            TimeSeriesBatch(batch.past_values, future_values=target, future_observed_mask=tf.ones_like(target)), output
        )
        self.assertAlmostEqual(float(expected.value), 0.5)
        self.assertEqual(float(expected.weight), 12)
        self.assertAlmostEqual(float(masked.value), float(expected.value))

    def test_masked_nan_targets_have_finite_gradients(self):
        prediction = tf.Variable([[[2.0], [4.0]]])
        target = tf.constant([[[1.0], [float("nan")]]])
        batch = TimeSeriesBatch(
            tf.zeros([1, 2, 1]), future_values=target, future_observed_mask=tf.constant([[[1.0], [0.0]]])
        )
        with tf.GradientTape() as tape:
            loss = MeanSquaredError().reduce(batch, ForecastOutput(predictions=prediction)).value
        np.testing.assert_array_equal(tape.gradient(loss, prediction), [[[2.0], [0.0]]])
