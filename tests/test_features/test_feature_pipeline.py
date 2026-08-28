import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from tfts import AutoConfig, AutoModelForForecasting
from tfts.contracts import InputLayout, ModelInputSpec
from tfts.data import SequenceMaterializer, TabularMaterializer, WindowIndexer, WindowSpec
from tfts.features import (
    CategoricalEncoderTransform,
    DatetimeTransform,
    FeatureDType,
    FeatureManifest,
    FeaturePipeline,
    FeatureRole,
    FeatureSelection,
    FeatureSpec,
    LagTransform,
    RollingTransform,
    TimeSeriesSchema,
    resolve_feature_plan,
)
from tfts.models import resolve_model_features


class FeaturePipelineTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=24, freq="D"),
                "sales": np.arange(24, dtype=np.float32),
                "promotion": np.arange(24) % 2,
                "store_type": np.ones(24, dtype=np.int32),
                "temperature": np.linspace(10, 20, 24),
            }
        )
        self.schema = TimeSeriesSchema(
            time_col="time",
            target_cols=("sales",),
            features=(
                FeatureSpec(
                    "promotion",
                    FeatureRole.KNOWN_FUTURE,
                    FeatureDType.CATEGORICAL,
                    tags={"commercial"},
                ),
                FeatureSpec("temperature", FeatureRole.OBSERVED_PAST, tags={"weather"}),
                FeatureSpec("store_type", FeatureRole.STATIC, FeatureDType.CATEGORICAL),
            ),
        )

    def test_pipeline_tracks_semantics_lineage_and_history(self):
        pipeline = FeaturePipeline(
            [
                LagTransform("sales", [1, 3]),
                RollingTransform("sales", [3], functions=["mean"]),
                DatetimeTransform(["dayofweek"]),
            ]
        )
        prepared = pipeline.fit_transform(self.frame, self.schema)

        self.assertEqual(prepared.manifest.required_history, 3)
        self.assertEqual(prepared.schema.get("sales_lag_3").role, FeatureRole.OBSERVED_PAST)
        self.assertEqual(prepared.schema.get("time_dayofweek").role, FeatureRole.KNOWN_FUTURE)
        self.assertTrue(np.isnan(prepared.frame.loc[0, "sales_roll_3_mean"]))
        self.assertEqual(prepared.frame.loc[3, "sales_roll_3_mean"], 1.0)

        transformed = pipeline.transform(self.frame.copy())
        self.assertEqual(transformed.manifest.fingerprint, prepared.manifest.fingerprint)
        pd.testing.assert_frame_equal(transformed.frame, prepared.frame)

    def test_manifest_round_trip_has_a_stable_fingerprint(self):
        prepared = FeaturePipeline([LagTransform("sales", [1])]).fit_transform(self.frame, self.schema)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/manifest.json"
            prepared.manifest.save(path)
            restored = FeatureManifest.load(path)
        self.assertEqual(restored, prepared.manifest)
        self.assertEqual(restored.fingerprint, prepared.manifest.fingerprint)
        json.dumps(restored.to_dict())

    def test_required_history_follows_transform_lineage(self):
        prepared = FeaturePipeline([LagTransform("sales", [2]), LagTransform("sales_lag_2", [3])]).fit_transform(
            self.frame, self.schema
        )
        self.assertEqual(prepared.manifest.required_history, 5)

    def test_static_features_are_validated(self):
        invalid = self.frame.copy()
        invalid.loc[5, "store_type"] = 2
        with self.assertRaisesRegex(ValueError, "static features vary"):
            FeaturePipeline().fit_transform(invalid, self.schema)

    def test_categorical_encoder_reuses_training_codes_and_handles_unknowns(self):
        frame = self.frame.copy()
        frame["store_type"] = ["urban", "rural"] * 12
        schema = TimeSeriesSchema(
            "time",
            ("sales",),
            (FeatureSpec("store_type", FeatureRole.KNOWN_FUTURE, FeatureDType.CATEGORICAL),),
        )
        pipeline = FeaturePipeline([CategoricalEncoderTransform("store_type")])
        prepared = pipeline.fit_transform(frame, schema)
        self.assertEqual(prepared.frame["store_type_encoded"].tolist()[:2], [1, 2])

        inference = frame.copy()
        inference.loc[0, "store_type"] = "unseen"
        transformed = pipeline.transform(inference)
        self.assertEqual(transformed.frame.loc[0, "store_type_encoded"], 0)
        self.assertEqual(
            transformed.manifest.fitted_state["0:CategoricalEncoderTransform"]["store_type"],
            ("urban", "rural"),
        )

    def test_categorical_encoder_reserves_configured_unknown_code(self):
        frame = self.frame.copy()
        frame["store_type"] = ["urban", "rural"] * 12
        schema = TimeSeriesSchema(
            "time",
            ("sales",),
            (FeatureSpec("store_type", FeatureRole.KNOWN_FUTURE, FeatureDType.CATEGORICAL),),
        )
        pipeline = FeaturePipeline([CategoricalEncoderTransform("store_type", unknown_value=1)])
        prepared = pipeline.fit_transform(frame, schema)
        self.assertEqual(prepared.frame["store_type_encoded"].tolist()[:2], [0, 2])
        self.assertEqual(prepared.schema.get("store_type_encoded").parameters["cardinality"], 3)

        inference = frame.copy()
        inference.loc[0, "store_type"] = "unseen"
        self.assertEqual(pipeline.transform(inference).frame.loc[0, "store_type_encoded"], 1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CategoricalEncoderTransform("store_type", unknown_value=-1)

    def test_manifest_contract_values_are_immutable(self):
        prepared = FeaturePipeline([LagTransform("sales", [1])]).fit_transform(self.frame, self.schema)
        fingerprint = prepared.manifest.fingerprint
        lag_spec = prepared.schema.get("sales_lag_1")

        with self.assertRaises(TypeError):
            lag_spec.parameters["lag"] = 99
        with self.assertRaises(TypeError):
            prepared.manifest.fitted_state["new"] = {}
        self.assertEqual(prepared.manifest.fingerprint, fingerprint)

    def test_selection_is_resolved_against_model_inputs(self):
        selection = FeatureSelection(include_tags={"commercial", "weather"}, exclude_names={"temperature"})
        input_spec = ModelInputSpec(
            layout=InputLayout.TABULAR,
            accepted_roles={FeatureRole.KNOWN_FUTURE.value},
            supports_categorical=True,
            supports_static=False,
        )
        plan = resolve_feature_plan(self.schema, selection, input_spec=input_spec)
        self.assertEqual(plan.feature_names, ("promotion",))
        self.assertEqual(plan.excluded["temperature"], "excluded by selection")

        with self.assertRaisesRegex(ValueError, "does not accept role"):
            resolve_feature_plan(
                self.schema,
                FeatureSelection(include_names={"temperature"}),
                input_spec=input_spec,
                unsupported="drop",
            )

    def test_registered_model_resolves_features_without_model_name_branches(self):
        tft_plan = resolve_model_features("tft", self.schema)
        self.assertEqual(
            tft_plan.feature_names,
            ("promotion", "temperature", "store_type"),
        )
        rnn_plan = resolve_model_features("rnn", self.schema, unsupported="drop")
        self.assertEqual(rnn_plan.feature_names, ("temperature",))

    def test_deep_ar_rejects_unsupported_static_real_and_multivariate_targets(self):
        schema = TimeSeriesSchema(
            "time",
            ("sales",),
            (FeatureSpec("store_size", FeatureRole.STATIC),),
        )
        plan = resolve_model_features("deep_ar", schema, unsupported="drop")
        self.assertEqual(plan.feature_names, ())
        self.assertIn("dtype", plan.excluded["store_size"])

        multivariate = TimeSeriesSchema("time", ("sales", "returns"))
        with self.assertRaisesRegex(ValueError, "multivariate targets"):
            resolve_model_features("deep_ar", multivariate, unsupported="drop")


class MaterializerTest(unittest.TestCase):
    def _prepared(self):
        observed = np.arange(18, dtype=np.float32)
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=18, freq="D"),
                "target": observed,
                "known_real": np.linspace(0, 1, 18),
                "known_cat": np.arange(18) % 3,
                "observed_real": observed * 2,
                "static_cat": np.full(18, 4),
            }
        )
        schema = TimeSeriesSchema(
            "time",
            ("target",),
            (
                FeatureSpec("known_real", FeatureRole.KNOWN_FUTURE),
                FeatureSpec("known_cat", FeatureRole.KNOWN_FUTURE, FeatureDType.CATEGORICAL),
                FeatureSpec("observed_real", FeatureRole.OBSERVED_PAST),
                FeatureSpec("static_cat", FeatureRole.STATIC, FeatureDType.CATEGORICAL),
            ),
        )
        return FeaturePipeline([LagTransform("target", [2])]).fit_transform(frame, schema)

    def test_same_windows_materialize_to_2d_and_3d(self):
        prepared = self._prepared()
        windows = WindowIndexer().build(prepared, WindowSpec(context_length=4, prediction_length=3))
        tabular = TabularMaterializer().materialize(prepared, windows)
        sequence = SequenceMaterializer().materialize(prepared, windows)

        self.assertEqual(tabular.X.ndim, 2)
        self.assertEqual(tabular.y.shape, (len(windows) * 3, 1))
        self.assertIn("forecast_horizon", tabular.feature_names)
        self.assertIn("known_cat@forecast", tabular.categorical_feature_names)
        self.assertEqual(sequence.past_values.shape, (len(windows), 4, 1))
        self.assertEqual(sequence.future_values.shape, (len(windows), 3, 1))
        self.assertEqual(sequence.future_time_features.shape, (len(windows), 3, 1))
        self.assertEqual(sequence.future_categorical_features.shape, (len(windows), 3, 1))
        np.testing.assert_allclose(tabular.y.reshape(len(windows), 3), sequence.future_values.numpy().squeeze(-1))

    def test_future_targets_cannot_change_model_inputs(self):
        prepared = self._prepared()
        windows = WindowIndexer().build(prepared, WindowSpec(4, 2))
        baseline = TabularMaterializer().materialize(prepared, windows)
        changed_frame = prepared.frame.copy()
        last_decoder = list(windows.windows[0].decoder_indices)
        changed_frame.loc[last_decoder, "target"] = 100000
        changed = type(prepared)(changed_frame, prepared.manifest)
        changed_batch = TabularMaterializer().materialize(changed, windows)

        np.testing.assert_allclose(baseline.X[0], changed_batch.X[0])
        self.assertFalse(np.allclose(baseline.y[0], changed_batch.y[0]))

    def test_prediction_requires_future_rows_only_for_known_features(self):
        prepared = self._prepared()
        future = prepared.frame.iloc[-2:].copy()
        future["time"] = pd.date_range(prepared.frame["time"].iloc[-1] + pd.Timedelta(days=1), periods=2, freq="D")
        future["target"] = np.nan
        future["known_real"] = [2.0, 3.0]
        future["known_cat"] = [1, 2]
        inference = type(prepared)(pd.concat([prepared.frame, future], ignore_index=True), prepared.manifest)
        windows = WindowIndexer().build(inference, WindowSpec(4, 2, mode="predict"))
        batch = SequenceMaterializer().materialize(inference, windows)

        self.assertIsNone(batch.labels)
        np.testing.assert_allclose(batch.future_time_features.numpy()[0, :, 0], [2.0, 3.0])
        np.testing.assert_array_equal(batch.future_categorical_features.numpy()[0, :, 0], [1, 2])

    def test_tf_dataset_retains_teacher_forcing_values_for_deep_ar(self):
        prepared = self._prepared()
        windows = WindowIndexer().build(prepared, WindowSpec(4, 2))
        batch = SequenceMaterializer().materialize(
            prepared,
            windows,
            selection=FeatureSelection(
                exclude_names={"known_real", "known_cat", "observed_real", "static_cat", "target_lag_2"}
            ),
        )
        inputs, labels = next(iter(SequenceMaterializer.as_tf_dataset(batch, batch_size=2)))
        model = AutoModelForForecasting.from_config(AutoConfig.for_model("deep_ar"), prediction_length=2)

        output = model(inputs, training=True)

        self.assertEqual(output.shape, (2, 2, 1))
        np.testing.assert_allclose(inputs["future_values"].numpy(), labels.numpy())

        inputs_without_future, _ = next(
            iter(SequenceMaterializer.as_tf_dataset(batch, batch_size=2, include_future_values=False))
        )
        self.assertNotIn("future_values", inputs_without_future)


if __name__ == "__main__":
    unittest.main()
