"""Behavioral tests for the consolidated dataframe path."""

import json
import unittest

import numpy as np
import pandas as pd

from tfts.data import FramePipeline, Schema, TemporalSplit
from tfts.data.regularize import validate_frame
from tfts.data.scaling import ChannelScaler
from tfts.data.schema import FeatureDType, FeatureRole, FeatureSpec


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"time": np.arange(20), "value": np.arange(20, dtype=float)})
        self.schema = Schema("time", ("value",))

    def test_train_only_statistics_and_reload(self):
        self.frame.loc[15:, "value"] = 10000
        pipeline = FramePipeline(self.schema, 4, 2)
        prepared, train, valid = pipeline.fit(self.frame, validation=0.25)
        self.assertEqual(pipeline.target_scaler.offset[0], 7)
        self.assertTrue(all(w.decoder_indices[-1] < 15 for w in train))
        self.assertTrue(all(w.decoder_indices[0] >= 15 for w in valid))
        restored = FramePipeline.from_dict(json.loads(json.dumps(pipeline.to_dict())))
        np.testing.assert_allclose(restored.transform(self.frame).frame.value, prepared.frame.value)
        self.assertEqual(train.starts.dtype, np.int64)
        self.assertFalse(train.starts.flags.writeable)

    def test_gaps_and_nan_labels_preserve_masks(self):
        frame = self.frame.drop(index=5).copy()
        frame.loc[8, "value"] = np.nan
        pipeline = FramePipeline(self.schema, 4, 2, normalize=None, freq=1)
        prepared, windows, _ = pipeline.fit(frame, validation=0)
        self.assertEqual(len(prepared.frame), 20)
        x, y = next(iter(pipeline.dataset(prepared, windows, batch_size=len(windows))))
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.isfinite(x["past_values"]).all())
        self.assertFalse(bool(x["future_observed_mask"][0, 1, 0]))
        self.assertEqual(float(y[0, 1, 0]), 0)
        self.assertFalse(bool(x["past_observed_mask"][5, 0, 0]))

    def test_dataset_preserves_static_window_shapes_for_graph_training(self):
        import tensorflow as tf

        from tfts import AutoConfig, AutoModelForForecasting

        pipeline = FramePipeline(self.schema, 4, 2)
        prepared, windows, _ = pipeline.fit(self.frame, validation=0)
        dataset = pipeline.dataset(prepared, windows, batch_size=2)
        inputs, labels = dataset.element_spec
        self.assertEqual(inputs["past_values"].shape.as_list(), [None, 4, 1])
        self.assertEqual(inputs["past_observed_mask"].shape.as_list(), [None, 4, 1])
        self.assertEqual(labels.shape.as_list(), [None, 2, 1])
        for name in ("nbeats", "deep_ar"):
            with self.subTest(model=name):
                model = AutoModelForForecasting.from_config(AutoConfig.for_model(name), output_chunk_length=2)
                model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=False)
                history = model.fit(dataset.take(1), epochs=1, verbose=0)
                self.assertTrue(np.isfinite(history.history["loss"]).all())
                tf.keras.backend.clear_session()

    def test_covariate_masks_distinguish_missing_and_unseen(self):
        schema = Schema(
            "time",
            ("value",),
            (
                FeatureSpec("category", FeatureRole.OBSERVED_PAST, FeatureDType.CATEGORICAL),
                FeatureSpec("real", FeatureRole.OBSERVED_PAST),
            ),
        )
        frame = self.frame.assign(category="a", real=0.0)
        frame.loc[3, ["category", "real"]] = [None, np.nan]
        frame.loc[16:, "category"] = "unseen"
        pipeline = FramePipeline(schema, 4, 2, normalize=None)
        prepared, train, valid = pipeline.fit(frame, validation=0.2)
        self.assertEqual(pipeline.vocabularies["category"], ["a"])
        x, _ = next(iter(pipeline.dataset(prepared, train, batch_size=1)))
        self.assertEqual(int(x["past_categorical_features"][0, 3, 0]), 0)
        self.assertFalse(bool(x["past_categorical_features_mask"][0, 3, 0]))
        self.assertFalse(bool(x["past_time_features_mask"][0, 3, 0]))
        x, _ = next(iter(pipeline.dataset(prepared, valid, batch_size=len(valid))))
        self.assertEqual(int(x["past_categorical_features"][-1, -1, 0]), 0)
        self.assertTrue(bool(x["past_categorical_features_mask"][-1, -1, 0]))

    def test_groups_never_mix_and_input_is_unchanged(self):
        frame = pd.concat([self.frame.assign(group="b"), self.frame.assign(group="a")], ignore_index=True)
        original = frame.copy(deep=True)
        pipeline = FramePipeline(Schema("time", ("value",), group_cols=("group",)), 4, 2)
        prepared, train, valid = pipeline.fit(frame)
        for window in list(train) + list(valid):
            rows = list(window.encoder_indices) + list(window.decoder_indices)
            self.assertEqual(prepared.frame.iloc[rows].group.nunique(), 1)
        pd.testing.assert_frame_equal(frame, original)

    def test_off_grid_duplicates_and_infinity_raise(self):
        for mutate, pattern in [
            (lambda f: f.assign(time=[0, 0.5] + list(range(2, 20))), "off-grid"),
            (lambda f: f.assign(time=[0, 0] + list(range(2, 20))), "Duplicate"),
            (lambda f: f.assign(value=np.inf), "infinite"),
        ]:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                FramePipeline(self.schema, 4, 2, freq=1).fit(mutate(self.frame), validation=0)

    def test_validation_reports_multiple_problems(self):
        frame = self.frame.assign(time=0, value=np.inf)
        with self.assertRaises(ValueError) as caught:
            validate_frame(frame, self.schema)
        self.assertIn("Duplicate", str(caught.exception))
        self.assertIn("infinite", str(caught.exception))

    def test_gap_excludes_training_labels_without_losing_validation_history(self):
        split = TemporalSplit(0.25, gap=2)
        rows, cutoffs = split.rows(self.frame)
        self.assertEqual(rows[-1], 12)
        self.assertEqual(cutoffs[()], 15)

    def test_learned_features_fit_only_training_rows_and_serialize(self):
        from tfts.data.features import ClipTransform, LagTransform

        clip = ClipTransform("value", lower=0, upper=1)
        pipeline = FramePipeline(self.schema, 4, 2, normalize=None, transforms=(clip, LagTransform("value", (2,))))
        frame = self.frame.copy()
        frame.loc[15:, "value"] = 10000
        prepared, train, valid = pipeline.fit(frame, validation=0.25)
        self.assertEqual(clip.limits_, {})  # fitted clone, caller configuration unchanged
        self.assertEqual(pipeline.features_.transforms[0].limits_["value"], [0.0, 14.0])
        self.assertEqual(prepared.frame.value_clipped.max(), 14.0)
        restored = FramePipeline.from_dict(json.loads(json.dumps(pipeline.to_dict())))
        pd.testing.assert_frame_equal(restored.transform(frame).frame, prepared.frame)
        expected, _, _ = pipeline.prediction(frame)
        actual, _, _ = restored.prediction(frame)
        pd.testing.assert_frame_equal(actual.frame, expected.frame)
        self.assertEqual(float(actual.frame.value_lag_2.iloc[2]), 14.0)

    def test_monthly_calendar_grid_does_not_require_equal_day_deltas(self):
        from tfts.datasets import load_air_passengers

        frame = load_air_passengers()
        pipeline = FramePipeline(Schema("date", ("value",)), 12, 3, freq="MS")
        prepared, train, valid = pipeline.fit(frame, validation=0.2)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(valid), 0)
        x, y = next(iter(pipeline.dataset(prepared, train)))
        self.assertEqual(tuple(y.shape[1:]), (3, 1))


class ScalerTest(unittest.TestCase):
    def test_masked_fit_roundtrip_and_constants(self):
        values = np.array([[1.0, 5.0], [2.0, 5.0], [10000.0, 5.0]])
        for method in [None, "standard", "minmax", "robust", "log1p"]:
            with self.subTest(method=method):
                scaler = ChannelScaler(method).fit(values, [1, 1, 0])
                restored = ChannelScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
                np.testing.assert_allclose(restored.inverse_transform(restored.transform(values)), values, rtol=1e-5)
                self.assertTrue((restored.scale > 0).all())
        self.assertEqual(ChannelScaler().fit(values, [1, 1, 0]).offset[0], 1.5)

    def test_unobserved_channels_and_invalid_weights_raise(self):
        for values, weight in [
            (np.full((3, 1), np.nan), None),
            (np.ones((3, 1)), [0, 0, 0]),
            (np.ones((3, 1)), [-1, 1, 1]),
        ]:
            with self.assertRaises(ValueError):
                ChannelScaler().fit(values, weight)
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            ChannelScaler("log1p").fit(np.array([[-1.0]]))
