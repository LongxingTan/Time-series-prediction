import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting, GenerationConfig, TimeSeriesBatch
from tfts.generation import (
    Clip,
    DirectDecoder,
    Feedback,
    Mean,
    NativeDecoder,
    RecursiveDecoder,
    Sample,
    StepProcessorList,
    TeacherForcing,
    Trajectory,
    run,
)
from tfts.registry import ComponentSpec, register_processor


class GenerationInvariantTest(unittest.TestCase):
    def setUp(self):
        self.batch = TimeSeriesBatch(tf.ones([2, 8, 1]))

    def model(self, name="dlinear", horizon=2, **kwargs):
        return AutoModelForForecasting.from_config(AutoConfig.for_model(name), output_chunk_length=horizon, **kwargs)

    def test_horizon_one_agrees_for_applicable_decoders(self):
        for name in ("seq2seq", "wavenet", "transformer", "deep_ar"):
            with self.subTest(name=name):
                model = self.model(name, horizon=1)
                fields = {"past_values": self.batch.past_values}
                if name == "deep_ar":
                    fields["static_categorical_features"] = tf.zeros([2, 1], tf.int32)
                batch = TimeSeriesBatch(**fields)
                direct = model.generate(batch, horizon=1, mode="direct", trajectory=())
                native = model.generate(batch, horizon=1, mode="native", trajectory=())
                np.testing.assert_allclose(direct.predictions, native.predictions, atol=2e-5)

    def test_direct_steps_once(self):
        model = self.model()

        class CountingDecoder(DirectDecoder):
            calls = 0

            def step(self, *args, **kwargs):
                self.calls += 1
                return super().step(*args, **kwargs)

        decoder = CountingDecoder(model)
        result = run(decoder, self.batch, horizon=2, processors=[Mean()])
        self.assertEqual(decoder.calls, 1)
        self.assertEqual(result.predictions.shape, (2, 2, 1))

    def test_missing_selector_and_invalid_order_fail_at_build(self):
        with self.assertRaisesRegex(ValueError, "exactly one Selector"):
            StepProcessorList([Clip(maximum=1.0)])
        with self.assertRaisesRegex(ValueError, "requires.*value"):
            StepProcessorList([Clip(maximum=1.0), Mean()])

    def test_head_is_never_bypassed(self):
        model = self.model("bert", head="point")
        with self.assertRaisesRegex(ValueError, "learned task head"):
            NativeDecoder(model)

    def test_custom_stochastic_selector_preserves_sample_breadth(self):
        class IdentitySample(Sample):
            def __init__(self):
                pass

            def __call__(self, chunk):
                return chunk.replace(value=chunk.prediction)

        model = self.model()
        output = model.generate(
            self.batch,
            processors=[IdentitySample()],
            num_samples=4,
            trajectory=(),
        )
        self.assertEqual(output.samples.shape, (2, 4, 2, 1))

    def test_feedback_changes_only_later_recursive_chunks(self):
        model = self.model(horizon=2)

        class AddOneFeedback(Feedback):
            def __call__(self, chunk):
                return chunk.replace(feedback=chunk.value + 1.0)

        baseline = run(RecursiveDecoder(model), self.batch, horizon=3, processors=[Mean()], seed=3)
        changed = run(
            RecursiveDecoder(model),
            self.batch,
            horizon=3,
            processors=[Mean(), AddOneFeedback()],
            seed=3,
        )
        np.testing.assert_allclose(changed.predictions[:, :1], baseline.predictions[:, :1])
        self.assertFalse(np.allclose(changed.predictions[:, 1:], baseline.predictions[:, 1:]))

        direct = run(DirectDecoder(model), self.batch, horizon=2, processors=[Mean()])
        direct_feedback = run(DirectDecoder(model), self.batch, horizon=2, processors=[Mean(), AddOneFeedback()])
        np.testing.assert_allclose(direct.predictions, direct_feedback.predictions)

    def test_teacher_forced_native_matches_parallel_path(self):
        model = self.model("transformer", horizon=3)
        batch = TimeSeriesBatch(
            past_values=tf.random.stateless_normal([2, 8, 1], [1, 2]),
            future_values=tf.random.stateless_normal([2, 3, 1], [3, 4]),
            future_time_features=tf.ones([2, 3, 1]),
        )
        parallel = model.backbone.decode_teacher_forced(batch, training=False)
        iterative = run(
            NativeDecoder(model),
            batch,
            horizon=3,
            processors=[Mean(), TeacherForcing(batch.future_values, probability=1.0)],
            seed=7,
        )
        np.testing.assert_allclose(iterative.predictions, parallel.predictions, atol=2e-5)

    def test_default_inverse_scale_returns_input_units(self):
        class DoubleScaler:
            @staticmethod
            def inverse_transform(value):
                return value * 2.0

        model = self.model()
        normalized = model.generate(self.batch, trajectory=())
        scaled_batch = TimeSeriesBatch(self.batch.past_values, metadata={"scaler": DoubleScaler()})
        restored = model.generate(scaled_batch)
        np.testing.assert_allclose(restored.predictions, normalized.predictions * 2.0)

    def test_trajectory_interface_covers_point_quantile_and_distribution(self):
        point = Trajectory(tf.ones([2, 3, 1]))
        self.assertEqual(point.mean.shape, (2, 3, 1))
        self.assertEqual(point.quantile(0.9).shape, (2, 3, 1))

        quantile = Trajectory(
            tf.ones([2, 3, 1]),
            quantile_values=tf.ones([2, 3, 1, 3]),
            quantiles=(0.1, 0.5, 0.9),
        )
        self.assertEqual(quantile.quantile(0.9).shape, (2, 3, 1))

        model = self.model("deep_ar", horizon=3)
        batch = TimeSeriesBatch(tf.ones([2, 8, 1]), static_categorical_features=tf.zeros([2, 1], tf.int32))
        distribution = model.generate(batch, trajectory=())
        self.assertEqual(distribution.mean.shape, (2, 3, 1))
        self.assertEqual(distribution.quantile(0.9).shape, (2, 3, 1))

    def test_registry_config_round_trip(self):
        @register_processor("invariant_identity_selector")
        class IdentitySelector(Mean):
            pass

        config = GenerationConfig(
            horizon=2,
            processors=(ComponentSpec("invariant_identity_selector"),),
            trajectory=(),
        )
        restored = GenerationConfig.from_dict(config.to_dict())
        model = self.model()
        first = model.generate(self.batch, config, trajectory=())
        second = model.generate(self.batch, restored, trajectory=())
        np.testing.assert_allclose(first.predictions, second.predictions)


if __name__ == "__main__":
    unittest.main()
