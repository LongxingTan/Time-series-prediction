import unittest

import numpy as np
import tensorflow as tf

from tfts.contracts import ForecastOutput, TimeSeriesBatch
from tfts.distributions import NormalOutput
from tfts.generation import ForecastGenerationConfig, RecursiveRollout, generate, prepare_generation_batch


class _RecordingRecursiveModel:
    def __init__(self):
        self.task_config = type("TaskConfig", (), {"prediction_length": 2})()
        self.calls = []

    def forward(self, batch, training=False):
        self.calls.append(batch)
        prediction = batch.past_values[:, -1:, :] + 1.0
        return ForecastOutput(predictions=prediction)


class _DistributionRecursiveModel(_RecordingRecursiveModel):
    def __init__(self):
        super().__init__()
        self.output_distribution = NormalOutput()

    def forward(self, batch, training=False):
        loc = tf.repeat(batch.past_values[:, -1:, :] + 1.0, 2, axis=1)
        scale = tf.ones_like(loc)
        return ForecastOutput(predictions=loc, distribution_params={"loc": loc, "scale": scale})


class GenerationBatchTest(unittest.TestCase):
    def test_variable_length_histories_become_a_canonical_batch(self):
        batch = prepare_generation_batch(
            [np.array([1.0, 2.0]), np.array([3.0])],
            sequence_length=3,
            padding_side="left",
        )

        np.testing.assert_array_equal(batch.past_values.numpy()[:, :, 0], [[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]])
        np.testing.assert_array_equal(batch.padding_mask.numpy(), [[False, True, True], [False, False, True]])
        batch.validate_for("forecasting")

    def test_generation_values_are_floating_point(self):
        batch = prepare_generation_batch([[1, 2], [3]], sequence_length=2)

        self.assertTrue(batch.past_values.dtype.is_floating)

    def test_recursive_feedback_keeps_temporal_fields_aligned(self):
        batch = TimeSeriesBatch(
            past_values=tf.constant([[[1.0], [2.0], [3.0]]]),
            future_values=tf.constant([[[99.0], [99.0]]]),
            past_time_features=tf.constant([[[10.0], [11.0], [12.0]]]),
            future_time_features=tf.constant([[[20.0], [21.0]]]),
            past_categorical_features=tf.constant([[[1], [2], [3]]]),
            future_categorical_features=tf.constant([[[4], [5]]]),
            past_observed_mask=tf.ones([1, 3, 1], tf.bool),
            future_observed_mask=tf.ones([1, 2, 1], tf.bool),
            padding_mask=tf.constant([[False, True, True]]),
            labels=tf.constant([[7.0]]),
        )
        model = _RecordingRecursiveModel()

        output = RecursiveRollout().run(model, batch, ForecastGenerationConfig(prediction_length=2))

        np.testing.assert_array_equal(output.predictions.numpy()[0, :, 0], [4.0, 5.0])
        second_batch = model.calls[1]
        np.testing.assert_array_equal(second_batch.past_values.numpy()[0, :, 0], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(second_batch.past_time_features.numpy()[0, :, 0], [11.0, 12.0, 20.0])
        np.testing.assert_array_equal(second_batch.past_categorical_features.numpy()[0, :, 0], [2, 3, 4])
        np.testing.assert_array_equal(second_batch.past_observed_mask.numpy()[0, :, 0], [True, True, False])
        np.testing.assert_array_equal(second_batch.padding_mask.numpy()[0], [True, True, True])
        self.assertIsNone(second_batch.future_values)
        self.assertIsNone(second_batch.future_observed_mask)
        self.assertIsNone(second_batch.labels)

    def test_recursive_sampling_keeps_one_timestep_per_step(self):
        model = _DistributionRecursiveModel()
        batch = TimeSeriesBatch(past_values=tf.constant([[[1.0], [2.0]]]))

        output = RecursiveRollout().run(
            model,
            batch,
            ForecastGenerationConfig(
                prediction_length=3,
                sampler="sample",
                num_samples=4,
                aggregation="mean",
                return_samples=True,
                seed=7,
            ),
        )

        self.assertEqual(output.predictions.shape, (1, 3, 1))
        self.assertEqual(output.samples.shape, (1, 4, 3, 1))

    def test_recursive_feedback_maps_known_future_features_by_name(self):
        batch = TimeSeriesBatch(
            past_values=tf.constant([[[1.0], [2.0]]]),
            past_time_features=tf.constant([[[10.0, 100.0], [11.0, 101.0]]]),
            future_time_features=tf.constant([[[200.0], [201.0]]]),
            metadata={
                "feature_names": {
                    "past_real": ("observed", "known"),
                    "future_real": ("known",),
                }
            },
        )
        model = _RecordingRecursiveModel()

        RecursiveRollout().run(model, batch, ForecastGenerationConfig(prediction_length=2))

        np.testing.assert_allclose(model.calls[1].past_time_features.numpy()[0, -1], [11.0, 200.0])

    def test_generation_rejects_padding_for_an_unsupported_backbone(self):
        model = _RecordingRecursiveModel()
        model.backbone = object()
        model.capabilities = type("Capabilities", (), {"forecast_modes": ()})()
        batch = prepare_generation_batch([[1.0, 2.0], [3.0]], sequence_length=2, padding_side="left")

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "does not support padded histories"):
            generate(model, batch, strategy="recursive", prediction_length=1)


if __name__ == "__main__":
    unittest.main()
