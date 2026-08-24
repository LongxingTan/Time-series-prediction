import unittest

import numpy as np
import tensorflow as tf

from tfts.generation import (
    DifferenceClipProcessor,
    FullFeatureFeedback,
    RemoveInvalidValuesProcessor,
    RollingWindowGenerationMixin,
    SamplingResult,
    SamplingStrategy,
    ValueClipProcessor,
)


class IncrementModel(RollingWindowGenerationMixin):
    predict_sequence_length = 4

    def __call__(self, inputs):
        target = inputs[:, -1:, 1:2]
        return tf.repeat(target + 1.0, self.predict_sequence_length, axis=1)


class FullFeatureIncrementModel(RollingWindowGenerationMixin):
    predict_sequence_length = 1

    def __call__(self, inputs):
        return inputs[:, -1:, :] + 1.0


class DoubleSampling(SamplingStrategy):
    def sample(self, step_output, *, step, seed=None, teacher=None, state=None):
        return SamplingResult(2.0 * step_output.prediction)


class GenerationStrategyTest(unittest.TestCase):
    def test_target_feedback_rolls_first_direct_step(self):
        model = IncrementModel()
        inputs = tf.constant([[[10.0, -1.0], [11.0, 0.0]]])

        output = model.generate(inputs, horizon=3, target_indices=(1,))

        np.testing.assert_allclose(output.predictions.numpy(), [[[1.0], [2.0], [3.0]]])

    def test_custom_sampling_controls_output_and_feedback(self):
        model = IncrementModel()
        inputs = tf.constant([[[10.0, -1.0], [11.0, 0.0]]])

        output = model.generate(inputs, horizon=3, strategy=DoubleSampling(), target_indices=(1,))

        np.testing.assert_allclose(output.predictions.numpy(), [[[2.0], [6.0], [14.0]]])

    def test_full_feature_feedback_matches_mitsui_style_rollout(self):
        model = FullFeatureIncrementModel()
        inputs = tf.constant([[[0.0, 10.0], [1.0, 11.0]]])

        output = model.generate(inputs, horizon=3, feedback=FullFeatureFeedback())

        np.testing.assert_allclose(output.predictions.numpy(), [[[2.0, 12.0], [3.0, 13.0], [4.0, 14.0]]])

    def test_processors_constrain_output_and_autoregressive_feedback(self):
        model = IncrementModel()
        inputs = tf.constant([[[10.0, -1.0], [11.0, 0.0]]])

        output = model.generate(
            inputs,
            horizon=4,
            target_indices=(1,),
            processors=[ValueClipProcessor(maximum=2.5)],
        )

        np.testing.assert_allclose(output.predictions.numpy(), [[[1.0], [2.0], [2.5], [2.5]]])

    def test_difference_processor_limits_changes_after_first_step(self):
        model = IncrementModel()
        inputs = tf.constant([[[10.0, -1.0], [11.0, 0.0]]])

        output = model.generate(
            inputs,
            horizon=3,
            strategy=DoubleSampling(),
            target_indices=(1,),
            processors=DifferenceClipProcessor(1.0),
        )

        np.testing.assert_allclose(output.predictions.numpy(), [[[2.0], [3.0], [4.0]]])

    def test_invalid_value_processor_repairs_non_finite_predictions(self):
        class InvalidModel(RollingWindowGenerationMixin):
            predict_sequence_length = 1

            def __call__(self, inputs):
                return tf.fill([tf.shape(inputs)[0], 1, 1], np.nan)

        output = InvalidModel().generate(
            tf.zeros([1, 2, 1]),
            horizon=2,
            processors=RemoveInvalidValuesProcessor(fallback=-1.0),
        )
        np.testing.assert_allclose(output.predictions.numpy(), [[[-1.0], [-1.0]]])


if __name__ == "__main__":
    unittest.main()
