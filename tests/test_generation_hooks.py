"""Cross-strategy extension and compatibility regressions."""

from dataclasses import replace
import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting
from tfts.contracts import TimeSeriesBatch
from tfts.generation import ForecastProcessor, GenStep, ValueClipProcessor
from tfts.training.scheduled_sampling import scheduled_sampling_decode


class ShiftParameters(ForecastProcessor):
    stage = "parameters"

    def __call__(self, step):
        return replace(
            step, parameters={"loc": tf.ones_like(step.prediction) * 7, "scale": tf.zeros_like(step.prediction)}
        )


class GenerationHooksTest(unittest.TestCase):
    def setUp(self):
        config = AutoConfig.for_model("deep_ar")
        config.update({"hidden_size": 8, "rnn_layers": 1, "dropout": 0.0})
        self.model = AutoModelForForecasting.from_config(config, prediction_length=3)
        self.values = tf.ones([2, 5, 1])

    def test_parameter_processors_precede_sampling_everywhere(self):
        for strategy in ("direct", "recursive", "autoregressive"):
            result = self.model.generate(
                self.values, strategy=strategy, processors=ShiftParameters(), num_samples=3, return_samples=True, seed=9
            )
            np.testing.assert_array_equal(result.samples, tf.ones([2, 3, 3, 1]) * 7)
            np.testing.assert_array_equal(result.distribution_params["loc"], result.samples)

    def test_hooks_receive_history_and_support_stopping_in_graph(self):
        def sampler(step, *, seed=None):
            self.assertIsInstance(step, GenStep)
            self.assertFalse(hasattr(step, "context"))
            return step.past_values[:, -1:, ...] + tf.cast(step.offset, tf.float32)

        for strategy in ("direct", "recursive", "autoregressive"):

            @tf.function
            def run(values):
                return self.model.generate(
                    values, strategy=strategy, sampler=sampler, stopping_criteria=[lambda step: step.offset >= 1]
                ).predictions

            np.testing.assert_array_equal(run(self.values), [[[1.0], [2.0]], [[1.0], [2.0]]])

    def test_value_constraints_leave_sampling_parameters_explicit(self):
        result = self.model.generate(
            self.values, processors=[ShiftParameters(), ValueClipProcessor(maximum=2)], num_samples=2
        )
        np.testing.assert_array_equal(result.predictions, tf.ones([2, 3, 1]) * 2)
        np.testing.assert_array_equal(result.distribution_params["loc"], tf.ones([2, 2, 3, 1]) * 7)

    def test_scheduled_sampling_compatibility_is_seeded_and_differentiable(self):
        targets = tf.ones([2, 3, 1]) * 2
        for stochastic in (False, True):

            def run(x):
                return scheduled_sampling_decode(
                    self.model, x, None, targets, tf.constant(0.5), stochastic=stochastic, seed=7
                )

            with tf.GradientTape() as tape:
                eager = run(self.values)
                loss = tf.reduce_sum(eager["loc"])
            self.assertTrue(any(g is not None for g in tape.gradient(loss, self.model.trainable_variables)))
            graph = tf.function(run)(self.values)
            np.testing.assert_allclose(eager["loc"], graph["loc"], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "shared output head"):
            scheduled_sampling_decode(self.model, self.values, None, targets, 0.0, distribution_output=object())

    def test_old_transformer_checkpoint_config_is_rejected(self):
        model = AutoModelForForecasting.from_config(AutoConfig.for_model("transformer"), prediction_length=2)
        config = model.get_config()
        config["backbone_config"].pop("decoder_format_version")
        with self.assertRaisesRegex(ValueError, "Legacy decoder weights require migration"):
            type(model).from_config(config)

    def test_schedule_before_compile_uses_initial_probability(self):
        model = AutoModelForForecasting.from_config(
            AutoConfig.for_model("seq2seq"), prediction_length=2, teacher_decay_steps=10
        )
        self.assertEqual(float(model._teacher_probability()), model.task_config.teacher_probability)
        model.forward(TimeSeriesBatch(past_values=self.values, future_values=tf.ones([2, 2, 1])), training=True)
