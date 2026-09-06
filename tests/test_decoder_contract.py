"""Behavioral contracts shared by every incremental forecasting architecture."""

from dataclasses import replace
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting
from tfts.contracts import TimeSeriesBatch
from tfts.generation import FeedbackPolicy, GenerationEngine, MeanSampler, StepOutput, decode


def model_for(name, **task):
    configs = {
        "seq2seq": dict(rnn_hidden_size=8, dense_hidden_size=8),
        "wavenet": dict(filters=8, dense_hidden_size=8, dilation_rates=[1, 4], kernel_sizes=[2, 2]),
        "transformer": dict(
            hidden_size=8, num_layers=1, num_decoder_layers=2, num_attention_heads=2, ffn_intermediate_size=16
        ),
        "deep_ar": dict(hidden_size=8, rnn_layers=1, dropout=0.0),
    }
    config = AutoConfig.for_model(name)
    config.update(configs[name])
    return AutoModelForForecasting.from_config(config, prediction_length=3, **task)


class FeedbackTest(unittest.TestCase):
    def run_decode(self, probability, seed=12):
        return (
            GenerationEngine(MeanSampler())
            .run(
                lambda previous, state, step: StepOutput(previous + 1, state=state),
                tf.ones([8, 1, 2]),
                (),
                3,
                teacher=tf.ones([8, 3, 2]) * 10,
                feedback_policy=FeedbackPolicy(probability),
                seed_for_step=lambda step: tf.stack([seed, step]),
            )
            .values
        )

    def test_teacher_is_used_only_for_next_prediction(self):
        np.testing.assert_allclose(self.run_decode(1.0)[0, :, 0], [2, 11, 11])
        np.testing.assert_allclose(self.run_decode(0.0)[0, :, 0], [2, 3, 4])

    def test_scheduled_sampling_is_seeded_and_graph_safe(self):
        eager = self.run_decode(tf.constant(0.5))
        graph = tf.function(self.run_decode)(tf.constant(0.5))
        np.testing.assert_array_equal(eager, graph)
        np.testing.assert_array_equal(eager[..., 0], eager[..., 1])
        self.assertGreater(len(np.unique(eager[:, 1, 0])), 1)

    def test_missing_teacher_elements_use_prediction(self):
        value = FeedbackPolicy(1.0).select(
            tf.constant([[[2.0, 3.0]]]),
            tf.constant([[[10.0, float("nan")]]]),
            tf.constant([[[True, False]]]),
        )
        np.testing.assert_array_equal(value, [[[10.0, 3.0]]])

    def test_block_decoder_returns_exact_dynamic_horizon(self):
        def run(horizon):
            def step(previous, state, offset):
                last = previous[:, -1:, :]
                return StepOutput(tf.concat([last + 1, last + 2], axis=1), state=state)

            return GenerationEngine(MeanSampler()).run(step, tf.zeros([2, 1, 1]), (), horizon).values

        np.testing.assert_array_equal(tf.function(run)(tf.constant(5))[0, :, 0], [1, 2, 3, 4, 5])


class DecoderContractTest(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(11)
        self.batch = TimeSeriesBatch(
            past_values=tf.random.normal([2, 6, 1]),
            future_values=tf.random.normal([2, 3, 1]),
            future_time_features=tf.random.normal([2, 3, 1]),
        )

    def test_all_decoders_are_causal_and_support_feedback(self):
        for name in ("seq2seq", "wavenet", "transformer", "deep_ar"):
            with self.subTest(model=name):
                model = model_for(name).backbone
                batch = self.batch if name != "deep_ar" else replace(self.batch, future_time_features=None)
                changed = replace(batch, future_values=batch.future_values + 100)
                first = decode(model, batch, 3, teacher_probability=1.0).predictions
                second = decode(model, changed, 3, teacher_probability=1.0).predictions
                np.testing.assert_allclose(first[:, 0], second[:, 0], atol=1e-6)
                self.assertGreater(float(tf.reduce_max(tf.abs(first[:, 1:] - second[:, 1:]))), 1e-6)
                free = decode(model, batch, 3).predictions
                free_changed = decode(model, changed, 3).predictions
                np.testing.assert_allclose(free, free_changed, atol=1e-6)

    def test_transformer_parallel_cached_and_uncached_agree(self):
        model = model_for("transformer").backbone
        cached = decode(model, self.batch, 3, teacher_probability=1.0).predictions
        full = model.decode_teacher_forced(self.batch).predictions
        np.testing.assert_allclose(cached, full, atol=2e-5)
        model.decoder.use_cache = False
        uncached = decode(model, self.batch, 3, teacher_probability=1.0).predictions
        np.testing.assert_allclose(cached, uncached, atol=2e-5)
        changed = replace(self.batch, future_values=self.batch.future_values + tf.constant([[[0.0], [0.0], [100.0]]]))
        np.testing.assert_allclose(full, model.decode_teacher_forced(changed).predictions, atol=2e-5)

    def test_graph_gradients_and_multivariate_generation(self):
        for name in ("seq2seq", "wavenet", "transformer"):
            with self.subTest(model=name):
                model = model_for(name, target_dim=2)
                batch = TimeSeriesBatch(past_values=tf.ones([2, 6, 2]), future_values=tf.ones([2, 3, 2]))
                model.forward(batch, training=True)

                @tf.function
                def train(values, targets):
                    with tf.GradientTape() as tape:
                        output = model.forward(
                            TimeSeriesBatch(past_values=values, future_values=targets),
                            training=True,
                            teacher_probability=tf.constant(0.5),
                        )
                        loss = tf.reduce_mean(tf.square(output.predictions - targets))
                    gradients = tape.gradient(loss, model.trainable_variables)
                    return loss, [g for g in gradients if g is not None]

                loss, gradients = train(batch.past_values, batch.future_values)
                self.assertTrue(bool(tf.math.is_finite(loss)))
                self.assertTrue(gradients)
                for gradient in gradients:
                    self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
                generated = model.generate(batch, prediction_length=5)
                self.assertEqual(generated.predictions.shape, (2, 5, 2))

    def test_keras_training_routes_targets_and_schedule(self):
        model = model_for("seq2seq", teacher_decay_steps=10, teacher_final_probability=0.2)
        model.compile(optimizer="adam", loss="mse")
        model.train_on_batch(self.batch.past_values, self.batch.future_values)
        model.optimizer.iterations.assign(10)
        self.assertAlmostEqual(float(model._teacher_probability()), 0.2, places=5)

    def test_save_load_preserves_forecast(self):
        for name in ("seq2seq", "wavenet", "transformer"):
            with self.subTest(model=name):
                model = model_for(name)
                expected = model(self.batch.past_values)
                with tempfile.TemporaryDirectory() as directory:
                    path = directory + "/model.keras"
                    model.save(path)
                    restored = tf.keras.models.load_model(path)
                    np.testing.assert_allclose(expected, restored(self.batch.past_values), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
