"""Regression tests for shared execution boundaries."""

import unittest

import tensorflow as tf

from tfts.contracts import ForecastOutput, ImputationOutput, TimeSeriesBatch
from tfts.data.preprocessing import inverse_output
from tfts.data.scaling import ChannelScaler
from tfts.generation.chunk import Chunk, State
from tfts.generation.loop import run
from tfts.generation.processors import Mean


class TestDesignContracts(unittest.TestCase):
    def test_inverse_scaling_replaces_outputs_without_mutating_them(self):
        scaler = ChannelScaler("minmax").fit([[0.0], [2.0]])
        for output, field in (
            (ForecastOutput(predictions=tf.ones([1, 2, 1])), "predictions"),
            (ImputationOutput(imputed_values=tf.ones([1, 2, 1])), "imputed_values"),
        ):
            with self.subTest(field=field):
                result = inverse_output(output, scaler)
                self.assertEqual(float(tf.reduce_mean(getattr(result, field))), 2.0)
                self.assertEqual(float(tf.reduce_mean(getattr(output, field))), 1.0)

    def test_decoder_does_not_require_a_model_attribute(self):
        class ConstantDecoder:
            output_chunk_length = 1
            output_distribution = None

            def start(self, batch, *, horizon, training=False):
                return State(batch)

            def step(self, state, *, offset, training=False):
                values = state.value.past_values[:, :1]
                return Chunk(offset=offset, past=values, generated=values[:, :0], prediction=values)

            def feed(self, state, feedback, *, offset):
                return state

        batch = TimeSeriesBatch(past_values=tf.ones([2, 3, 1]))
        result = run(ConstantDecoder(), batch, horizon=2, processors=[Mean()])
        self.assertEqual(tuple(result.predictions.shape), (2, 2, 1))
        self.assertEqual(float(tf.reduce_mean(result.predictions)), 1.0)
