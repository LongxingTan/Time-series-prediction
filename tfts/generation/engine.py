"""Tensor-based iterative forecast driver shared by training and generation."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import tensorflow as tf

from .feedback import FeedbackPolicy
from .processors import resolve_forecast_processors
from .samplers import SamplingResult, StepOutput


@dataclass
class RolloutOutput:
    values: tf.Tensor
    predictions: tf.Tensor
    distribution_params: Optional[Mapping[str, tf.Tensor]]
    state: Any


def _time_first(value):
    return tf.transpose(value, tf.concat([[1, 0], tf.range(2, tf.rank(value))], axis=0))


def _invariant(value):
    if isinstance(value, tf.TensorArray):
        return tf.TensorShape(None)
    shape = value.shape.as_list()
    return tf.TensorShape([None] * (len(shape) - 1) + shape[-1:]) if shape else value.shape


class GenerationEngine:
    """Decode one or more timesteps per call with a single feedback policy.

    Model state and inputs must be tensor nests. Model objects and immutable
    conditioning belong in closures/context. The first step runs outside the
    graph loop so Keras layers can create their variables once.
    """

    def __init__(self, sampler, feedback=None, processors=None):
        self.sampler = sampler
        self.feedback = feedback or (lambda current, result, **kwargs: result.value)
        self.processors = resolve_forecast_processors(processors)

    def run(
        self,
        step_fn,
        initial_input,
        initial_state,
        horizon,
        *,
        teacher=None,
        teacher_observed_mask=None,
        feedback_policy=None,
        context=None,
        seed_for_step=None,
    ):
        horizon = tf.cast(horizon, tf.int32)
        tf.debugging.assert_positive(horizon, message="horizon must be >= 1")
        policy = feedback_policy or FeedbackPolicy()
        state = () if initial_state is None else initial_state

        def predict(current, state, offset, history):
            output = step_fn(current, state, offset)
            if not isinstance(output, StepOutput):
                raise TypeError("step_fn must return StepOutput")
            seed = seed_for_step(offset) if seed_for_step is not None else None
            selected = self.sampler.sample(output, step=offset, seed=seed)
            selected = self.processors(history, selected, step=offset, step_output=output, context=context)
            tf.debugging.assert_equal(
                tf.shape(selected.value),
                tf.shape(output.prediction),
                message="sampler and processors must preserve the forecast block shape",
            )
            tf.debugging.assert_positive(tf.shape(selected.value)[1], message="decoder emitted an empty block")
            return output, selected

        output, selected = predict(initial_input, state, tf.constant(0), None)
        state = () if output.state is None else output.state

        def new_array(value):
            shape = tf.TensorShape([None, None]).concatenate(value.shape[2:])
            return tf.TensorArray(value.dtype, size=0, dynamic_size=True, element_shape=shape).write(
                0, _time_first(value)
            )

        values = new_array(selected.value)
        predictions = new_array(output.prediction)
        parameters = tf.nest.map_structure(new_array, output.parameters or {})

        def body(offset, index, current, state, selected_value, values, predictions, parameters):
            previous_offset = offset - tf.shape(selected_value)[1]
            seed = seed_for_step(previous_offset) if seed_for_step is not None else None
            fed = policy.select(
                selected_value,
                teacher,
                teacher_observed_mask,
                offset=previous_offset,
                seed=seed,
            )
            current = self.feedback(current, SamplingResult(fed), step=previous_offset, context=context)
            history = _time_first(values.concat()) if self.processors else None
            output, selected = predict(current, state, offset, history)
            values = values.write(index, _time_first(selected.value))
            predictions = predictions.write(index, _time_first(output.prediction))
            parameters = tf.nest.map_structure(
                lambda array, value: array.write(index, _time_first(value)),
                parameters,
                output.parameters or {},
            )
            return (
                offset + tf.shape(selected.value)[1],
                index + 1,
                current,
                () if output.state is None else output.state,
                selected.value,
                values,
                predictions,
                parameters,
            )

        loop = (
            tf.shape(selected.value)[1],
            tf.constant(1),
            initial_input,
            state,
            selected.value,
            values,
            predictions,
            parameters,
        )
        result = tf.while_loop(
            lambda offset, *_: offset < horizon,
            body,
            loop,
            shape_invariants=tf.nest.map_structure(_invariant, loop),
        )
        _, _, _, state, _, values, predictions, parameters = result
        collect = lambda array: _time_first(array.concat())[:, :horizon, ...]
        return RolloutOutput(
            collect(values),
            collect(predictions),
            tf.nest.map_structure(collect, parameters) or None,
            state,
        )
