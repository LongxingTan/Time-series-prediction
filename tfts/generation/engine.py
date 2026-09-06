"""Tensor-based iterative forecast driver shared by training and generation."""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import tensorflow as tf

from .feedback import TeacherForcingPolicy
from .processors import resolve_forecast_processors
from .samplers import StepOutput
from .state import GenStep
from .steps import StepDecoder
from .stopping import MaxHorizon, StoppingCriteriaList


@dataclass
class RolloutOutput:
    values: tf.Tensor
    predictions: tf.Tensor
    distribution_params: Optional[Mapping[str, tf.Tensor]]
    state: Any
    quantile_values: Optional[tf.Tensor] = None


def _time_first(value):
    return tf.transpose(value, tf.concat([[1, 0], tf.range(2, tf.rank(value))], axis=0))


def _invariant(value):
    if isinstance(value, tf.TensorArray):
        return tf.TensorShape(None)
    shape = value.shape.as_list()
    return tf.TensorShape([None] * (len(shape) - 1) + shape[-1:]) if shape else value.shape


class TimeAxisEngine:
    """Decode one or more timesteps per call with a single feedback policy.

    Model state and inputs must be tensor nests. Model objects and immutable
    conditioning belong in closures/context. The first step runs outside the
    graph loop so Keras layers can create their variables once.
    """

    def __init__(self, sampler, processors=None, stopping_criteria=None):
        self.sampler = sampler
        self.processors = resolve_forecast_processors(processors)
        self.stopping_criteria = stopping_criteria or ()

    def run(
        self,
        step_fn,
        initial_input,
        initial_state,
        horizon,
        *,
        teacher=None,
        teacher_observed_mask=None,
        teacher_forcing_policy=None,
        past_values=None,
        seed_for_step=None,
    ):
        horizon = tf.cast(horizon, tf.int32)
        tf.debugging.assert_positive(horizon, message="horizon must be >= 1")
        policy = teacher_forcing_policy or TeacherForcingPolicy()
        stopping = StoppingCriteriaList([MaxHorizon(horizon), *self.stopping_criteria])
        state = () if initial_state is None else initial_state
        decoder = step_fn if isinstance(step_fn, StepDecoder) else StepDecoder(step_fn)

        def predict(current, state, offset, history):
            output = decoder.step(current, state, offset)
            if not isinstance(output, StepOutput):
                raise TypeError("step_fn must return StepOutput")
            seed = seed_for_step(offset) if seed_for_step is not None else None
            empty = output.prediction[:, :0, ...]
            step = GenStep(
                offset,
                empty if past_values is None else past_values,
                empty if history is None else history,
                output.prediction,
                output.distribution,
                output.parameters,
                quantile_values=output.quantile_values,
            )
            step = self.processors(step, stage="parameters")
            selected = self.sampler(step, seed=seed)
            step = self.processors(replace(step, value=selected), stage="values")
            selected = step.value
            output = replace(
                output, parameters=step.parameters, distribution=step.distribution, quantile_values=step.quantile_values
            )
            tf.debugging.assert_equal(
                tf.shape(selected),
                tf.shape(output.prediction),
                message="sampler and processors must preserve the forecast block shape",
            )
            tf.debugging.assert_positive(tf.shape(selected)[1], message="decoder emitted an empty block")
            return output, selected, stopping(step)

        output, selected, stopped = predict(initial_input, state, tf.constant(0), None)
        state = () if output.state is None else output.state

        def new_array(value):
            shape = tf.TensorShape([None, None]).concatenate(value.shape[2:])
            return tf.TensorArray(value.dtype, size=0, dynamic_size=True, element_shape=shape).write(
                0, _time_first(value)
            )

        values = new_array(selected)
        predictions = new_array(output.prediction)

        def auxiliary(output):
            return {
                "parameters": output.parameters or {},
                "quantiles": {} if output.quantile_values is None else output.quantile_values,
            }

        parameters = tf.nest.map_structure(new_array, auxiliary(output))

        def body(offset, index, current, state, selected_value, values, predictions, parameters, stopped):
            previous_offset = offset - tf.shape(selected_value)[1]
            seed = seed_for_step(previous_offset) if seed_for_step is not None else None
            fed = policy.select(
                selected_value,
                teacher,
                teacher_observed_mask,
                offset=previous_offset,
                seed=seed,
            )
            current = decoder.next_input(current, fed, offset=previous_offset)
            history = _time_first(values.concat())
            output, selected, stopped = predict(current, state, offset, history)
            values = values.write(index, _time_first(selected))
            predictions = predictions.write(index, _time_first(output.prediction))
            parameters = tf.nest.map_structure(
                lambda array, value: array.write(index, _time_first(value)),
                parameters,
                auxiliary(output),
            )
            return (
                offset + tf.shape(selected)[1],
                index + 1,
                current,
                () if output.state is None else output.state,
                selected,
                values,
                predictions,
                parameters,
                stopped,
            )

        loop = (
            tf.shape(selected)[1],
            tf.constant(1),
            initial_input,
            state,
            selected,
            values,
            predictions,
            parameters,
            stopped,
        )
        result = tf.while_loop(
            lambda offset, *args: (offset < horizon) & ~args[-1],
            body,
            loop,
            shape_invariants=tf.nest.map_structure(_invariant, loop),
        )
        _, _, _, state, _, values, predictions, parameters, _ = result
        collect = lambda array: _time_first(array.concat())[:, :horizon, ...]
        collected = tf.nest.map_structure(collect, parameters)
        return RolloutOutput(
            collect(values),
            collect(predictions),
            collected["parameters"] or None,
            state,
            None if isinstance(collected["quantiles"], dict) else collected["quantiles"],
        )
