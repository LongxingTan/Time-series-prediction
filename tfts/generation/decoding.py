"""Small decoder protocol and its shared execution entry point."""

from dataclasses import dataclass
from typing import Any, Protocol

import tensorflow as tf

from tfts.contracts import OutputPort, TimeSeriesBatch

from .engine import TimeAxisEngine
from .feedback import TeacherForcingPolicy
from .samplers import StepOutput, resolve_value_sampler
from .steps import StepDecoder


@dataclass
class DecodeSession:
    context: Any
    state: Any
    previous: tf.Tensor


class IncrementalDecoder(Protocol):
    def initialize_decode(self, batch: TimeSeriesBatch, *, horizon, training=False) -> DecodeSession:
        """Prepare fixed conditioning, tensor state, and a seed input."""
        ...

    def decode_step(self, previous, state, context, *, offset, training=False) -> StepOutput:
        """Consume the previous accepted block and predict the next block."""
        ...

    def next_input(self, value, context, *, offset) -> tf.Tensor:
        """Convert an accepted block into the next decoder input."""
        ...


def decode(
    model,
    batch,
    horizon,
    *,
    training=False,
    teacher_probability=0.0,
    sampler="point",
    processors=None,
    stopping_criteria=None,
    seed=None,
    detach_feedback=True,
):
    """Run a decoder; targets enter only through the explicit feedback policy."""
    session = model.initialize_decode(batch, horizon=horizon, training=training)
    active_sampler = resolve_value_sampler(sampler, probabilistic=model.capabilities.has_port(OutputPort.DISTRIBUTION))

    def step_fn(previous, state, offset):
        return model.decode_step(previous, state, session.context, offset=offset, training=training)

    def step_seed(offset):
        return None if seed is None else tf.stack([tf.cast(seed, tf.int32), tf.cast(offset, tf.int32)])

    return TimeAxisEngine(active_sampler, processors=processors, stopping_criteria=stopping_criteria).run(
        StepDecoder(step_fn, lambda current, value, *, offset: model.next_input(value, session.context, offset=offset)),
        session.previous,
        session.state,
        horizon,
        teacher=batch.future_values,
        teacher_observed_mask=batch.future_observed_mask,
        teacher_forcing_policy=TeacherForcingPolicy(teacher_probability, detach_feedback),
        past_values=batch.past_values,
        seed_for_step=step_seed,
    )
