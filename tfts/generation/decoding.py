"""Small decoder protocol and its shared execution entry point."""

from dataclasses import dataclass
from typing import Any, Protocol

import tensorflow as tf

from tfts.contracts import TimeSeriesBatch

from .engine import GenerationEngine
from .feedback import FeedbackPolicy
from .samplers import StepOutput, resolve_value_sampler


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


def decode(
    model,
    batch,
    horizon,
    *,
    training=False,
    teacher_probability=0.0,
    sampler="mean",
    processors=None,
    seed=None,
    detach_feedback=True,
):
    """Run a decoder; targets enter only through the explicit feedback policy."""
    session = model.initialize_decode(batch, horizon=horizon, training=training)
    active_sampler = resolve_value_sampler(
        sampler, probabilistic=getattr(model, "output_distribution", None) is not None
    )

    def step_fn(previous, state, offset):
        return model.decode_step(previous, state, session.context, offset=offset, training=training)

    def step_seed(offset):
        return None if seed is None else tf.stack([tf.cast(seed, tf.int32), tf.cast(offset, tf.int32)])

    return GenerationEngine(active_sampler, processors=processors).run(
        step_fn,
        session.previous,
        session.state,
        horizon,
        teacher=batch.future_values,
        teacher_observed_mask=batch.future_observed_mask,
        feedback_policy=FeedbackPolicy(teacher_probability, detach_feedback),
        context=session.context,
        seed_for_step=step_seed,
    )
