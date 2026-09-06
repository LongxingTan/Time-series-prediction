"""Stopping hooks evaluated after each accepted forecast block."""

from typing import Protocol

import tensorflow as tf

from .state import GenStep


class StoppingCriterion(Protocol):
    def __call__(self, step: GenStep) -> tf.Tensor:
        """Return a scalar or per-example boolean stop request."""
        ...


class MaxHorizon:
    def __init__(self, horizon):
        self.horizon = horizon

    def __call__(self, step):
        return step.offset + tf.shape(step.value)[1] >= self.horizon


class StoppingCriteriaList(list):
    """Stop when any criterion requests termination for the whole batch.

    A per-example criterion terminates the batch when all examples agree.
    The returned horizon is rectangular; individual sequence lengths are not
    currently represented.
    """

    def __call__(self, step):
        result = tf.constant(False)
        for criterion in self:
            result = result | tf.reduce_all(tf.cast(criterion(step), tf.bool))
        return result
