"""Stopping criteria evaluated between emitted chunks."""

from typing import Protocol

import tensorflow as tf

from .chunk import Chunk


class Stop(Protocol):
    def __call__(self, chunk: Chunk) -> tf.Tensor:
        """Return a scalar boolean stop request."""


class StoppingCriteriaList(list):
    def __init__(self, criteria=()):
        if any(not callable(item) for item in criteria):
            raise TypeError("every stopping criterion must be callable")
        super().__init__(criteria)

    def __call__(self, chunk):
        stopped = tf.constant(False)
        for criterion in self:
            result = tf.convert_to_tensor(criterion(chunk))
            tf.debugging.assert_rank(result, 0, message="stopping criteria must return a scalar boolean")
            stopped = stopped | tf.cast(result, tf.bool)
        return stopped


__all__ = ["Stop", "StoppingCriteriaList"]
