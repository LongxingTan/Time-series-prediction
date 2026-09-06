"""One ordered processor pipeline for selection and feedback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, FrozenSet

import tensorflow as tf

from tfts.registry import build, register_processor

from .chunk import Chunk


class StepProcessor(ABC):
    needs_history: ClassVar[bool] = False
    requires: ClassVar[FrozenSet[str]] = frozenset()
    provides: ClassVar[FrozenSet[str]] = frozenset()

    @abstractmethod
    def __call__(self, chunk: Chunk) -> Chunk:
        raise NotImplementedError


class Selector(StepProcessor):
    stochastic: ClassVar[bool] = True
    provides = frozenset({"value"})


class Feedback(StepProcessor):
    requires = frozenset({"value"})
    provides = frozenset({"feedback"})


class CallableStepProcessor(StepProcessor):
    def __init__(self, fn):
        self.fn = fn
        self.needs_history = bool(getattr(fn, "needs_history", False))
        self.requires = frozenset(getattr(fn, "requires", ()))
        self.provides = frozenset(getattr(fn, "provides", ()))

    def __call__(self, chunk):
        return self.fn(chunk)


class StepProcessorList(list):
    def __init__(self, processors=()):
        resolved = []
        for processor in processors:
            processor = build("processor", processor)
            if not isinstance(processor, StepProcessor):
                if not callable(processor):
                    raise TypeError("every processor must be callable")
                processor = CallableStepProcessor(processor)
            resolved.append(processor)
        super().__init__(resolved)
        selectors = sum(isinstance(item, Selector) for item in self)
        if selectors != 1:
            raise ValueError(
                f"processors must contain exactly one Selector (got {selectors}). " "Add Mean(), Sample(), or Median()."
            )
        feedback = sum(isinstance(item, Feedback) for item in self)
        if feedback > 1:
            raise ValueError(f"processors may contain at most one Feedback (got {feedback})")
        available = {"offset", "past", "prediction", "parameters", "quantiles", "seed"}
        for processor in self:
            missing = processor.requires - available
            if missing:
                raise ValueError(f"{type(processor).__name__} requires {sorted(missing)} before they are available")
            available.update(processor.provides)

    @property
    def stochastic(self):
        return next(item.stochastic for item in self if isinstance(item, Selector))

    @property
    def needs_history(self):
        return any(item.needs_history for item in self)

    def __call__(self, chunk):
        for processor in self:
            chunk = processor(chunk)
            if not isinstance(chunk, Chunk):
                raise TypeError("processors must return Chunk")
        if chunk.feedback is None:
            chunk = chunk.replace(feedback=chunk.value)
        return chunk


@register_processor("mean")
class Mean(Selector):
    stochastic = False

    def __init__(self, distribution=None):
        self.distribution = distribution

    def __call__(self, chunk):
        if chunk.parameters is not None and self.distribution is not None:
            return chunk.replace(value=self.distribution.mean(chunk.parameters))
        return chunk.replace(value=chunk.prediction)


@register_processor("sample")
class Sample(Selector):
    def __init__(self, distribution=None):
        self.distribution = distribution

    def __call__(self, chunk):
        if chunk.parameters is None or self.distribution is None:
            raise ValueError("Sample requires distribution parameters")
        return chunk.replace(value=self.distribution.sample(chunk.parameters, seed=chunk.seed))


@register_processor("median")
class Median(Selector):
    stochastic = False

    def __call__(self, chunk):
        if chunk.quantiles is None:
            return chunk.replace(value=chunk.prediction)
        count = tf.shape(chunk.quantiles)[-1]
        ordered = tf.sort(chunk.quantiles, axis=-1)
        return chunk.replace(value=(ordered[..., (count - 1) // 2] + ordered[..., count // 2]) / 2.0)


@register_processor("quantile")
class Quantile(Selector):
    stochastic = False

    def __init__(self, q, levels=None, distribution=None):
        if not 0 < float(q) < 1:
            raise ValueError("q must lie strictly between 0 and 1")
        self.q = float(q)
        self.levels = None if levels is None else tuple(float(x) for x in levels)
        self.distribution = distribution

    def __call__(self, chunk):
        if chunk.quantiles is not None and self.levels:
            index = min(range(len(self.levels)), key=lambda i: abs(self.levels[i] - self.q))
            return chunk.replace(value=chunk.quantiles[..., index])
        if self.distribution is not None and hasattr(self.distribution, "quantile"):
            return chunk.replace(value=self.distribution.quantile(chunk.parameters, self.q))
        raise ValueError("Quantile requires quantile values or a distribution with quantile()")


@register_processor("clip")
class Clip(StepProcessor):
    requires = frozenset({"value"})
    provides = frozenset({"value"})

    def __init__(self, minimum=None, maximum=None):
        if minimum is None and maximum is None:
            raise ValueError("minimum or maximum is required")
        self.minimum, self.maximum = minimum, maximum

    def __call__(self, chunk):
        value = chunk.value
        if self.minimum is not None:
            value = tf.maximum(value, tf.cast(self.minimum, value.dtype))
        if self.maximum is not None:
            value = tf.minimum(value, tf.cast(self.maximum, value.dtype))
        return chunk.replace(value=value)


@register_processor("non_negative")
class NonNegative(StepProcessor):
    requires = frozenset({"value"})
    provides = frozenset({"value"})

    def __call__(self, chunk):
        return chunk.replace(value=tf.maximum(chunk.value, tf.cast(0, chunk.value.dtype)))


@register_processor("remove_invalid")
class RemoveInvalid(StepProcessor):
    requires = frozenset({"value"})
    provides = frozenset({"value"})

    def __init__(self, fallback=0.0):
        self.fallback = fallback

    def __call__(self, chunk):
        value = tf.where(
            tf.math.is_finite(chunk.value),
            chunk.value,
            tf.cast(self.fallback, chunk.value.dtype),
        )
        return chunk.replace(value=value)


@register_processor("difference_clip")
class DifferenceClip(StepProcessor):
    needs_history = True
    requires = frozenset({"value", "generated"})
    provides = frozenset({"value"})

    def __init__(self, max_decrease, max_increase=None):
        self.max_decrease = max_decrease
        self.max_increase = max_decrease if max_increase is None else max_increase

    def __call__(self, chunk):
        previous = tf.cond(
            tf.shape(chunk.generated)[1] > 0,
            lambda: chunk.generated[:, -1:, ...],
            lambda: chunk.past[:, -1:, ...],
        )
        lower = previous - tf.cast(self.max_decrease, chunk.value.dtype)
        upper = previous + tf.cast(self.max_increase, chunk.value.dtype)
        return chunk.replace(value=tf.clip_by_value(chunk.value, lower, upper))


@register_processor("teacher_forcing")
class TeacherForcing(Feedback):
    def __init__(self, targets, probability=1.0, observed_mask=None, detach=True):
        self.targets = targets
        self.probability = probability
        self.observed_mask = observed_mask
        self.detach = bool(detach)

    def __call__(self, chunk):
        value = tf.stop_gradient(chunk.value) if self.detach else chunk.value
        probability = tf.cast(self.probability, tf.float32)
        width = tf.shape(value)[1]
        target = tf.cast(self.targets[:, chunk.offset : chunk.offset + width, ...], value.dtype)
        shape = tf.concat([tf.shape(value)[:1], tf.ones([tf.rank(value) - 1], tf.int32)], axis=0)
        uniform = tf.random.stateless_uniform(shape, tf.random.experimental.stateless_fold_in(chunk.seed, 1))
        use_teacher = uniform < probability
        if self.observed_mask is not None:
            mask = self.observed_mask[:, chunk.offset : chunk.offset + width, ...]
            use_teacher = use_teacher & tf.cast(mask, tf.bool)
        return chunk.replace(feedback=tf.where(use_teacher, target, value))


@register_processor("noise_injection")
class NoiseInjection(Feedback):
    def __init__(self, std, ramp=1.0):
        self.std, self.ramp = std, ramp

    def __call__(self, chunk):
        scale = tf.cast(self.std, chunk.value.dtype) * tf.cast(self.ramp, chunk.value.dtype)
        noise = tf.random.stateless_normal(tf.shape(chunk.value), chunk.seed, dtype=chunk.value.dtype)
        return chunk.replace(feedback=chunk.value + scale * noise)


__all__ = [
    "Clip",
    "DifferenceClip",
    "Feedback",
    "Mean",
    "Median",
    "NoiseInjection",
    "NonNegative",
    "Quantile",
    "RemoveInvalid",
    "Sample",
    "Selector",
    "StepProcessor",
    "StepProcessorList",
    "TeacherForcing",
]
