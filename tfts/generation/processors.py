"""Parameter-space and value-space transformations of generation state."""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Literal

import tensorflow as tf

from .state import GenStep


class ForecastProcessor(ABC):
    stage: Literal["parameters", "values"] = "values"

    @abstractmethod
    def __call__(self, step: GenStep) -> GenStep:
        raise NotImplementedError


class CallableForecastProcessor(ForecastProcessor):
    def __init__(self, fn):
        self.fn = fn
        self.stage = getattr(fn, "stage", "values")

    def __call__(self, step):
        return self.fn(step)


class ForecastProcessorList(list):
    def __init__(self, processors=()):
        super().__init__(resolve_forecast_processor(p) for p in processors)
        if any(p.stage not in {"parameters", "values"} for p in self):
            raise ValueError("Processor stage must be parameters or values")

    def __call__(self, step, *, stage):
        for processor in self:
            if processor.stage == stage:
                step = processor(step)
                if not isinstance(step, GenStep):
                    raise TypeError("Forecast processors must return GenStep")
        return step


class ValueClipProcessor(ForecastProcessor):
    """Clamp selected values; this does not implement a truncated distribution."""

    def __init__(self, minimum=None, maximum=None):
        if minimum is None and maximum is None:
            raise ValueError("At least one of minimum or maximum must be provided.")
        self.minimum, self.maximum = minimum, maximum

    def __call__(self, step):
        value = step.value
        if self.minimum is not None:
            value = tf.maximum(value, tf.cast(self.minimum, value.dtype))
        if self.maximum is not None:
            value = tf.minimum(value, tf.cast(self.maximum, value.dtype))
        return replace(step, value=value)


class DifferenceClipProcessor(ForecastProcessor):
    """Limit change from the previous emitted value, or last observation."""

    def __init__(self, max_decrease, max_increase=None):
        self.max_decrease = max_decrease
        self.max_increase = max_decrease if max_increase is None else max_increase
        for limit in (self.max_decrease, self.max_increase):
            if not tf.is_tensor(limit) and float(limit) < 0:
                raise ValueError("Difference limits must be non-negative")

    def __call__(self, step):
        decrease = tf.cast(self.max_decrease, step.value.dtype)
        increase = tf.cast(self.max_increase, step.value.dtype)
        tf.debugging.assert_non_negative(decrease)
        tf.debugging.assert_non_negative(increase)
        previous = tf.cond(
            tf.shape(step.generated)[1] > 0, lambda: step.generated[:, -1:, ...], lambda: step.past_values[:, -1:, ...]
        )
        return replace(step, value=tf.clip_by_value(step.value, previous - decrease, previous + increase))


class RemoveInvalidValuesProcessor(ForecastProcessor):
    def __init__(self, fallback=0.0):
        self.fallback = fallback

    def __call__(self, step):
        return replace(
            step, value=tf.where(tf.math.is_finite(step.value), step.value, tf.cast(self.fallback, step.value.dtype))
        )


def resolve_forecast_processor(processor):
    if isinstance(processor, ForecastProcessor):
        return processor
    if callable(processor):
        return CallableForecastProcessor(processor)
    raise TypeError("Each processor must be a ForecastProcessor or callable")


def resolve_forecast_processors(processors):
    if processors is None:
        return ForecastProcessorList()
    if isinstance(processors, ForecastProcessorList):
        return processors
    if callable(processors):
        processors = [processors]
    return ForecastProcessorList(processors)
