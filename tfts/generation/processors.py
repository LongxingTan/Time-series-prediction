"""Composable post-selection processors for continuous forecast generation.

Language generation processors operate on token scores.  Forecasting has no token
vocabulary, so the useful equivalent operates on the continuous value selected by a
sampling strategy, before that value is returned and fed into the next decoding step.
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Callable, Iterable, Optional

import tensorflow as tf

from .samplers import SamplingResult, StepOutput


class ForecastProcessor(ABC):
    """Base class for a transformation or constraint applied during rollout."""

    @abstractmethod
    def __call__(
        self,
        history: Optional[tf.Tensor],
        result: SamplingResult,
        *,
        step: int,
        step_output: StepOutput,
        context: Any = None,
    ) -> SamplingResult:
        raise NotImplementedError


class CallableForecastProcessor(ForecastProcessor):
    """Adapt a callable returning either a tensor or :class:`SamplingResult`."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def __call__(self, history, result, *, step, step_output, context=None):
        processed = self.fn(history, result, step=step, step_output=step_output, context=context)
        if isinstance(processed, SamplingResult):
            return processed
        return replace(result, value=tf.convert_to_tensor(processed))


class ForecastProcessorList(list):
    """Apply forecast processors sequentially, like Transformers' processor lists."""

    def __init__(self, processors: Iterable[Any] = ()):
        super().__init__(resolve_forecast_processor(processor) for processor in processors)

    def __call__(self, history, result, *, step, step_output, context=None):
        for processor in self:
            result = processor(
                history,
                result,
                step=step,
                step_output=step_output,
                context=context,
            )
            if not isinstance(result, SamplingResult):
                raise TypeError("Forecast processors must return SamplingResult.")
        return result


class ValueClipProcessor(ForecastProcessor):
    """Clamp generated values to per-target physical or business bounds."""

    def __init__(self, minimum=None, maximum=None):
        if minimum is None and maximum is None:
            raise ValueError("At least one of minimum or maximum must be provided.")
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, history, result, *, step, step_output, context=None):
        value = result.value
        if self.minimum is not None:
            value = tf.maximum(value, tf.cast(self.minimum, value.dtype))
        if self.maximum is not None:
            value = tf.minimum(value, tf.cast(self.maximum, value.dtype))
        return replace(result, value=value)


class DifferenceClipProcessor(ForecastProcessor):
    """Limit the change from the previous generated value.

    The first generated step is unchanged because no generated history exists yet.
    ``max_decrease`` and ``max_increase`` may be scalars or one value per target.
    """

    def __init__(self, max_decrease, max_increase=None):
        self.max_decrease = max_decrease
        self.max_increase = max_decrease if max_increase is None else max_increase
        if tf.is_tensor(max_decrease) or tf.is_tensor(self.max_increase):
            return
        if any(float(value) < 0 for value in (max_decrease, self.max_increase)):
            raise ValueError("Difference limits must be non-negative.")

    def __call__(self, history, result, *, step, step_output, context=None):
        if history is None:
            return result
        previous = history[:, -1:, :]
        lower = previous - tf.cast(self.max_decrease, result.value.dtype)
        upper = previous + tf.cast(self.max_increase, result.value.dtype)
        return replace(result, value=tf.clip_by_value(result.value, lower, upper))


class RemoveInvalidValuesProcessor(ForecastProcessor):
    """Replace NaN and infinite generated values with a finite fallback."""

    def __init__(self, fallback=0.0):
        self.fallback = fallback

    def __call__(self, history, result, *, step, step_output, context=None):
        fallback = tf.cast(self.fallback, result.value.dtype)
        value = tf.where(tf.math.is_finite(result.value), result.value, fallback)
        return replace(result, value=value)


def resolve_forecast_processor(processor) -> ForecastProcessor:
    """Normalize a processor instance or callable."""
    if isinstance(processor, ForecastProcessor):
        return processor
    if callable(processor):
        return CallableForecastProcessor(processor)
    raise TypeError("Each processor must be a ForecastProcessor or callable.")


def resolve_forecast_processors(processors) -> ForecastProcessorList:
    """Normalize ``None``, one processor, or an iterable into a processor list."""
    if processors is None:
        return ForecastProcessorList()
    if isinstance(processors, ForecastProcessorList):
        return processors
    if isinstance(processors, ForecastProcessor) or callable(processors):
        return ForecastProcessorList([processors])
    return ForecastProcessorList(processors)
