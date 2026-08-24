"""Composable policies for selecting and feeding back forecast values."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import tensorflow as tf


@dataclass
class StepOutput:
    """Model output for one forecast step."""

    prediction: tf.Tensor
    state: Any = None
    distribution: Any = None
    parameters: Optional[Dict[str, tf.Tensor]] = None
    auxiliary: Optional[Dict[str, Any]] = None


@dataclass
class SamplingResult:
    """Value returned to the user and value used by the next decoding step."""

    value: tf.Tensor
    feedback: Optional[tf.Tensor] = None
    state: Any = None


class SamplingStrategy(ABC):
    """Select a value from a model's one-step output."""

    def initialize(self, **kwargs):
        return None

    @abstractmethod
    def sample(self, step_output: StepOutput, *, step: int, seed=None, teacher=None, state=None) -> SamplingResult:
        raise NotImplementedError


class GreedySampling(SamplingStrategy):
    """Use the model's point prediction."""

    def sample(self, step_output, *, step, seed=None, teacher=None, state=None):
        return SamplingResult(step_output.prediction, state=state)


class AncestralSampling(SamplingStrategy):
    """Draw from the model's predictive distribution."""

    def sample(self, step_output, *, step, seed=None, teacher=None, state=None):
        if step_output.distribution is None or step_output.parameters is None:
            raise ValueError("AncestralSampling requires a distribution and its parameters.")
        value = step_output.distribution.sample(step_output.parameters, seed=seed)
        return SamplingResult(value, state=state)


class TeacherForcingSampling(SamplingStrategy):
    """Return the prediction but feed the supplied next decoder input."""

    def sample(self, step_output, *, step, seed=None, teacher=None, state=None):
        if teacher is None:
            raise ValueError("TeacherForcingSampling requires a teacher tensor.")
        return SamplingResult(step_output.prediction, feedback=teacher[:, step : step + 1, :], state=state)


class CallableSampling(SamplingStrategy):
    """Adapt a callable to :class:`SamplingStrategy`."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def sample(self, step_output, *, step, seed=None, teacher=None, state=None):
        result = self.fn(step_output, step=step, seed=seed, teacher=teacher, state=state)
        if isinstance(result, SamplingResult):
            return result
        return SamplingResult(result, state=state)


class FeedbackPolicy(ABC):
    """Transform a selected value into the next model input."""

    @abstractmethod
    def update(self, current, result: SamplingResult, *, step: int, context=None):
        raise NotImplementedError


class ValueFeedback(FeedbackPolicy):
    """Feed ``result.feedback`` when present, otherwise ``result.value``."""

    def update(self, current, result, *, step, context=None):
        return result.value if result.feedback is None else result.feedback


class CallableFeedback(FeedbackPolicy):
    """Adapt a feature/window update callable to :class:`FeedbackPolicy`."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def update(self, current, result, *, step, context=None):
        return self.fn(current, result, step=step, context=context)


class FullFeatureFeedback(FeedbackPolicy):
    """Append a predicted complete feature row to a fixed rolling window."""

    def update(self, current, result, *, step, context=None):
        value = result.value if result.feedback is None else result.feedback
        if current.shape[-1] != value.shape[-1]:
            raise ValueError("FullFeatureFeedback requires the prediction to contain every input feature.")
        return tf.concat([current[:, 1:, :], value], axis=1)


class TargetFeedback(FeedbackPolicy):
    """Replace selected columns in the last row and append it to the window.

    Non-target columns are carried forward. For known future covariates, subclass this
    policy or use :class:`CallableFeedback` to construct the complete next row.
    """

    def __init__(self, target_indices=(0,)):
        self.target_indices = tuple(int(i) for i in target_indices)

    def update(self, current, result, *, step, context=None):
        value = result.value if result.feedback is None else result.feedback
        feature_count = current.shape[-1]
        if feature_count is None:
            raise ValueError("TargetFeedback requires a statically known feature dimension.")
        if value.shape[-1] is not None and int(value.shape[-1]) != len(self.target_indices):
            raise ValueError("Prediction width must match target_indices.")
        indices = tf.constant(self.target_indices, dtype=tf.int32)
        weights = tf.one_hot(indices, int(feature_count), dtype=current.dtype)
        updates = tf.einsum("btd,df->btf", tf.cast(value, current.dtype), weights)
        mask = tf.reduce_sum(weights, axis=0)[None, None, :]
        next_row = current[:, -1:, :] * (1.0 - mask) + updates
        return tf.concat([current[:, 1:, :], next_row], axis=1)


_STRATEGIES = {
    "ancestral": AncestralSampling,
    "sample": AncestralSampling,
    "greedy": GreedySampling,
    "recursive": GreedySampling,
    "teacher_forced": TeacherForcingSampling,
}


def register_sampling_strategy(name: str, strategy_class):
    """Register a named strategy class without requiring registration for custom use."""
    if not name or not isinstance(name, str):
        raise ValueError("Strategy name must be a non-empty string.")
    if not issubclass(strategy_class, SamplingStrategy):
        raise TypeError("strategy_class must inherit SamplingStrategy.")
    _STRATEGIES[name] = strategy_class


def resolve_sampling_strategy(strategy) -> SamplingStrategy:
    if isinstance(strategy, SamplingStrategy):
        return strategy
    if callable(strategy) and not isinstance(strategy, str):
        return CallableSampling(strategy)
    if isinstance(strategy, str) and strategy in _STRATEGIES:
        return _STRATEGIES[strategy]()
    raise ValueError(f"Unknown sampling strategy: {strategy!r}")


def resolve_feedback_policy(feedback) -> FeedbackPolicy:
    if feedback is None:
        return ValueFeedback()
    if isinstance(feedback, FeedbackPolicy):
        return feedback
    if callable(feedback):
        return CallableFeedback(feedback)
    raise TypeError("feedback must be a FeedbackPolicy, callable, or None.")
