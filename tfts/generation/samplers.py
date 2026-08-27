"""Value selection from one forecast step."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import tensorflow as tf


@dataclass
class StepOutput:
    prediction: tf.Tensor
    state: Any = None
    distribution: Any = None
    parameters: Optional[Dict[str, tf.Tensor]] = None


@dataclass
class SamplingResult:
    value: tf.Tensor
    state: Any = None


class ValueSampler(ABC):
    @abstractmethod
    def sample(self, step_output: StepOutput, *, step: int, seed=None) -> SamplingResult:
        raise NotImplementedError


class MeanSampler(ValueSampler):
    def sample(self, step_output, *, step, seed=None):
        return SamplingResult(step_output.prediction)


class DistributionSampler(ValueSampler):
    def sample(self, step_output, *, step, seed=None):
        if step_output.distribution is None or step_output.parameters is None:
            raise ValueError("DistributionSampler requires distribution parameters")
        return SamplingResult(step_output.distribution.sample(step_output.parameters, seed=seed))


class CallableSampler(ValueSampler):
    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def sample(self, step_output, *, step, seed=None):
        result = self.fn(step_output, step=step, seed=seed)
        return result if isinstance(result, SamplingResult) else SamplingResult(result)


def resolve_value_sampler(sampler, probabilistic=False) -> ValueSampler:
    if sampler is None or sampler == "auto":
        return DistributionSampler() if probabilistic else MeanSampler()
    if isinstance(sampler, ValueSampler):
        return sampler
    if callable(sampler) and not isinstance(sampler, str):
        return CallableSampler(sampler)
    mapping = {"mean": MeanSampler, "sample": DistributionSampler}
    try:
        return mapping[sampler]()
    except KeyError as error:
        raise ValueError("Unknown sampler %r. Available: %s" % (sampler, sorted(mapping))) from error
