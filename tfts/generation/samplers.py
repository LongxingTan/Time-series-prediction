"""Value selection from one forecast step."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from .state import Distribution, GenStep


@dataclass
class StepOutput:
    prediction: tf.Tensor
    state: Any = None
    distribution: Optional[Distribution] = None
    parameters: Optional[Dict[str, tf.Tensor]] = None
    quantile_values: Optional[tf.Tensor] = None


class ValueSampler(ABC):
    stochastic: bool = False

    @abstractmethod
    def __call__(self, step: GenStep, *, seed=None) -> tf.Tensor:
        raise NotImplementedError


class PointSampler(ValueSampler):
    def __call__(self, step, *, seed=None):
        return step.prediction


class DistributionSampler(ValueSampler):
    stochastic = True

    def __call__(self, step, *, seed=None):
        if step.distribution is None or step.parameters is None:
            raise ValueError("DistributionSampler requires distribution parameters")
        return step.distribution.sample(step.parameters, seed=seed)


class CallableSampler(ValueSampler):
    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def __call__(self, step, *, seed=None):
        return self.fn(step, seed=seed)


def resolve_value_sampler(sampler, probabilistic=False) -> ValueSampler:
    """Resolve value selection consistently for every decoding strategy.

    ``auto`` samples probabilistic outputs and uses predictions otherwise.
    """
    mapping = {"point": PointSampler, "sample": DistributionSampler}
    if sampler is None or sampler == "auto":
        resolved = DistributionSampler() if probabilistic else PointSampler()
    elif isinstance(sampler, ValueSampler):
        resolved = sampler
    elif callable(sampler):
        resolved = CallableSampler(sampler)
    elif sampler in mapping:
        resolved = mapping[sampler]()
    else:
        raise ValueError("Unknown sampler %r. Available: %s" % (sampler, sorted(mapping)))
    if isinstance(resolved, DistributionSampler) and not probabilistic:
        raise ValueError("sampler='sample' requires a model output distribution")
    return resolved
