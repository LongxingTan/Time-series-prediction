"""Forecast rollout, sampling, and continuous-value processing."""

from .api import generate
from .configuration import ForecastGenerationConfig
from .engine import GenerationEngine, RolloutOutput
from .outputs import ForecastGenerationOutput
from .processors import (
    CallableForecastProcessor,
    DifferenceClipProcessor,
    ForecastProcessor,
    ForecastProcessorList,
    RemoveInvalidValuesProcessor,
    ValueClipProcessor,
)
from .rollout import AutoregressiveRollout, DirectRollout, RecursiveRollout, RolloutStrategy
from .samplers import CallableSampler, DistributionSampler, MeanSampler, SamplingResult, StepOutput, ValueSampler

__all__ = [
    "AutoregressiveRollout",
    "CallableForecastProcessor",
    "CallableSampler",
    "DifferenceClipProcessor",
    "DirectRollout",
    "DistributionSampler",
    "ForecastGenerationConfig",
    "ForecastGenerationOutput",
    "ForecastProcessor",
    "ForecastProcessorList",
    "GenerationEngine",
    "MeanSampler",
    "RecursiveRollout",
    "RemoveInvalidValuesProcessor",
    "RolloutOutput",
    "RolloutStrategy",
    "SamplingResult",
    "StepOutput",
    "ValueClipProcessor",
    "ValueSampler",
    "generate",
]
