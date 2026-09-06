"""Forecast rollout, sampling, and continuous-value processing."""

from .api import generate, prepare_generation_batch
from .configuration import ForecastGenerationConfig
from .decoding import DecodeSession, IncrementalDecoder, decode
from .engine import GenerationEngine, RolloutOutput
from .feedback import FeedbackPolicy
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
    "DecodeSession",
    "IncrementalDecoder",
    "FeedbackPolicy",
    "decode",
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
    "prepare_generation_batch",
]
