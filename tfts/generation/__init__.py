"""Forecast rollout, sampling, and continuous-value processing."""

from .api import generate, prepare_generation_batch
from .configuration import ForecastGenerationConfig
from .decoding import DecodeSession, IncrementalDecoder, decode
from .engine import RolloutOutput, TimeAxisEngine
from .feedback import TeacherForcingPolicy
from .outputs import ForecastGenerationOutput, GenerationOutput
from .processors import (
    CallableForecastProcessor,
    DifferenceClipProcessor,
    ForecastProcessor,
    ForecastProcessorList,
    RemoveInvalidValuesProcessor,
    ValueClipProcessor,
)
from .rollout import AutoregressiveRollout, DirectRollout, RecursiveRollout, RolloutStrategy, SampleAggregator
from .samplers import CallableSampler, DistributionSampler, PointSampler, StepOutput, ValueSampler
from .state import GenStep
from .stopping import MaxHorizon, StoppingCriteriaList, StoppingCriterion

__all__ = [
    "GenStep",
    "GenerationOutput",
    "TimeAxisEngine",
    "MaxHorizon",
    "StoppingCriteriaList",
    "StoppingCriterion",
    "SampleAggregator",
    "PointSampler",
    "TeacherForcingPolicy",
    "DecodeSession",
    "IncrementalDecoder",
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
    "RecursiveRollout",
    "RemoveInvalidValuesProcessor",
    "RolloutOutput",
    "RolloutStrategy",
    "StepOutput",
    "ValueClipProcessor",
    "ValueSampler",
    "generate",
    "prepare_generation_batch",
]
