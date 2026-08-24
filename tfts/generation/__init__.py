"""Autoregressive / generative inference utilities for tfts models.

Fifth component of the 5-stage pipeline (sample aggregation and the rollout driver):

    model architecture -> predictive distribution -> autoregressive
    generator -> sample aggregation -> inverse transformation

The public entry point is ``ForecastGenerationConfig`` + ``ForecastGenerationOutput``,
driven by the ``AutoregressiveGenerationMixin`` (``model.generate(...)``). The split
between a *distribution* (``tfts.distributions``) and a *sampling/aggregation control
object* (here) is what makes the sampled point forecast an optional, explicit feature
instead of a baked-in behaviour of the model ``__call__``.
"""

from .autoregressive import AutoregressiveGenerationMixin
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
from .rolling import RollingWindowGenerationMixin
from .schedules import FeedbackTrainingConfig, ScheduleConfig
from .strategy import (
    AncestralSampling,
    CallableFeedback,
    CallableSampling,
    FeedbackPolicy,
    FullFeatureFeedback,
    GreedySampling,
    SamplingResult,
    SamplingStrategy,
    StepOutput,
    TargetFeedback,
    TeacherForcingSampling,
    ValueFeedback,
    register_sampling_strategy,
)

__all__ = [
    "ForecastGenerationConfig",
    "ForecastGenerationOutput",
    "ForecastProcessor",
    "ForecastProcessorList",
    "CallableForecastProcessor",
    "ValueClipProcessor",
    "DifferenceClipProcessor",
    "RemoveInvalidValuesProcessor",
    "AutoregressiveGenerationMixin",
    "ScheduleConfig",
    "FeedbackTrainingConfig",
    "GenerationEngine",
    "RolloutOutput",
    "StepOutput",
    "SamplingResult",
    "SamplingStrategy",
    "GreedySampling",
    "AncestralSampling",
    "TeacherForcingSampling",
    "CallableSampling",
    "FeedbackPolicy",
    "ValueFeedback",
    "CallableFeedback",
    "FullFeatureFeedback",
    "TargetFeedback",
    "RollingWindowGenerationMixin",
    "register_sampling_strategy",
]
