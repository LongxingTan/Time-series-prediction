"""Chunked generation and its user extension points."""

from .api import generate, prepare_generation_batch
from .chunk import Chunk, State
from .config import GenerationConfig
from .decoders import Decoder, DirectDecoder, NativeDecoder, RecursiveDecoder, decoder_for
from .loop import run
from .processors import (
    Clip,
    DifferenceClip,
    Feedback,
    Mean,
    Median,
    NoiseInjection,
    NonNegative,
    Quantile,
    RemoveInvalid,
    Sample,
    Selector,
    StepProcessor,
    StepProcessorList,
    TeacherForcing,
)
from .stopping import Stop, StoppingCriteriaList
from .trajectory import (
    ConformalCalibrate,
    InverseScale,
    MeanSamples,
    MedianSamples,
    ReconcileHierarchy,
    RepairQuantileCrossing,
    Trajectory,
    TrajectoryTransform,
)

__all__ = [
    "Chunk",
    "Clip",
    "ConformalCalibrate",
    "Decoder",
    "DifferenceClip",
    "DirectDecoder",
    "Feedback",
    "GenerationConfig",
    "InverseScale",
    "Mean",
    "MeanSamples",
    "Median",
    "MedianSamples",
    "NativeDecoder",
    "NoiseInjection",
    "NonNegative",
    "Quantile",
    "ReconcileHierarchy",
    "RecursiveDecoder",
    "RemoveInvalid",
    "RepairQuantileCrossing",
    "Sample",
    "Selector",
    "State",
    "StepProcessor",
    "StepProcessorList",
    "Stop",
    "StoppingCriteriaList",
    "TeacherForcing",
    "Trajectory",
    "TrajectoryTransform",
    "decoder_for",
    "generate",
    "prepare_generation_batch",
    "run",
]
