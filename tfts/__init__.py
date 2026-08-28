"""TFTS — Deep Learning for Time Series.

The primary composition is explicit: backbone config + task config -> task model.
"""

from tfts.cli import pipeline
from tfts.contracts import (
    AnomalyDetectionTaskConfig,
    ClassificationTaskConfig,
    ForecastTaskConfig,
    ImputationTaskConfig,
    TaskType,
    TimeSeriesBatch,
)
from tfts.data import (
    AutoPreprocessor,
    DataProcessor,
    SequenceMaterializer,
    TabularMaterializer,
    TimeSeriesSequence,
    WindowIndexer,
    WindowSpec,
    get_data,
)
from tfts.features import (
    AutoFeatureEngineer,
    FeaturePipeline,
    FeatureRegistry,
    FeatureRole,
    FeatureSelection,
    FeatureSpec,
    TimeSeriesSchema,
)
from tfts.generation import ForecastGenerationConfig
from tfts.metrics import evaluate as evaluate_metrics
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import (
    AutoBackbone,
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForAnomalyDetection,
    AutoModelForClassification,
    AutoModelForForecasting,
    AutoModelForImputation,
    AutoModelForPrediction,
    AutoModelForQuantile,
    AutoModelForTimeSeriesClassification,
)
from tfts.models.registry import list_models, list_supported_tasks, resolve_model_features
from tfts.saving import load_model
from tfts.tasks.pipeline import Pipeline, TaskPipeline
from tfts.trainer import EagerTrainer, KerasTrainer, Trainer, set_seed
from tfts.training_args import TrainingArguments
from tfts.tuner import OptunaTuner

try:
    import sys

    import benchmark as _benchmark
    from benchmark import BenchmarkConfig, BenchmarkResults, BenchmarkRunner, Dataset, DatasetRegistry, ModelRegistry
    import benchmark.base as _benchmark_base
    import benchmark.datasets as _benchmark_datasets
    import benchmark.formatter as _benchmark_formatter
    import benchmark.metrics as _benchmark_metrics
    import benchmark.registry as _benchmark_registry
    import benchmark.runner as _benchmark_runner

    sys.modules.setdefault("tfts.benchmark", _benchmark)
    sys.modules.setdefault("tfts.benchmark.base", _benchmark_base)
    sys.modules.setdefault("tfts.benchmark.datasets", _benchmark_datasets)
    sys.modules.setdefault("tfts.benchmark.formatter", _benchmark_formatter)
    sys.modules.setdefault("tfts.benchmark.metrics", _benchmark_metrics)
    sys.modules.setdefault("tfts.benchmark.registry", _benchmark_registry)
    sys.modules.setdefault("tfts.benchmark.runner", _benchmark_runner)

    _BENCHMARK_EXPORTS = [
        "BenchmarkConfig",
        "BenchmarkResults",
        "BenchmarkRunner",
        "Dataset",
        "DatasetRegistry",
        "ModelRegistry",
    ]
except (ImportError, ModuleNotFoundError):
    _BENCHMARK_EXPORTS = []

__all__ = [
    # -- Primary API --
    "pipeline",
    "ForecastingPipeline",
    "DataProcessor",
    # -- Preprocessing --
    "AutoPreprocessor",
    # -- Features --
    "AutoFeatureEngineer",
    "FeatureRegistry",
    "FeaturePipeline",
    "FeatureRole",
    "FeatureSelection",
    "FeatureSpec",
    "TimeSeriesSchema",
    # -- Models --
    "AutoModel",
    "AutoBackbone",
    "AutoModelForForecasting",
    "AutoModelForPrediction",
    "AutoModelForQuantile",
    "AutoModelForClassification",
    "AutoModelForTimeSeriesClassification",
    "AutoModelForImputation",
    "AutoModelForAnomaly",
    "AutoModelForAnomalyDetection",
    "AutoConfig",
    "ForecastTaskConfig",
    "ClassificationTaskConfig",
    "ImputationTaskConfig",
    "AnomalyDetectionTaskConfig",
    "ForecastGenerationConfig",
    "TaskType",
    "TimeSeriesBatch",
    "list_models",
    "list_supported_tasks",
    "resolve_model_features",
    "load_model",
    # -- Training --
    "Trainer",
    "KerasTrainer",
    "EagerTrainer",
    "TrainingArguments",
    "set_seed",
    # -- Tuning --
    "OptunaTuner",
    # -- Data --
    "get_data",
    "TimeSeriesSequence",
    "WindowIndexer",
    "WindowSpec",
    "TabularMaterializer",
    "SequenceMaterializer",
    # -- Evaluation --
    "evaluate_metrics",
    # -- Pipelines --
    "TaskPipeline",
    "Pipeline",
] + _BENCHMARK_EXPORTS

__version__ = "0.0.5"


def __getattr__(name: str):
    if name == "ForecastingPipeline":
        from tfts.cli.forecasting import ForecastingPipeline

        return ForecastingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
