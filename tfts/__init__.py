"""TFTS — Deep Learning for Time Series.

Usage:
    >>> import tfts
    >>> pipe = tfts.pipeline("forecasting", model="dlinear", lookback=96, horizon=24)
    >>> pipe.fit(df, target_col="value", epochs=50)
    >>> preds = pipe.predict(steps=24)
    >>> tfts.list_models()
"""

from tfts.cli import pipeline
from tfts.data import AutoPreprocessor, DataProcessor, TimeSeriesSequence, get_data
from tfts.features import AutoFeatureEngineer, FeatureRegistry
from tfts.metrics import evaluate as evaluate_metrics
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import (
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForClassification,
    AutoModelForPrediction,
    AutoModelForSegmentation,
    AutoModelForUncertainty,
)
from tfts.models.registry import list_models
from tfts.saving import load_model

# Legacy compatibility
from tfts.tasks.pipeline import Pipeline
from tfts.trainer import EagerTrainer, KerasTrainer, Trainer, set_seed
from tfts.training.exposure_bias import (
    add_exposure_bias_noise,
    add_exposure_bias_noise_np,
    annealed_noise_std,
    position_ramp,
)
from tfts.training.scheduled_sampling import scheduled_sampling_decode, teacher_forcing_decay
from tfts.training.window_trainer import WindowedTrainer, final_windows, sampled_windows, smape_score
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
    # -- Models --
    "AutoModel",
    "AutoModelForPrediction",
    "AutoModelForClassification",
    "AutoModelForSegmentation",
    "AutoModelForAnomaly",
    "AutoModelForUncertainty",
    "AutoConfig",
    "list_models",
    "load_model",
    # -- Training --
    "Trainer",
    "KerasTrainer",
    "EagerTrainer",
    "WindowedTrainer",
    "TrainingArguments",
    "set_seed",
    "final_windows",
    "sampled_windows",
    "smape_score",
    # -- Exposure-bias (autoregressive) regularizers --
    "add_exposure_bias_noise",
    "add_exposure_bias_noise_np",
    "annealed_noise_std",
    "position_ramp",
    "scheduled_sampling_decode",
    "teacher_forcing_decay",
    # -- Tuning --
    "OptunaTuner",
    # -- Data --
    "get_data",
    "TimeSeriesSequence",
    # -- Evaluation --
    "evaluate_metrics",
    # -- Legacy --
    "Pipeline",
] + _BENCHMARK_EXPORTS

__version__ = "0.0.5"


def __getattr__(name: str):
    if name == "ForecastingPipeline":
        from tfts.cli.forecasting import ForecastingPipeline

        return ForecastingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
