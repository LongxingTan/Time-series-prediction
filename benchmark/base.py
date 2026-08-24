"""Base classes for the TFTS benchmark system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        models: List of model names to evaluate (e.g., ["rnn", "transformer"]).
            Use ``["all"]`` to run all registered models.
        datasets: List of dataset names to evaluate (e.g., ["sine", "grocery_sales"]).
            Use ``["all"]`` to run all registered datasets.
        metrics: List of metric names to compute.
            Available: ``"mae"``, ``"mse"``, ``"rmse"``, ``"mape"``, ``"smape"``, ``"r2"``.
        runs: Number of runs per model-dataset pair (for statistical significance).
        epochs: Number of training epochs per run.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        train_length: Lookback window length. If None, each dataset provides its own.
        predict_sequence_length: Forecast horizon. If None, each dataset provides its own.
        seed: Base seed. Each run uses ``seed + run_idx`` for reproducibility.
        output_dir: Directory to save results.
        save_models: Whether to save trained model weights.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        device: Device to run on (e.g., ``"/gpu:0"`` or ``"/cpu:0"``).
        per_dataset_config: Optional per-dataset configuration overrides.
            Keys are dataset names, values are dicts with keys like ``train_length``,
            ``predict_sequence_length``, ``epochs``, etc.
        per_model_config: Optional model architecture overrides. Keys are model
            names and values are applied to that model's TFTS config.
    """

    models: List[str] = field(default_factory=lambda: ["all"])
    datasets: List[str] = field(default_factory=lambda: ["all"])
    metrics: List[str] = field(default_factory=lambda: ["mae", "rmse", "mape"])
    runs: int = 1
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    train_length: Optional[int] = None
    predict_sequence_length: Optional[int] = None
    seed: int = 42
    output_dir: str = "benchmark_results"
    save_models: bool = False
    verbose: int = 1
    device: str = ""
    per_dataset_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_model_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_model_training: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_model_prediction: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.runs < 1:
            raise ValueError("runs must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")

    @classmethod
    def from_yaml(cls, path: Union[str, os.PathLike]) -> "BenchmarkConfig":
        """Load a benchmark configuration from a YAML file.

        ``models`` and ``datasets`` may be simple lists, or mappings whose
        values contain per-item configuration. Global options can be placed
        under ``benchmark`` (recommended) or at the top level.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load benchmark YAML files") from exc

        with open(path, encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Benchmark YAML must contain a mapping at the top level")

        benchmark_options = raw.pop("benchmark", {})
        if not isinstance(benchmark_options, dict):
            raise ValueError("'benchmark' must be a mapping")
        options = dict(raw)
        options.update(benchmark_options)

        models, model_config = cls._parse_named_items(options.pop("models", ["all"]), "models")
        datasets, dataset_config = cls._parse_named_items(options.pop("datasets", ["all"]), "datasets")

        explicit_dataset_config = options.pop("per_dataset_config", {})
        explicit_model_config = options.pop("per_model_config", {})
        if not isinstance(explicit_dataset_config, dict) or not isinstance(explicit_model_config, dict):
            raise ValueError("per_dataset_config and per_model_config must be mappings")
        dataset_config.update(explicit_dataset_config)
        model_config.update(explicit_model_config)

        model_training = options.pop("per_model_training", {})
        model_prediction = options.pop("per_model_prediction", {})
        for name, item in list(model_config.items()):
            if any(key in item for key in ("config", "training", "prediction")):
                architecture = item.get("config", {})
                if not isinstance(architecture, dict):
                    raise ValueError(f"Model config for '{name}' must be a mapping")
                model_training[name] = item.get("training", {})
                model_prediction[name] = item.get("prediction", {})
                model_config[name] = architecture
        if not isinstance(model_training, dict) or not isinstance(model_prediction, dict):
            raise ValueError("per_model_training and per_model_prediction must be mappings")
        for section_name, section in (("training", model_training), ("prediction", model_prediction)):
            if not all(isinstance(value, dict) for value in section.values()):
                raise ValueError(f"All model {section_name} options must be mappings")

        valid_fields = set(cls.__dataclass_fields__) - {
            "models",
            "datasets",
            "per_dataset_config",
            "per_model_config",
            "per_model_training",
            "per_model_prediction",
        }
        unknown = set(options) - valid_fields
        if unknown:
            raise ValueError(f"Unknown benchmark configuration options: {sorted(unknown)}")
        return cls(
            models=models,
            datasets=datasets,
            per_dataset_config=dataset_config,
            per_model_config=model_config,
            per_model_training=model_training,
            per_model_prediction=model_prediction,
            **options,
        )

    @staticmethod
    def _parse_named_items(value: Any, field_name: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Normalize a YAML list or ``name: config`` mapping."""
        if isinstance(value, list):
            if not all(isinstance(name, str) for name in value):
                raise ValueError(f"'{field_name}' list entries must be strings")
            return value, {}
        if isinstance(value, dict):
            config = {}
            for name, item_config in value.items():
                if not isinstance(name, str):
                    raise ValueError(f"'{field_name}' names must be strings")
                if item_config is None:
                    item_config = {}
                if not isinstance(item_config, dict):
                    raise ValueError(f"Configuration for {field_name} '{name}' must be a mapping")
                config[name] = item_config
            return list(value), config
        raise ValueError(f"'{field_name}' must be a list or mapping")

    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset, merging with defaults."""
        config = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "train_length": self.train_length,
            "predict_sequence_length": self.predict_sequence_length,
        }
        if dataset_name in self.per_dataset_config:
            config.update(self.per_dataset_config[dataset_name])
        return config

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Return model architecture overrides for a specific model."""
        return dict(self.per_model_config.get(model_name, {}))

    def get_model_training(self, model_name: str) -> Dict[str, Any]:
        """Return Trainer overrides for a specific model."""
        return dict(self.per_model_training.get(model_name, {}))

    def get_model_prediction(self, model_name: str) -> Dict[str, Any]:
        """Return point-forecast extraction options for a specific model."""
        return dict(self.per_model_prediction.get(model_name, {}))


class Dataset(ABC):
    """Abstract base class for benchmark datasets.

        Each dataset subclass implements ``prepare_data()`` to load and format
    the data. The class also provides metadata about the dataset.

        Attributes:
            name: Unique identifier for the dataset.
            description: Human-readable description.
            train_length: Default lookback window length (can be overridden by config).
            predict_sequence_length: Default forecast horizon (can be overridden by config).
    """

    name: str = ""
    description: str = ""
    train_length: int = 24
    predict_sequence_length: int = 8
    num_features: int = 1
    is_multivariate: bool = False
    is_grouped: bool = False
    target_column: str = "target"
    time_column: str = "time"
    group_column: Optional[str] = None

    @abstractmethod
    def prepare_data(self, **kwargs) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        tf.data.Dataset,
    ]:
        """Prepare and return the dataset.

        Returns:
            Either (x_train, y_train), ((x_train, y_train), (x_valid, y_valid)),
            or a tf.data.Dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_valid_split(self, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Return train and validation splits.

        Returns:
            Tuple of (x_train, y_train), (x_valid, y_valid).
        """
        raise NotImplementedError

    @classmethod
    def list_params(cls) -> Dict[str, Any]:
        """Return dataset parameters for display."""
        return {
            "name": cls.name,
            "description": cls.description,
            "train_length": cls.train_length,
            "predict_sequence_length": cls.predict_sequence_length,
            "num_features": cls.num_features,
            "is_multivariate": cls.is_multivariate,
            "is_grouped": cls.is_grouped,
        }
