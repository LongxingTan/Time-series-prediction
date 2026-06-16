"""Base classes for the TFTS benchmark system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
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

    def __post_init__(self):
        if self.runs < 1:
            raise ValueError("runs must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")

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
    def prepare_data(
        self, **kwargs
    ) -> Union[
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
