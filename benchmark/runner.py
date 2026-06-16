"""Benchmark runner for the TFTS benchmark system.

Orchestrates running multiple models on multiple datasets with multiple runs
and collects results."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from benchmark.base import BenchmarkConfig, Dataset
from benchmark.formatter import BenchmarkResults
from benchmark.metrics import BenchmarkMetrics
from benchmark.registry import DatasetRegistry, ModelRegistry
from tfts import AutoConfig, AutoModel, Trainer, set_seed

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run a benchmark experiment.

    Example::

        from tfts.benchmark import BenchmarkRunner, BenchmarkConfig

        config = BenchmarkConfig(
            models=["rnn", "transformer"],
            datasets=["sine", "air_passengers"],
            metrics=["mae", "rmse"],
            runs=3,
            epochs=50,
        )
        runner = BenchmarkRunner(config)
        results = runner.run()
        results.to_latex("results.tex")
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        dataset_registry: Optional[DatasetRegistry] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        self.config = config
        self.dataset_registry = dataset_registry or _default_dataset_registry()
        self.model_registry = model_registry or ModelRegistry()
        self.metrics = BenchmarkMetrics(config.metrics)
        self.results: List[Dict[str, Any]] = []

    def run(self) -> BenchmarkResults:
        """Execute the benchmark and return results.

        Returns:
            BenchmarkResults: Container with raw and formatted results.
        """
        datasets = self._resolve_datasets()
        models = self._resolve_models()

        logger.info("=" * 60)
        logger.info("Starting TFTS Benchmark")
        logger.info("Models: %s", models)
        logger.info("Datasets: %s", datasets)
        logger.info("Runs per experiment: %d", self.config.runs)
        logger.info("=" * 60)

        for dataset_name in datasets:
            for model_name in models:
                self._run_experiment(dataset_name, model_name)

        results = BenchmarkResults(self.results)
        self._save_results(results)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_datasets(self) -> List[str]:
        """Return the actual list of dataset names to run."""
        registered = self.dataset_registry.list_datasets()
        if self.config.datasets == ["all"]:
            return registered
        missing = set(self.config.datasets) - set(registered)
        if missing:
            raise ValueError(f"Unknown datasets: {missing}. Available: {registered}")
        return self.config.datasets

    def _resolve_models(self) -> List[str]:
        """Return the actual list of model names to run."""
        available = self.model_registry.available_models
        if self.config.models == ["all"]:
            return available
        missing = set(self.config.models) - set(available)
        if missing:
            raise ValueError(f"Unknown models: {missing}. Available: {available}")
        return self.config.models

    def _run_experiment(self, dataset_name: str, model_name: str) -> None:
        """Run all trials for a single dataset-model pair."""
        logger.info("-" * 60)
        logger.info("Experiment: %s / %s", dataset_name, model_name)

        ds_config = self.config.get_dataset_config(dataset_name)
        cls_ = self.dataset_registry.get(dataset_name)
        dataset = cls_()

        for run_idx in range(self.config.runs):
            seed = self.config.seed + run_idx
            set_seed(seed)

            result = self._run_single_trial(
                dataset=dataset,
                dataset_name=dataset_name,
                model_name=model_name,
                run_idx=run_idx,
                seed=seed,
                ds_config=ds_config,
            )
            self.results.append(result)

    def _run_single_trial(
        self,
        dataset: Dataset,
        dataset_name: str,
        model_name: str,
        run_idx: int,
        seed: int,
        ds_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single trial and return the result dict."""
        logger.info("  Run %d/%d (seed=%d)", run_idx + 1, self.config.runs, seed)

        train_data, valid_data = dataset.get_train_valid_split(**ds_config)

        train_length = ds_config.get("train_length") or dataset.train_length
        predict_length = ds_config.get("predict_sequence_length") or dataset.predict_sequence_length
        epochs = ds_config.get("epochs", self.config.epochs)
        batch_size = ds_config.get("batch_size", self.config.batch_size)
        learning_rate = ds_config.get("learning_rate", self.config.learning_rate)

        # Build model
        model_config = AutoConfig.for_model(model_name)
        # Adjust input shape if known
        if hasattr(model_config, "input_shape") and train_data[0].ndim == 3:
            model_config.input_shape = train_data[0].shape[1:]

        model = AutoModel.from_config(model_config, predict_sequence_length=predict_length)
        trainer = Trainer(model)

        # Train
        history = trainer.train(
            train_dataset=train_data,
            valid_dataset=valid_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0 if self.config.verbose < 2 else 1,
        )

        # Evaluate
        x_valid, y_valid = valid_data
        y_pred = trainer.predict(x_valid)
        metrics = self.metrics.compute(y_valid, y_pred)

        result = {
            "dataset": dataset_name,
            "model": model_name,
            "run": run_idx,
            "seed": seed,
            "train_length": train_length,
            "predict_sequence_length": predict_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "metrics": metrics,
            "history": {k: [float(v) for v in vals] for k, vals in (history.history if history else {}).items()},
        }
        return result

    def _save_results(self, results: BenchmarkResults) -> None:
        """Save results to the output directory."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        results.to_json(os.path.join(self.config.output_dir, "results.json"))
        results.to_csv(os.path.join(self.config.output_dir, "results.csv"))
        results.to_latex(os.path.join(self.config.output_dir, "results.tex"))


# --------------------------------------------------------------------------
# Default registry population
# --------------------------------------------------------------------------


def _default_dataset_registry() -> DatasetRegistry:
    """Build a :class:`DatasetRegistry` with built-in datasets."""
    registry = DatasetRegistry()

    # Lazy import to avoid circular dependency at package top-level
    from benchmark.datasets import (
        AirPassengersDataset,
        CMIDetectSleepStatesDataset,
        ForecastingStickerSalesDataset,
        GrocerysalesDataset,
        RecruitRestaurantDataset,
        SineDataset,
    )

    registry.register("sine", SineDataset)
    registry.register("air_passengers", AirPassengersDataset)
    registry.register("grocery_sales", GrocerysalesDataset)
    registry.register("recruit_restaurant", RecruitRestaurantDataset)
    registry.register("forecasting_sticker_sales", ForecastingStickerSalesDataset)
    registry.register("CMI_detect_sleep_states", CMIDetectSleepStatesDataset)
    return registry
