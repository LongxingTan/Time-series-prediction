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
        self._register_configured_datasets()
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

    def _register_configured_datasets(self) -> None:
        """Register YAML aliases such as ``stallion: {type: npz, ...}``."""
        for name, dataset_config in self.config.per_dataset_config.items():
            dataset_type = dataset_config.get("type")
            if dataset_type and name not in self.dataset_registry:
                try:
                    dataset_class = self.dataset_registry.get(dataset_type)
                except KeyError as exc:
                    raise ValueError(f"Unknown dataset type '{dataset_type}' for '{name}'") from exc
                self.dataset_registry.register(name, dataset_class)

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
        ds_config.pop("type", None)
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
        model_overrides = self.config.get_model_config(model_name)
        if model_overrides:
            model_config.update(model_overrides)
        # Adjust input shape if known
        train_inputs = train_data[0]
        if hasattr(model_config, "input_shape"):
            if isinstance(train_inputs, dict):
                model_config.input_shape = {key: value.shape[1:] for key, value in train_inputs.items()}
            elif getattr(train_inputs, "ndim", 0) == 3:
                model_config.input_shape = train_inputs.shape[1:]

        model = AutoModel.from_config(model_config, prediction_length=predict_length)
        trainer = Trainer(model)
        training_options = self.config.get_model_training(model_name)
        loss_fn = self._resolve_loss(training_options.pop("loss", None), model_config)
        optimizer = self._resolve_optimizer(training_options.pop("optimizer", None), learning_rate)

        # Train
        history = trainer.train(
            train_dataset=train_data,
            valid_dataset=valid_data,
            epochs=epochs,
            batch_size=batch_size,
            loss_fn=loss_fn,
            optimizer=optimizer,
            verbose=0 if self.config.verbose < 2 else 1,
            **training_options,
        )

        # Evaluate
        x_valid, y_valid = valid_data
        try:
            raw_prediction = trainer.predict(x_valid)
        except AttributeError:
            # Distribution models can return a dictionary, while Trainer.predict
            # currently assumes a single tensor and calls .numpy() directly.
            raw_prediction = trainer.model(x_valid, training=False)
        y_pred = self._point_prediction(
            raw_prediction,
            model_config,
            self.config.get_model_prediction(model_name),
        )
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
            "model_config": model_overrides,
            "metrics": metrics,
            "history": {k: [float(v) for v in vals] for k, vals in (history.history if history else {}).items()},
        }
        return result

    @staticmethod
    def _resolve_loss(name: Optional[str], model_config: Any):
        if name is None:
            return None
        if name == "multi_quantile":
            from tfts.losses.loss import MultiQuantileLoss

            return MultiQuantileLoss(model_config.quantiles)
        if name == "smape":
            return lambda y_true, y_pred: tf.reduce_mean(
                200.0 * tf.abs(y_pred - y_true) / (tf.abs(y_true) + tf.abs(y_pred) + 1e-8)
            )
        return name

    @staticmethod
    def _resolve_optimizer(value: Any, learning_rate: float):
        if value is None:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if isinstance(value, str):
            optimizer = tf.keras.optimizers.get(value)
            optimizer.learning_rate = learning_rate
            return optimizer
        return tf.keras.optimizers.get(value)

    @staticmethod
    def _point_prediction(prediction: Any, model_config: Any, options: Dict[str, Any]) -> np.ndarray:
        if isinstance(prediction, dict):
            output_key = options.get("output_key")
            if output_key is None:
                output_key = "prediction" if "prediction" in prediction else "loc"
            if output_key not in prediction:
                raise ValueError(f"Prediction output '{output_key}' is unavailable: {list(prediction)}")
            prediction = prediction[output_key]
        prediction = np.asarray(prediction)
        quantiles = getattr(model_config, "quantiles", None)
        if options.get("quantile") is not None and quantiles and prediction.shape[-1] == len(quantiles):
            index = int(np.argmin(np.abs(np.asarray(quantiles) - float(options["quantile"]))))
            prediction = prediction[..., index : index + 1]
        return prediction

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
        M4Dataset,
        NpzDataset,
        RecruitRestaurantDataset,
        SineDataset,
    )

    registry.register("sine", SineDataset)
    registry.register("air_passengers", AirPassengersDataset)
    registry.register("grocery_sales", GrocerysalesDataset)
    registry.register("m4", M4Dataset)
    registry.register("npz", NpzDataset)
    registry.register("recruit_restaurant", RecruitRestaurantDataset)
    registry.register("forecasting_sticker_sales", ForecastingStickerSalesDataset)
    registry.register("CMI_detect_sleep_states", CMIDetectSleepStatesDataset)
    return registry
