"""Small task-aware estimator over ordinary TFTS Keras models."""

import copy
from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.batch import TimeSeriesBatch
from tfts.data.arrays import ArrayPreprocessor, input_arrays, target_arrays
from tfts.data.features import Schema
from tfts.data.pipeline import FramePipeline, index_frame
from tfts.models.auto_config import AutoConfig
from tfts.models.catalog import get_model_capabilities
from tfts.registry import resolve
from tfts.tasks.auto import AutoModel
from tfts.tasks.base import TaskType
from tfts.training import Trainer


class NotFittedError(RuntimeError):
    """The estimator has no successfully fitted model."""


class Estimator:
    """Fit a registered model to arrays, a dataset, or a forecasting dataframe.

    Construction only records parameters. Model construction and preprocessing
    happen in ``fit``; fitted state ends in ``_``. Model authors implement a
    Keras model and register it beside its config, without editing this class.
    """

    _PARAMETERS = (
        "task",
        "model",
        "config",
        "task_config",
        "objective",
        "context",
        "horizon",
        "schema",
        "calendar",
        "freq",
        "validation",
        "normalize",
        "batch_size",
        "learning_rate",
        "epochs",
        "early_stopping_patience",
        "seed",
    )

    def __init__(
        self,
        task="forecasting",
        model="rnn",
        *,
        config=None,
        task_config=None,
        objective=None,
        context=96,
        horizon=24,
        schema=None,
        calendar=(),
        freq=None,
        validation=0.2,
        normalize="standard",
        batch_size=32,
        learning_rate=1e-3,
        epochs=50,
        early_stopping_patience=5,
        seed=42,
        **model_options,
    ):
        self.task = task
        self.model = model
        self.config = config
        self.task_config = task_config
        self.objective = objective
        self.context = context
        self.horizon = horizon
        self.schema = schema
        self.calendar = calendar
        self.freq = freq
        self.validation = validation
        self.normalize = normalize
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self._model_options = model_options

    def get_params(self, deep=True):
        """Return constructor parameters, independently of fitted state."""
        return {**{name: getattr(self, name) for name in self._PARAMETERS}, **self._model_options}

    def set_params(self, **params):
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"Unknown estimator parameters: {sorted(unknown)}")
        for name, value in params.items():
            if name in self._PARAMETERS:
                setattr(self, name, value)
            else:
                self._model_options[name] = value
        self._clear_fitted()
        return self

    def _clear_fitted(self):
        for name in tuple(vars(self)):
            if name.endswith("_"):
                delattr(self, name)

    def _require_fitted(self):
        if not hasattr(self, "model_"):
            raise NotFittedError("Call fit before predict, evaluate, or save")

    def _configuration(self, horizon=None, target_dim=None, num_labels=None):
        task = TaskType.normalize(self.task)
        if not isinstance(self.model, str):
            if self.config is not None or self.task_config is not None or self._model_options:
                raise ValueError("Config arguments cannot be used with an instantiated model")
            if TaskType.normalize(self.model.task_name) != task:
                raise ValueError("The supplied model task does not match the estimator task")
            return self.model.config, self.model.task_config
        config = copy.deepcopy(self.config) if self.config is not None else AutoConfig.for_model(self.model)
        if isinstance(config, dict):
            values = config
            config = AutoConfig.for_model(self.model)
            config.update(values)
        if config.model_type != self.model:
            raise ValueError("model and config.model_type must agree")
        config_class = resolve("task", task.value).metadata["config_class"]
        task_fields = {field.name for field in fields(config_class)} - {"task"}
        options = dict(self._model_options)
        task_options = {key: options.pop(key) for key in tuple(options) if key in task_fields}
        known = set(config.to_dict()) | set(config._attribute_mapping())
        unknown = set(options) - known
        if unknown:
            raise ValueError(f"Unknown options for {config.model_type}: {sorted(unknown)}")
        config.update(options)
        if self.task_config is not None and task_options:
            raise ValueError("Pass task_config or task options, not both")
        task_config = self.task_config or config_class(**task_options)
        if task_config.task != task:
            raise ValueError("task_config.task must match task")
        inferred = {}
        for name, value in (("output_chunk_length", horizon), ("target_dim", target_dim), ("num_labels", num_labels)):
            if name in task_fields and value is not None:
                # An explicit class count may include classes absent in this split.
                if name == "num_labels" and (self.task_config is not None or name in task_options):
                    if value > task_config.num_labels:
                        raise ValueError("Training labels exceed configured num_labels")
                    continue
                inferred[name] = value
        return config, replace(task_config, **inferred)

    def _training_targets(self, inputs, targets=None):
        """Apply the task's target contract to arrays and dataset batches alike."""
        task = TaskType.normalize(self.task)
        batch = TimeSeriesBatch.from_inputs(inputs)
        targets = resolve("task", task.value).factory.resolve_targets(batch, targets)
        if targets is None:
            raise ValueError(f"{task.value} training requires targets")
        target = tf.convert_to_tensor(targets)
        if task == TaskType.CLASSIFICATION:
            if target.shape.rank != 1:
                raise ValueError("Classification targets must have shape (B,) with nonnegative integer class ids")
            tf.debugging.assert_all_finite(tf.cast(target, tf.float64), "Class ids must be finite")
            tf.debugging.assert_non_negative(target, message="Class ids must be nonnegative")
            if target.dtype.is_floating:
                tf.debugging.assert_equal(target, tf.floor(target), message="Class ids must be integers")
            target = tf.cast(target, tf.int64)
        else:
            if target.shape.rank != batch.past_values.shape.rank:
                raise ValueError("Regression/reconstruction targets must have the same rank as past_values")
            if task != TaskType.FORECASTING:
                tf.debugging.assert_equal(
                    tf.shape(target)[1],
                    tf.shape(batch.past_values)[1],
                    message=f"{task.value} requires one target timestep per input timestep",
                )
        tf.debugging.assert_positive(tf.shape(target), message="Targets must be nonempty")
        tf.debugging.assert_equal(
            tf.shape(target)[0],
            tf.shape(batch.past_values)[0],
            message="Inputs and targets must have the same sample count",
        )
        return target

    def _configuration_for_targets(self, target, *, dataset=False):
        if TaskType.normalize(self.task) == TaskType.CLASSIFICATION:
            if (
                dataset
                and isinstance(self.model, str)
                and self.task_config is None
                and "num_labels" not in self._model_options
            ):
                raise ValueError(
                    "Dataset classification requires explicit num_labels or task_config; "
                    "a batch may not contain every class"
                )
            return self._configuration(num_labels=max(2, int(tf.reduce_max(target)) + 1))
        return self._configuration(horizon=int(tf.shape(target)[1]), target_dim=int(tf.shape(target)[-1]))

    def fit(self, inputs, targets=None, *, validation_data=None, sample_weight=None, **fit_options):
        """Fit from scratch and return self; Keras History is available as ``history_``."""
        if "loss_fn" in fit_options or "metrics" in fit_options:
            raise ValueError(
                "Task training uses objective, not compiled loss_fn/metrics. Set objective on the estimator"
            )
        allowed = set(inspect.signature(Trainer.train).parameters) - {
            "self",
            "kwargs",
            "train_dataset",
            "valid_dataset",
        }
        if set(fit_options) - allowed:
            raise ValueError(f"Unknown fit options: {sorted(set(fit_options) - allowed)}")
        if self.batch_size < 1 or self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("batch_size, epochs, and learning_rate must be positive")
        self._clear_fitted()
        tf.keras.utils.set_random_seed(self.seed)
        task = TaskType.normalize(self.task)
        batch_size = fit_options.pop("batch_size", self.batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        from .sources import SourceOptions, source_for

        options = SourceOptions(
            task,
            batch_size,
            self.seed,
            self.schema,
            self.context,
            self.horizon,
            self.normalize,
            self.calendar,
            self.freq,
            self.validation,
            self._configuration,
            self._configuration_for_targets,
            self._training_targets,
        )
        data = source_for(inputs).prepare(inputs, targets, sample_weight, validation_data, options)
        train, valid = data.train, data.valid
        config, task_config = data.config, data.task_config
        model = (
            AutoModel.from_config(config, task=task, task_config=task_config)
            if isinstance(self.model, str)
            else self.model
        )
        if self.objective is not None:
            model.configure_objective(self.objective)
        first = next(iter(train)) if isinstance(train, tf.data.Dataset) else train[0]
        first_x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(first)
        model(first_x, training=False)
        # Validate labels, masks, objective, and output widths before starting fit.
        model.dry_run(first)
        trainer = Trainer(model)
        patience = fit_options.pop("early_stopping_patience", self.early_stopping_patience)
        history = trainer.train(
            train,
            valid_dataset=valid,
            optimizer=fit_options.pop("optimizer", tf.keras.optimizers.Adam(self.learning_rate)),
            epochs=fit_options.pop("epochs", self.epochs),
            batch_size=batch_size,
            early_stopping_patience=patience if valid is not None else None,
            **fit_options,
        )
        self.model_, self.trainer_, self.history_ = model, trainer, history
        self.preprocessor_, self.input_kind_, self.fit_report_ = data.preprocessor, data.input_kind, data.report
        self.batch_size_ = batch_size
        self.input_shapes_ = {
            key: list(value.shape[1:])
            for key, value in TimeSeriesBatch.from_inputs(first_x).as_tensor_dict().items()
            if key not in {"future_values", "future_observed_mask", "labels", "sample_weight"}
        }
        self.context_ = int(TimeSeriesBatch.from_inputs(first_x).past_values.shape[1])
        self.horizon_ = model.output_chunk_length
        return self

    def _prediction_data(self, inputs, future=None):
        self._require_fitted()
        if self.input_kind_ == "frame":
            prepared, windows, index = self.preprocessor_.prediction(inputs, future)
            return self.preprocessor_.dataset(prepared, windows, batch_size=self.batch_size_), index
        if future is not None:
            raise ValueError("future is only accepted for dataframe prediction")
        if isinstance(inputs, (tf.data.Dataset, tf.keras.utils.Sequence)):
            if self.preprocessor_ is not None:
                raise ValueError("Predict with prepared arrays when fitted from arrays")
            return inputs, None
        arrays = input_arrays(inputs)
        if self.preprocessor_ is not None:
            return self.preprocessor_.dataset(arrays, batch_size=self.batch_size_), None
        return tf.data.Dataset.from_tensor_slices(arrays).batch(self.batch_size_), None

    def predict_output(self, inputs, *, future=None):
        """Return the typed task output in original units."""
        dataset, _ = self._prediction_data(inputs, future)
        return self._predict_dataset(dataset)

    def _predict_dataset(self, dataset):
        outputs = []
        for data in dataset:
            x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
            outputs.append(self.model_(x, training=False, compute_loss=False))
        if not outputs:
            raise ValueError("Prediction data is empty")
        first = outputs[0]
        combined = {}
        for field in fields(first):
            value = getattr(first, field.name)
            if field.name in {"loss", "backbone_output"} or value is None:
                continue
            if tf.is_tensor(value):
                combined[field.name] = (
                    tf.concat([getattr(output, field.name) for output in outputs], axis=0)
                    if value.shape.rank
                    else value
                )
            elif isinstance(value, dict):
                combined[field.name] = {
                    key: tf.concat([getattr(output, field.name)[key] for output in outputs], axis=0) for key in value
                }
            else:
                combined[field.name] = value
        result = type(first)(**combined)
        return self.preprocessor_.inverse_output(result) if self.preprocessor_ is not None else result

    def predict(self, inputs, *, future=None, steps=None):
        """Return a forecast dataframe or a NumPy task prediction."""
        self._require_fitted()
        if steps is not None and (steps < 1 or steps > self.horizon_):
            raise ValueError(
                "steps must be between 1 and the fitted horizon; use model_.generate for recursive forecasts"
            )
        dataset, index = self._prediction_data(inputs, future)
        output = self._predict_dataset(dataset)
        if self.input_kind_ == "frame":
            return self.preprocessor_.to_frame(output, index, steps)

        values = np.asarray(output.prediction())
        return values[:, :steps] if steps is not None else values

    def evaluate(self, inputs, targets=None, *, sample_weight=None, **kwargs):
        self._require_fitted()
        if self.input_kind_ == "frame":
            from tfts.data.windows import WindowIndexer, WindowSpec

            prepared = self.preprocessor_.transform(inputs)
            windows = index_frame(prepared, WindowSpec(self.context_, self.horizon_, require_regular=False))
            dataset = self.preprocessor_.dataset(prepared, windows, batch_size=self.batch_size_)
        elif self.preprocessor_ is not None:
            arrays = input_arrays(inputs)
            target, weight = target_arrays(self._training_targets(arrays, targets), sample_weight)
            dataset = self.preprocessor_.dataset(arrays, target, weight, batch_size=self.batch_size_)
        else:
            dataset = inputs
        if getattr(self.model_, "optimizer", None) is None:
            self.model_.compile(optimizer="adam")
        return self.model_.evaluate(dataset, return_dict=True, **kwargs)

    def save(self, path):
        """Save model, fitted preprocessing, and a versioned manifest together."""
        self._require_fitted()
        params = self.get_params()
        params["model"] = self.model_.config.model_type
        params["objective"] = None if self.model_._objective_spec is None else self.model_._objective_spec.to_dict()
        params["config"] = (
            None if self.config is None else self.config.to_dict() if hasattr(self.config, "to_dict") else self.config
        )
        params["task_config"] = None if self.task_config is None else self.task_config.to_dict()
        params["schema"] = None if self.schema is None else self.schema.to_dict()
        state = {
            "params": params,
            "input_kind": self.input_kind_,
            "batch_size": self.batch_size_,
            "context": self.context_,
            "horizon": self.horizon_,
            "input_shapes": self.input_shapes_,
            "preprocessing": None if self.preprocessor_ is None else self.preprocessor_.to_dict(),
        }
        payload = json.dumps(state, indent=2, sort_keys=True)
        import tfts

        manifest = {
            "format_version": 2,
            "tfts_version": tfts.__version__,
            "tensorflow_version": tf.__version__,
            "keras_version": getattr(tf.keras, "__version__", None),
            "state_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model_.save(str(path / "model.keras"))
        (path / "estimator.json").write_text(payload, encoding="utf-8")
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        version = manifest.get("format_version")
        if isinstance(version, int) and version > 2:
            raise ValueError("Newer estimator format version; upgrade TFTS to load this artifact")
        if version != 2:
            raise ValueError("Unsupported estimator format version; this restructure requires format 2")
        payload = (path / "estimator.json").read_text(encoding="utf-8")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != manifest.get("state_sha256"):
            raise ValueError("Estimator state does not match its manifest")
        state = json.loads(payload)
        params = state["params"]
        if params["schema"] is not None:
            params["schema"] = Schema.from_dict(params["schema"])
        if params["task_config"] is not None:
            from tfts.tasks.auto import task_config_from_dict

            params["task_config"] = task_config_from_dict(params["task_config"])
        result = cls(**params)
        result.model_ = tf.keras.models.load_model(str(path / "model.keras"), compile=False)
        result.trainer_ = Trainer(result.model_)
        result.input_kind_ = state["input_kind"]
        result.preprocessor_ = None
        if state["preprocessing"] is not None:
            processor = FramePipeline if result.input_kind_ == "frame" else ArrayPreprocessor
            result.preprocessor_ = processor.from_dict(state["preprocessing"])
        result.batch_size_, result.context_, result.horizon_ = state["batch_size"], state["context"], state["horizon"]
        result.input_shapes_, result.fit_report_, result.history_ = state["input_shapes"], {}, None
        return result


class Forecaster(Estimator):
    """Forecasting convenience constructor; all behavior lives in Estimator."""

    def __init__(self, model="rnn", *, task="forecasting", **kwargs):
        if TaskType.normalize(task) != TaskType.FORECASTING:
            raise ValueError("Forecaster requires task='forecasting'; use Estimator for other tasks")
        super().__init__(task=task, model=model, **kwargs)
