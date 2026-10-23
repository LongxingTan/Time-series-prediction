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

from tfts.contracts import TaskType, TimeSeriesBatch
from tfts.data.frame import FramePreprocessor
from tfts.data.preprocessing import ArrayPreprocessor, input_arrays, target_arrays
from tfts.features import TimeSeriesSchema
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import AutoModel
from tfts.models.registry import get_model_capabilities
from tfts.registry import resolve
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
        if "bidirectional" in options:
            if "bi_direction" in options:
                raise ValueError("Pass only one direction option")
            options["bi_direction"] = options.pop("bidirectional")
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

    def fit(self, inputs, targets=None, *, validation_data=None, sample_weight=None, **fit_options):
        """Fit from scratch and return self; Keras History is available as history_."""
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
        preprocessor, report = None, {}
        if isinstance(inputs, pd.DataFrame):
            if task != TaskType.FORECASTING:
                raise ValueError("Dataframe fitting is for forecasting; use arrays or a dataset for other tasks")
            if targets is not None or sample_weight is not None or validation_data is not None:
                raise ValueError("Declare dataframe targets in schema and use validation for the temporal split")
            if self.schema is None:
                raise ValueError("Dataframe fitting requires TimeSeriesSchema(time_col=..., target_cols=...)")
            if self.context < 1 or self.horizon < 1:
                raise ValueError("context and horizon must be positive")
            config, task_config = self._configuration(self.horizon, len(self.schema.target_cols))
            capabilities = get_model_capabilities(config.model_type)
            preprocessor = FramePreprocessor(
                self.schema, self.context, self.horizon, self.normalize, self.calendar, self.freq
            )
            prepared, train_windows, valid_windows = preprocessor.fit(inputs, self.validation, capabilities.input_spec)
            train = preprocessor.dataset(prepared, train_windows, batch_size=batch_size, shuffle=True, seed=self.seed)
            valid = (
                None if valid_windows is None else preprocessor.dataset(prepared, valid_windows, batch_size=batch_size)
            )
            report = preprocessor.fit_report
            input_kind = "frame"
        elif isinstance(inputs, (tf.data.Dataset, tf.keras.utils.Sequence)):
            if targets is not None or sample_weight is not None:
                raise ValueError("Include targets and weights in the dataset batches")
            train, valid = inputs, validation_data
            first = next(iter(train)) if isinstance(train, tf.data.Dataset) else train[0]
            x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(first)
            batch = TimeSeriesBatch.from_inputs(x)
            y = y if y is not None else (batch.future_values if task == TaskType.FORECASTING else batch.labels)
            y = None if y is None else tf.convert_to_tensor(y)
            horizon = int(y.shape[1]) if y is not None and y.shape.rank == 3 else self.horizon
            width = int(y.shape[-1]) if y is not None and y.shape.rank == 3 else None
            config, task_config = self._configuration(horizon, width)
            input_kind = "dataset"
        else:
            x = input_arrays(inputs)
            if targets is None:
                if task == TaskType.ANOMALY_DETECTION:
                    targets = x["past_values"]
                else:
                    raise ValueError("Prepared-array training requires targets")
            y, weight = target_arrays(targets, sample_weight)
            classification = task == TaskType.CLASSIFICATION
            if classification:
                if y.ndim != 1 or not np.equal(y, np.floor(y)).all() or (y < 0).any():
                    raise ValueError("Classification targets must have shape (B,) with nonnegative integer class ids")
                y = y.astype(np.int32)
            elif y.ndim != 3:
                raise ValueError("Regression/reconstruction targets must have shape (B,T,D)")
            if len(y) != len(x["past_values"]):
                raise ValueError("Inputs and targets must have the same sample count")
            if (
                task in {TaskType.SEQUENCE_REGRESSION, TaskType.IMPUTATION, TaskType.ANOMALY_DETECTION}
                and x["past_values"].shape[1] != y.shape[1]
            ):
                raise ValueError(f"{task.value} requires one target timestep per input timestep")
            config, task_config = self._configuration(
                None if classification else y.shape[1],
                None if classification else y.shape[-1],
                max(2, int(y.max()) + 1) if classification else None,
            )
            spec = get_model_capabilities(config.model_type).input_spec
            preprocessor = ArrayPreprocessor(self.normalize).fit(
                x,
                y,
                weight,
                scale_target=not classification,
                target_history=spec.requires_target_history
                or task in {TaskType.IMPUTATION, TaskType.ANOMALY_DETECTION},
                matching_time_features=spec.requires_matching_time_features,
            )
            train = preprocessor.dataset(x, y, weight, batch_size=batch_size, shuffle=True, seed=self.seed)
            valid = None
            if validation_data is not None:
                if not isinstance(validation_data, (tuple, list)) or len(validation_data) not in (2, 3):
                    raise ValueError("Provide validation_data=(X_valid, y_valid[, sample_weight])")
                vx = input_arrays(validation_data[0])
                vy, vw = target_arrays(*validation_data[1:])
                if vy.shape[1:] != y.shape[1:]:
                    raise ValueError("Training and validation target shapes must match")
                valid = preprocessor.dataset(vx, vy, vw, batch_size=batch_size)
            input_kind = "array"
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
        batch = model._batch_from_keras(first)
        model._objective_result(batch, model(batch, training=False, compute_loss=False))
        trainer = Trainer(model)
        history = trainer.train(
            train,
            valid_dataset=valid,
            optimizer=fit_options.pop("optimizer", tf.keras.optimizers.Adam(self.learning_rate)),
            epochs=fit_options.pop("epochs", self.epochs),
            batch_size=batch_size,
            early_stopping_patience=(
                fit_options.pop("early_stopping_patience", self.early_stopping_patience) if valid is not None else None
            ),
            **fit_options,
        )
        self.model_, self.trainer_, self.history_ = model, trainer, history
        self.preprocessor_, self.input_kind_, self.fit_report_ = preprocessor, input_kind, report
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
        from tfts.training.trainer import _output_tensor

        values = np.asarray(_output_tensor(output))
        return values[:, :steps] if steps is not None else values

    def evaluate(self, inputs, targets=None, *, sample_weight=None, **kwargs):
        self._require_fitted()
        if self.input_kind_ == "frame":
            from tfts.data.windowing import WindowIndexer, WindowSpec

            frame = self.preprocessor_._validate(inputs)
            self.preprocessor_._check_time(frame)
            prepared = self.preprocessor_._encode(self.preprocessor_._calendar_frame(frame))
            windows = WindowIndexer().build(prepared, WindowSpec(self.context_, self.horizon_))
            dataset = self.preprocessor_.dataset(prepared, windows, batch_size=self.batch_size_)
        elif self.preprocessor_ is not None:
            target, weight = target_arrays(targets, sample_weight)
            dataset = self.preprocessor_.dataset(input_arrays(inputs), target, weight, batch_size=self.batch_size_)
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
            "format_version": 1,
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
        if manifest.get("format_version") != 1:
            raise ValueError("Unsupported estimator format version; upgrade TFTS to load this artifact")
        payload = (path / "estimator.json").read_text(encoding="utf-8")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != manifest.get("state_sha256"):
            raise ValueError("Estimator state does not match its manifest")
        state = json.loads(payload)
        params = state["params"]
        if params["schema"] is not None:
            params["schema"] = TimeSeriesSchema.from_dict(params["schema"])
        if params["task_config"] is not None:
            from tfts.models.auto_model import task_config_from_dict

            params["task_config"] = task_config_from_dict(params["task_config"])
        result = cls(**params)
        result.model_ = tf.keras.models.load_model(str(path / "model.keras"), compile=False)
        result.trainer_ = Trainer(result.model_)
        result.input_kind_ = state["input_kind"]
        result.preprocessor_ = None
        if state["preprocessing"] is not None:
            processor = FramePreprocessor if result.input_kind_ == "frame" else ArrayPreprocessor
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
