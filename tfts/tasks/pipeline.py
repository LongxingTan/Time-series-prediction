"""Task-neutral inference and training pipeline."""

import copy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf

from tfts.contracts import TaskType
from tfts.data.scaling import ChannelScaler
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import AutoModel
from tfts.training import Trainer, set_seed


class TaskPipeline:
    """Compose preprocessing, a task model, and postprocessing.

    The processor is intentionally a boundary object. It may be a callable or
    expose ``transform``/``inverse_transform``; model and task layers never
    import it.
    """

    def __init__(
        self,
        task,
        model="rnn",
        *,
        config=None,
        task_config=None,
        processor: Optional[Callable] = None,
        batch_size=32,
        normalize="standard",
        learning_rate=1e-3,
        epochs=50,
        early_stopping_patience=5,
        seed=42,
        **task_kwargs,
    ):
        self.task = TaskType.normalize(task)
        if batch_size < 1 or epochs < 1 or learning_rate <= 0:
            raise ValueError("batch_size, epochs, and learning_rate must be positive")
        ChannelScaler(normalize)  # Validate the requested scaling method.
        self.processor = processor
        self.batch_size = batch_size
        self.normalize = normalize
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        set_seed(seed)
        self._fitted = False
        self._model_supplied = not isinstance(model, str)
        self._scalers = {}
        self._target_scaler = None
        self._input_shapes = None
        self.config = (
            copy.deepcopy(config)
            if config is not None
            else (AutoConfig.for_model(model) if isinstance(model, str) else model.config)
        )
        if isinstance(model, str):
            config = self.config
            self.model = AutoModel.from_config(config, task=self.task, task_config=task_config, **task_kwargs)
        else:
            if config is not None or task_config is not None or task_kwargs:
                raise ValueError("Config arguments cannot be used with an instantiated model")
            self.model = model
        self.trainer = Trainer(self.model)

    def preprocess(self, inputs):
        if self._target_scaler is not None:
            return self._transform_inputs(self._arrays(inputs))
        if self.processor is None:
            return inputs
        if hasattr(self.processor, "transform"):
            return self.processor.transform(inputs)
        return self.processor(inputs)

    def forward(self, model_inputs, generation_config=None, **kwargs):
        if generation_config is not None:
            if self.task != TaskType.FORECASTING:
                raise ValueError("generation_config is only valid for forecasting")
            return self.model.generate(model_inputs, generation_config, **kwargs)
        if self.task == TaskType.ANOMALY_DETECTION and kwargs.pop("detect", False):
            return self.model.detect(model_inputs)
        return self.model(model_inputs, training=False)

    def postprocess(self, output):
        if self._target_scaler is not None:
            return output.replace(predictions=self._target_scaler.inverse_transform(output.predictions))
        if self.processor is None or not hasattr(self.processor, "inverse_transform"):
            return output
        if hasattr(output, "predictions") and output.predictions is not None:
            return output.replace(predictions=self.processor.inverse_transform(output.predictions))
        elif hasattr(output, "imputed_values") and output.imputed_values is not None:
            return output.replace(imputed_values=self.processor.inverse_transform(output.imputed_values))
        return output

    def __call__(self, inputs, generation_config=None, **kwargs):
        model_inputs = self.preprocess(inputs)
        output = self.forward(model_inputs, generation_config=generation_config, **kwargs)
        return self.postprocess(output)

    def fit(self, inputs, targets=None, *, validation_data=None, sample_weight=None, valid_dataset=None, **kwargs):
        """Fit prepared (B,T,C) arrays, or pass a dataset to the existing trainer.

        Array training requires an explicit validation set; windows are never
        randomly split because they may overlap or belong to the same group.
        """
        if isinstance(inputs, (tf.data.Dataset, tf.keras.utils.Sequence, tuple, list)):
            validation = [value for value in (targets, valid_dataset, validation_data) if value is not None]
            if len(validation) > 1:
                raise ValueError("Pass validation data only once")
            if sample_weight is not None:
                raise ValueError("Include sample weights in the dataset's (inputs, targets, weights) batches")
            return self.trainer.train(inputs, valid_dataset=validation[0] if validation else None, **kwargs)
        if targets is None:
            raise ValueError("Prepared-array training requires targets")
        if self.task not in {TaskType.FORECASTING, TaskType.SEQUENCE_REGRESSION}:
            raise ValueError("Array fit is available for forecasting and sequence_regression")
        if self.processor is not None:
            raise ValueError("Array fit manages scaling; use dataset training with a custom processor")
        if validation_data is None or len(validation_data) not in (2, 3):
            raise ValueError("Provide validation_data=(X_valid, y_valid[, sample_weight])")
        x, y, weight = self._arrays(inputs, targets, sample_weight)
        vx, vy, vw = self._arrays(*validation_data)
        if set(x) != set(vx) or any(x[key].shape[1:] != vx[key].shape[1:] for key in x):
            raise ValueError("Training and validation inputs must have identical fields and window shapes")
        if y.shape[1:] != vy.shape[1:]:
            raise ValueError("Training and validation target shapes must match")
        if self.task == TaskType.SEQUENCE_REGRESSION and x["past_values"].shape[1] != y.shape[1]:
            raise ValueError("sequence_regression requires one target timestep per input timestep")
        if self._fitted:
            raise ValueError("Create a new pipeline to fit new data; fitted preprocessing is immutable")
        set_seed(self.seed)
        if self._model_supplied:
            raise ValueError(
                "Use dataset training for an instantiated model; array fit infers a new model's dimensions"
            )
        config = copy.deepcopy(self.config)
        config.target_dim = y.shape[-1]
        if config.model_type == "dlinear":
            config.channels = x["past_values"].shape[-1]
        self.model = AutoModel.from_config(
            config,
            task=self.task,
            task_config=replace(self.model.task_config, output_chunk_length=y.shape[1], target_dim=y.shape[-1]),
        )
        self.config = self.model.config
        self._input_shapes = {key: list(value.shape[1:]) for key, value in x.items()}
        # Validate architecture compatibility before fitting transforms or training.
        self.model({key: value[:1] for key, value in x.items()}, training=False)
        self._scalers = {key: ChannelScaler(self.normalize).fit(value) for key, value in x.items()}
        self._target_scaler = ChannelScaler(self.normalize).fit(y, weight)
        if config.model_type == "tide" and "past_time_features" in x:
            covariates = np.concatenate([x["past_time_features"], x["future_time_features"]], axis=1)
            scaler = ChannelScaler(self.normalize).fit(covariates)
            self._scalers.update(past_time_features=scaler, future_time_features=scaler)
        if self.model.capabilities.input_spec.requires_target_history:
            # Native target-history models must see history and outputs in the same units.
            self._scalers["past_values"] = self._target_scaler
        self.batch_size = kwargs.pop("batch_size", self.batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        train = self._dataset(x, y, weight, shuffle=True)
        valid = self._dataset(vx, vy, vw)
        self.trainer = Trainer(self.model)
        history = self.trainer.train(
            train,
            valid_dataset=valid,
            optimizer=kwargs.pop("optimizer", tf.keras.optimizers.Adam(self.learning_rate)),
            epochs=kwargs.pop("epochs", self.epochs),
            early_stopping_patience=kwargs.pop("early_stopping_patience", self.early_stopping_patience),
            **kwargs,
        )
        self._fitted = True
        return history

    @staticmethod
    def _arrays(inputs, targets=None, sample_weight=None):
        inputs = inputs if isinstance(inputs, dict) else {"past_values": inputs}
        allowed = {"past_values", "past_time_features", "future_time_features"}
        if "past_values" not in inputs or set(inputs) - allowed:
            raise ValueError("Prepared inputs require past_values and optional past_time_features/future_time_features")
        arrays = {key: np.asarray(value, dtype=np.float32) for key, value in inputs.items()}
        for key, value in arrays.items():
            if value.ndim != 3 or not all(value.shape) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a nonempty finite array with shape (samples, time, channels)")
            if len(value) != len(arrays["past_values"]):
                raise ValueError("All input fields must have the same sample count")
        if targets is None:
            return arrays
        target = np.asarray(targets, dtype=np.float32)
        if target.ndim != 3 or not all(target.shape) or not np.isfinite(target).all():
            raise ValueError("Targets must be finite with shape (samples, time, targets)")
        if len(target) != len(arrays["past_values"]):
            raise ValueError("Inputs and targets must have the same sample count")
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        if weight is not None:
            if weight.ndim not in (1, 2, 3) or weight.shape != target.shape[: weight.ndim]:
                raise ValueError("sample_weight must have shape (B,), (B,T), or (B,T,D) matching targets")
            if not np.isfinite(weight).all() or (weight < 0).any() or not (weight > 0).any():
                raise ValueError("sample_weight must be finite, nonnegative, and contain positive weights")
        return arrays, target, weight

    def _transform_inputs(self, inputs):
        if set(inputs) != set(self._input_shapes) or any(
            list(value.shape[1:]) != self._input_shapes[key] for key, value in inputs.items()
        ):
            raise ValueError("Prediction inputs must match the fitted fields and window shapes")
        return {key: self._scalers[key].transform(value) for key, value in inputs.items()}

    def _dataset(self, inputs, target, weight=None, shuffle=False):
        x = self._transform_inputs(inputs)
        y = self._target_scaler.transform(target)
        data = (x, y) if weight is None else (x, y, weight)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            dataset = dataset.shuffle(len(target), seed=self.seed)
        return dataset.batch(self.batch_size)

    def predict(self, inputs):
        """Return predictions in the original target units."""
        if not self._fitted:
            raise RuntimeError("Call fit before predict")
        x = self._transform_inputs(self._arrays(inputs))
        dataset = tf.data.Dataset.from_tensor_slices(x).batch(self.batch_size)
        return self._target_scaler.inverse_transform(self.trainer.predict(dataset))

    def save(self, path):
        """Save model and training-fitted transforms together."""
        if not self._fitted or self._target_scaler is None:
            raise RuntimeError("Fit prepared arrays before saving this pipeline")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "model.keras"))
        state = dict(
            version=1,
            task=self.task.value,
            batch_size=self.batch_size,
            input_shapes=self._input_shapes,
            scalers={key: value.to_dict() for key, value in self._scalers.items()},
            target_scaler=self._target_scaler.to_dict(),
        )
        (path / "pipeline.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Restore a prepared-array pipeline without refitting transforms."""
        path = Path(path)
        state = json.loads((path / "pipeline.json").read_text(encoding="utf-8"))
        if state["version"] != 1:
            raise ValueError("Unsupported pipeline version")
        model = tf.keras.models.load_model(str(path / "model.keras"), compile=False)
        result = TaskPipeline(state["task"], model, batch_size=state["batch_size"])
        result._input_shapes = state["input_shapes"]
        result._scalers = {key: ChannelScaler.from_dict(value) for key, value in state["scalers"].items()}
        result._target_scaler = ChannelScaler.from_dict(state["target_scaler"])
        result._fitted = True
        return result

    def calibrate(self, inputs):
        if self.task != TaskType.ANOMALY_DETECTION:
            raise ValueError("calibrate() is only valid for anomaly detection")
        return self.model.calibrate(self.preprocess(inputs))


Pipeline = TaskPipeline
