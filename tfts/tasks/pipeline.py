"""Deprecated argument adapters for the single estimator implementation."""

import json
from pathlib import Path
import warnings

import numpy as np
import tensorflow as tf

from tfts.contracts import TaskType
from tfts.data.preprocessing import ArrayPreprocessor, input_arrays
from tfts.data.scaling import ChannelScaler
from tfts.estimator import Estimator
from tfts.models.auto_model import AutoModel
from tfts.training import Trainer


class TaskPipeline:
    """Compatibility wrapper. New code should use :class:`tfts.Estimator`."""

    def __init__(
        self,
        task,
        model="rnn",
        *,
        config=None,
        task_config=None,
        processor=None,
        batch_size=32,
        normalize="standard",
        learning_rate=1e-3,
        epochs=50,
        early_stopping_patience=5,
        seed=42,
        **task_kwargs,
    ):
        warnings.warn(
            "TaskPipeline is deprecated; use tfts.Estimator. "
            "It will be removed no earlier than the next minor release.",
            FutureWarning,
            stacklevel=2,
        )
        self._estimator = Estimator(
            task,
            model,
            config=config,
            task_config=task_config,
            batch_size=batch_size,
            normalize=normalize,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
            **task_kwargs,
        )
        self.processor = processor

    @property
    def task(self):
        return TaskType.normalize(self._estimator.task)

    @property
    def model(self):
        if not hasattr(self._estimator, "model_"):
            config, task_config = self._estimator._configuration(
                horizon=self._estimator.horizon if self.task == TaskType.FORECASTING else None
            )
            self._estimator.model_ = (
                AutoModel.from_config(config, task=self.task, task_config=task_config)
                if isinstance(self._estimator.model, str)
                else self._estimator.model
            )
        return self._estimator.model_

    @property
    def config(self):
        return self.model.config

    @property
    def trainer(self):
        return getattr(self._estimator, "trainer_", None) or Trainer(self.model)

    @property
    def _fitted(self):
        return hasattr(self._estimator, "history_")

    @property
    def _scalers(self):
        return self._estimator.preprocessor_.scalers

    @property
    def _target_scaler(self):
        return getattr(getattr(self._estimator, "preprocessor_", None), "target_scaler", None)

    @property
    def _input_shapes(self):
        return getattr(self._estimator, "input_shapes_", None)

    @property
    def batch_size(self):
        return self._estimator.batch_size

    def preprocess(self, inputs):
        processor = getattr(self._estimator, "preprocessor_", None) or self.processor
        if processor is None:
            return inputs
        if isinstance(processor, ArrayPreprocessor):
            return processor.transform(input_arrays(inputs))
        return processor.transform(inputs) if hasattr(processor, "transform") else processor(inputs)

    def forward(self, model_inputs, generation_config=None, **kwargs):
        if generation_config is not None:
            if self.task != TaskType.FORECASTING:
                raise ValueError("generation_config is only valid for forecasting")
            return self.model.generate(model_inputs, generation_config, **kwargs)
        if self.task == TaskType.ANOMALY_DETECTION and kwargs.pop("detect", False):
            return self.model.detect(model_inputs)
        return self.model(model_inputs, training=False)

    def postprocess(self, output):
        processor = getattr(self._estimator, "preprocessor_", None)
        if processor is not None:
            return processor.inverse_output(output)
        if self.processor is not None and hasattr(self.processor, "inverse_transform"):
            for field in ("predictions", "imputed_values"):
                value = getattr(output, field, None)
                if value is not None:
                    return output.replace(**{field: self.processor.inverse_transform(value)})
        return output

    def __call__(self, inputs, generation_config=None, **kwargs):
        return self.postprocess(self.forward(self.preprocess(inputs), generation_config=generation_config, **kwargs))

    def fit(self, inputs, targets=None, *, validation_data=None, sample_weight=None, valid_dataset=None, **kwargs):
        if valid_dataset is not None:
            if validation_data is not None:
                raise ValueError("Pass validation data only once")
            validation_data = valid_dataset
        if isinstance(inputs, (tf.data.Dataset, tf.keras.utils.Sequence)):
            if targets is not None:
                if validation_data is not None:
                    raise ValueError("Pass validation data only once")
                validation_data, targets = targets, None
        elif isinstance(inputs, (tuple, list)) and targets is None:
            inputs, targets = inputs
        else:
            if self.processor is not None:
                raise ValueError("Use dataset training with a custom processor")
            if validation_data is None:
                raise ValueError("Provide validation_data=(X_valid, y_valid[, sample_weight])")
            if self._fitted:
                raise ValueError("Create a new pipeline to fit new data; use Estimator for refitting")
        self._estimator.fit(inputs, targets, validation_data=validation_data, sample_weight=sample_weight, **kwargs)
        return self._estimator.history_

    def predict(self, inputs):
        return self._estimator.predict(inputs)

    def save(self, path):
        if self.processor is not None:
            raise ValueError("Save custom preprocessing separately, or use Estimator's fitted preprocessing")
        return self._estimator.save(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if (path / "manifest.json").exists():
            estimator = Estimator.load(path)
        else:
            # One migration boundary for the historical prepared-array artifact.
            state = json.loads((path / "pipeline.json").read_text(encoding="utf-8"))
            if state.get("version") != 1:
                raise ValueError("Unsupported pipeline version")
            model = tf.keras.models.load_model(str(path / "model.keras"), compile=False)
            estimator = Estimator(state["task"], model.config.model_type, batch_size=state["batch_size"])
            estimator.model_, estimator.trainer_, estimator.history_ = model, Trainer(model), None
            processor = ArrayPreprocessor()
            processor.shapes = state["input_shapes"]
            processor.scalers = {key: ChannelScaler.from_dict(value) for key, value in state["scalers"].items()}
            processor.target_scaler = ChannelScaler.from_dict(state["target_scaler"])
            estimator.preprocessor_, estimator.input_kind_ = processor, "array"
            estimator.input_shapes_, estimator.batch_size_ = state["input_shapes"], state["batch_size"]
            estimator.context_, estimator.horizon_, estimator.fit_report_ = (
                state["input_shapes"]["past_values"][0],
                model.output_chunk_length,
                {},
            )
        result = cls.__new__(cls)
        result._estimator, result.processor = estimator, None
        return result

    def calibrate(self, inputs):
        if self.task != TaskType.ANOMALY_DETECTION:
            raise ValueError("calibrate() is only valid for anomaly detection")
        return self.model.calibrate(self.preprocess(inputs))


Pipeline = TaskPipeline
