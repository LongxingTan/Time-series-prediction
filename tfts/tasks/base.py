"""Base classes for task heads and task models."""

from abc import ABC, abstractmethod
from dataclasses import asdict
import json
import os
from typing import Any, Tuple

import tensorflow as tf

from tfts.contracts import ModelOutput, TimeSeriesBatch


class BaseHead(tf.keras.layers.Layer, ABC):
    """Trainable projection from a declared backbone output port."""

    @abstractmethod
    def call(self, inputs, **kwargs):
        raise NotImplementedError


class TimeSeriesTaskModel(tf.keras.Model, ABC):
    """Composition root for a backbone, task head, loss, and typed output."""

    task_name = None
    required_output_port = None

    def __init__(self, backbone, task_config, capabilities, **kwargs):
        super().__init__(**kwargs)
        from tfts.models.adapters import BackboneAdapter

        self.backbone = backbone
        self.backbone_config = backbone.config
        self.task_config = task_config
        self.capabilities = capabilities
        self.adapter = BackboneAdapter(backbone, capabilities)

    @property
    def config(self):
        return self.backbone_config

    @property
    def predict_sequence_length(self):
        return getattr(self.task_config, "prediction_length", 1)

    def __call__(self, inputs=None, *args, **kwargs):
        # Keras rejects non-tensor positional values before reaching ``call``.
        # A canonical batch is a supported public input, so route it by name.
        if isinstance(inputs, TimeSeriesBatch):
            return super().__call__(*args, inputs=inputs, **kwargs)
        return super().__call__(inputs, *args, **kwargs)

    def normalize_batch(self, inputs: Any) -> TimeSeriesBatch:
        batch = TimeSeriesBatch.from_inputs(inputs)
        batch.validate_for(self.task_name)
        return batch

    @abstractmethod
    def forward(self, inputs, training=None) -> ModelOutput:
        raise NotImplementedError

    @abstractmethod
    def primary_output(self, output: ModelOutput) -> tf.Tensor:
        raise NotImplementedError

    def call(self, inputs, training=None, return_dict=False):
        output = self.forward(inputs, training=training)
        return output if return_dict else self.primary_output(output)

    def build_from_config(self, config):
        """Build every child layer before Keras restores saved variables."""

        def make_dummy(shape):
            shape = tf.TensorShape(shape).as_list()
            dimensions = [dimension if dimension is not None else 1 for dimension in shape]
            return tf.zeros(dimensions, dtype=self.compute_dtype)

        def make_inputs(shape):
            if isinstance(shape, dict):
                return {key: make_inputs(value) for key, value in shape.items()}
            if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple, tf.TensorShape)):
                return [make_inputs(value) for value in shape]
            return make_dummy(shape)

        input_shape = config.get("input_shape")
        if input_shape is not None:
            self(make_inputs(input_shape))

    def get_config(self):
        """Return a Keras-serializable description of the task model.

        The task model is constructed from a backbone instance, a task
        dataclass, and registry metadata.  Those live Python objects cannot be
        passed directly through a Keras config, so persist their stable
        representations instead.  The child backbone and task head weights
        remain tracked by Keras and are restored from the same archive.
        """
        config = super().get_config()
        config.update(
            {
                "backbone_config": self.backbone_config.to_dict(),
                "task_config": self.task_config.to_dict(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a task model from a Keras object config."""
        from tfts.models.auto_config import AutoConfig
        from tfts.models.auto_model import build_task_model, task_config_from_dict

        config = dict(config)
        backbone_values = config.pop("backbone_config", None)
        task_values = config.pop("task_config", None)
        if not isinstance(backbone_values, dict) or not isinstance(task_values, dict):
            raise ValueError("Serialized task model config must contain backbone_config and task_config mappings")

        model_type = backbone_values.get("model_type")
        if model_type is None:
            raise ValueError("Serialized task model config is missing backbone model_type")

        backbone_config = AutoConfig.for_model(model_type)
        backbone_config.update(backbone_values)

        model = build_task_model(backbone_config, task_config_from_dict(task_values), model_kwargs=config)
        if not isinstance(model, cls):
            raise ValueError("Serialized task config does not match %s" % cls.__name__)
        return model

    def save_pretrained(self, save_directory):
        """Save one coherent architecture + task + weights artifact."""
        from tfts.constants import TF2_WEIGHTS_NAME

        if not self.built:
            raise ValueError("Model must be built before it can be saved")
        os.makedirs(save_directory, exist_ok=True)
        self.backbone_config.save_pretrained(save_directory)
        task_values = asdict(self.task_config)
        task_values["task"] = self.task_config.task.value
        with open(os.path.join(save_directory, "task_config.json"), "w", encoding="utf-8") as file:
            json.dump({"schema_version": 1, "task_config": task_values}, file, indent=2)
        self.save_weights(os.path.join(save_directory, TF2_WEIGHTS_NAME))

    @property
    @abstractmethod
    def default_loss(self):
        raise NotImplementedError

    @property
    def default_metrics(self) -> Tuple[Any, ...]:
        return ()


class BaseTask(ABC):
    """Semantic base for non-layer task services such as anomaly scorers."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


from tfts.contracts import ModelOutput as ModelOutput  # noqa: E402,F401
