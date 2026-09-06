"""Base classes for task heads and task models."""

from abc import ABC, abstractmethod
from dataclasses import asdict
import json
import os
from typing import Any

import tensorflow as tf

from tfts.contracts import ModelOutput, TimeSeriesBatch
from tfts.registry import build


class BaseHead(tf.keras.layers.Layer, ABC):
    """Trainable projection from a declared backbone output port."""

    @abstractmethod
    def call(self, inputs, **kwargs):
        raise NotImplementedError


class TimeSeriesTaskModel(tf.keras.Model, ABC):
    """Composition root for a backbone, task head, loss, and typed output."""

    task_name = None
    required_output_port = None
    default_objective = None

    def __init__(self, backbone, task_config, capabilities, **kwargs):
        super().__init__(**kwargs)
        from tfts.models.adapters import BackboneAdapter, SpatialAdapter

        self.backbone = backbone
        self.backbone_config = backbone.config
        self.task_config = task_config
        self.capabilities = capabilities
        self.adapter = BackboneAdapter(backbone, capabilities)
        spatial_strategy = getattr(task_config, "spatial_strategy", "raise")
        self.spatial_adapter = SpatialAdapter(capabilities.input_spec, self.adapter.model_type, spatial_strategy)
        self.objective = None
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def config(self):
        return self.backbone_config

    @property
    def predict_sequence_length(self):
        return self.output_chunk_length

    @property
    def output_chunk_length(self):
        return getattr(self.task_config, "output_chunk_length", 1)

    def __call__(self, inputs=None, *args, **kwargs):
        # Keras rejects non-tensor positional values before reaching ``call``.
        # A canonical batch is a supported public input, so route it by name.
        if isinstance(inputs, TimeSeriesBatch):
            return super().__call__(*args, inputs=inputs, **kwargs)
        return super().__call__(inputs, *args, **kwargs)

    def normalize_batch(self, inputs: Any) -> TimeSeriesBatch:
        from dataclasses import replace

        batch = TimeSeriesBatch.from_inputs(inputs)
        casts = {}
        for name in (
            "past_values",
            "future_values",
            "past_time_features",
            "future_time_features",
            "static_real_features",
        ):
            value = getattr(batch, name)
            if value is not None and value.dtype.is_floating and value.dtype != self.compute_dtype:
                casts[name] = tf.cast(value, self.compute_dtype)
        if casts:
            batch = replace(batch, **casts)
        batch.validate_for(self.task_name)
        if not hasattr(self, "_batch_build_specs"):
            shared_names = set()
            if batch.structure is not None:
                _, shared = batch.structure.split_tensor_dict()
                shared_names = set(shared)
            self._batch_build_specs = {}
            for name, value in batch.as_tensor_dict().items():
                shape = tuple(value.shape)
                shape = shape if name in shared_names else (None,) + shape[1:]
                spec = {"shape": shape, "dtype": value.dtype.name}
                if name == "structure.graph.num_nodes":
                    stored = tf.get_static_value(value)
                    spec["value"] = stored.tolist() if hasattr(stored, "tolist") else stored
                self._batch_build_specs[name] = spec
        return batch

    def prepare_backbone_batch(self, batch):
        """Validate direct spatial support or apply the configured fallback."""
        transformed, restore = self.spatial_adapter.to_backbone(batch)
        return transformed, lambda value: self.spatial_adapter.from_backbone(value, restore)

    @abstractmethod
    def forward(self, inputs, training=None) -> ModelOutput:
        raise NotImplementedError

    def configure_objective(self, objective=None, **context):
        component = self.default_objective if objective is None else objective
        self.objective = None if component is None else build("objective", component, **context)

    def call(self, inputs, training=None):
        output = self.forward(inputs, training=training)
        batch = self.normalize_batch(inputs)
        if self.objective is not None and self.has_targets(batch):
            output = output.replace(loss=self.objective(batch, output))
        return output

    def has_targets(self, batch):
        if self.task_name == "forecasting":
            return batch.future_values is not None
        if self.task_name == "classification":
            return batch.labels is not None
        if self.task_name == "imputation":
            return batch.labels is not None
        if self.task_name == "anomaly_detection":
            return batch.labels is not None
        return batch.labels is not None or batch.future_values is not None

    def _batch_from_keras(self, data):
        from dataclasses import replace

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported by batch-aware Objectives")
        batch = self.normalize_batch(x)
        if y is None:
            if self.task_name == "anomaly_detection" and batch.labels is None:
                batch = replace(batch, labels=batch.past_values)
            return batch
        field = "future_values" if self.task_name == "forecasting" else "labels"
        target = tf.convert_to_tensor(y)
        if target.dtype.is_floating:
            target = tf.cast(target, self.compute_dtype)
        return replace(batch, **{field: target})

    @property
    def metrics(self):
        return [self._loss_tracker]

    def train_step(self, data):
        batch = self._batch_from_keras(data)
        with tf.GradientTape() as tape:
            output = self(batch, training=True)
            if output.loss is None:
                raise ValueError(f"{self.task_name} training requires targets and an Objective")
            loss = output.loss + (tf.add_n(self.losses) if self.losses else 0.0)
        gradients = tape.gradient(loss, self.trainable_variables)
        pairs = [(g, v) for g, v in zip(gradients, self.trainable_variables) if g is not None]
        if not pairs:
            raise ValueError("Objective produced no gradients for trainable variables")
        self.optimizer.apply_gradients(pairs)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        batch = self._batch_from_keras(data)
        output = self(batch, training=False)
        if output.loss is None:
            raise ValueError(f"{self.task_name} evaluation requires targets and an Objective")
        loss = output.loss + (tf.add_n(self.losses) if self.losses else 0.0)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def build_from_config(self, config):
        """Build every child layer before Keras restores saved variables."""

        def make_dummy(spec):
            if isinstance(spec, dict) and "shape" in spec:
                shape, dtype = spec["shape"], spec["dtype"]
                if "value" in spec:
                    return tf.convert_to_tensor(spec["value"], dtype=dtype)
            else:
                shape, dtype = spec, self.compute_dtype
            shape = tf.TensorShape(shape).as_list()
            dimensions = [dimension if dimension is not None else 1 for dimension in shape]
            if tf.dtypes.as_dtype(dtype) == tf.string:
                return tf.fill(dimensions, "")
            return tf.zeros(dimensions, dtype=dtype)

        def make_inputs(shape):
            if isinstance(shape, dict):
                return {key: make_inputs(value) for key, value in shape.items()}
            if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple, tf.TensorShape)):
                return [make_inputs(value) for value in shape]
            return make_dummy(shape)

        batch_specs = config.get("batch_specs")
        if batch_specs is not None:
            self({name: make_dummy(spec) for name, spec in batch_specs.items()})
            return
        input_shape = config.get("input_shape")
        if input_shape is not None:
            self(make_inputs(input_shape))

    def get_build_config(self):
        """Persist canonical batch shapes and dtypes for Keras restoration."""
        batch_specs = getattr(self, "_batch_build_specs", None)
        if batch_specs is not None:
            return {"batch_specs": batch_specs}
        return super().get_build_config()

    def save(self, filepath, *args, **kwargs):
        """Save while tolerating Keras 2 callbacks' empty native-save options."""
        if os.fspath(filepath).endswith(".keras"):
            kwargs.pop("options", None)
        return super().save(filepath, *args, **kwargs)

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
        backbone_config.validate_checkpoint_config(backbone_values)
        backbone_config.update(backbone_values)

        model = build_task_model(
            backbone_config,
            task_config_from_dict(task_values),
            model_kwargs=config,
        )
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
            json.dump(
                {
                    "schema_version": 1,
                    "task_config": task_values,
                },
                file,
                indent=2,
            )
        self.save_weights(os.path.join(save_directory, TF2_WEIGHTS_NAME))


class BaseTask(ABC):
    """Semantic base for non-layer task services such as anomaly scorers."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


from tfts.contracts import ModelOutput as ModelOutput  # noqa: E402,F401
