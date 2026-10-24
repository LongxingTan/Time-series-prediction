"""Base classes for task heads and task models."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
from typing import Any, Dict

import tensorflow as tf

from tfts.batch import TimeSeriesBatch
from tfts.models.base import ModelOutput
from tfts.registry import build


class TaskType(str, Enum):
    FORECASTING = "forecasting"
    SEQUENCE_REGRESSION = "sequence_regression"
    IMPUTATION = "imputation"
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        normalized = str(value).lower().replace("-", "_")
        return cls(normalized)


@dataclass(frozen=True)
class TaskConfig:
    task: TaskType

    def __post_init__(self):
        object.__setattr__(self, "task", TaskType.normalize(self.task))

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["task"] = self.task.value
        return values


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
    target_field = "labels"
    default_target_field = None

    def __init__(self, backbone, task_config, capabilities, **kwargs):
        super().__init__(**kwargs)
        from tfts.models._adapters import BackboneAdapter, SpatialAdapter

        self.backbone = backbone
        self.backbone_config = backbone.config
        self.task_config = task_config
        self.capabilities = capabilities
        if not capabilities.input_spec.supports_multivariate_target and getattr(task_config, "target_dim", 1) > 1:
            raise ValueError("This backbone does not support multivariate targets")
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

    def dry_run(self, data):
        """Validate a Keras batch, output and objective without updating weights."""
        batch = self._batch_from_keras(data)
        output = self(batch, training=False, compute_loss=False)
        self._objective_result(batch, output)
        return output

    def configure_objective(self, objective=None, **context):
        from tfts.registry import ComponentSpec

        component = self.default_objective if objective is None else objective
        self._objective_spec = None
        if isinstance(component, (str, dict, ComponentSpec)):
            spec = ComponentSpec.from_value(component)
            self._objective_spec = ComponentSpec(
                spec.name, {key: value for key, value in spec.kwargs.items() if key != "distribution"}
            )
            if spec.name == "nll" and "distribution" not in spec.kwargs:
                context["distribution"] = getattr(self, "output_distribution", None)
        self.objective = None if component is None else build("objective", component, **context)

    def call(self, inputs, training=None, compute_loss=True):
        output = self.forward(inputs, training=training)
        batch = self.normalize_batch(inputs)
        if compute_loss and self.objective is not None and self.has_targets(batch):
            output = output.replace(loss=self._objective_result(batch, output).value)
        return output

    def has_targets(self, batch):
        return getattr(batch, self.target_field) is not None

    @classmethod
    def resolve_targets(cls, batch, targets=None):
        """Resolve explicit, embedded, or task-default labels in that order."""
        if targets is None:
            targets = getattr(batch, cls.target_field)
        if targets is None and cls.default_target_field is not None:
            targets = getattr(batch, cls.default_target_field)
        return targets

    def _batch_from_keras(self, data):
        from dataclasses import replace

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch = self.normalize_batch(x)
        changes = {}
        if sample_weight is not None:
            changes["sample_weight"] = tf.convert_to_tensor(sample_weight)
        target = self.resolve_targets(batch, y)
        if target is not None:
            target = tf.convert_to_tensor(target)
            changes[self.target_field] = tf.cast(target, tf.float32) if target.dtype.is_floating else target
        return replace(batch, **changes) if changes else batch

    @property
    def metrics(self):
        return [self._loss_tracker]

    def _objective_result(self, batch, output):
        if self.objective is None or not self.has_targets(batch):
            raise ValueError(f"{self.task_name} training requires targets and an Objective")
        if batch.sample_weight is not None and not getattr(self.objective, "supports_sample_weight", False):
            raise ValueError("This objective does not support sample_weight")
        if not hasattr(self.objective, "reduce"):
            raise TypeError("Training objectives must implement reduce(batch, output) -> LossResult(total, weight)")
        return self.objective.reduce(batch, output)

    def train_step(self, data):
        batch = self._batch_from_keras(data)
        with tf.GradientTape() as tape:
            output = self(batch, training=True, compute_loss=False)
            result = self._objective_result(batch, output)
            loss = result.value + (
                tf.add_n([tf.cast(value, tf.float32) for value in self.losses]) if self.losses else 0.0
            )
            if hasattr(self.optimizer, "scale_loss"):
                differentiated_loss = self.optimizer.scale_loss(loss)
            elif hasattr(self.optimizer, "get_scaled_loss"):
                differentiated_loss = self.optimizer.get_scaled_loss(loss)
            else:
                differentiated_loss = loss
        gradients = tape.gradient(differentiated_loss, self.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        pairs = [(g, v) for g, v in zip(gradients, self.trainable_variables) if g is not None]
        if not pairs:
            raise ValueError("Objective produced no gradients for trainable variables")
        self.optimizer.apply_gradients(pairs)
        self._loss_tracker.update_state(result.value, sample_weight=result.weight)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        batch = self._batch_from_keras(data)
        output = self(batch, training=False, compute_loss=False)
        result = self._objective_result(batch, output)
        self._loss_tracker.update_state(result.value, sample_weight=result.weight)
        return {"loss": self._loss_tracker.result()}

    def predict_step(self, data):

        inputs, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return self(inputs, training=False, compute_loss=False).prediction()

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
        if self.objective is not None and self._objective_spec is None:
            raise ValueError("To save a custom objective, register it and configure it with ComponentSpec")
        config = super().get_config()
        config.update(
            {
                "backbone_config": self.backbone_config.to_dict(),
                "task_config": self.task_config.to_dict(),
                "objective": None if self._objective_spec is None else self._objective_spec.to_dict(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct a task model from a Keras object config."""
        from tfts.models.auto_config import AutoConfig
        from tfts.tasks.auto import build_task_model, task_config_from_dict

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


from tfts.models.base import ModelOutput as ModelOutput
