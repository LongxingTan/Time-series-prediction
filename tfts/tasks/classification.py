from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from tfts.forecast import ForecastOutput
from tfts.models.base import BackboneOutput, ModelOutput, OutputPort
from tfts.registry import ComponentSpec, register_task, resolve

from .auto_task import ClassificationHead, ReconstructionHead
from .base import TaskConfig, TaskType, TimeSeriesTaskModel


@dataclass(frozen=True)
class ClassificationTaskConfig(TaskConfig):
    task: TaskType = TaskType.CLASSIFICATION
    num_labels: int = 2
    hidden_units: Tuple[int, ...] = (128,)
    dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "hidden_units", tuple(int(units) for units in self.hidden_units))
        if self.num_labels <= 1:
            raise ValueError("num_labels must be greater than one")
        if any(units <= 0 for units in self.hidden_units):
            raise ValueError("hidden_units must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True)
class ClassificationOutput(ModelOutput):
    logits: Optional[tf.Tensor] = None
    probabilities: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None

    def prediction(self):
        """Return the task's primary prediction tensor."""
        return self.logits


@register_task("classification", config_class=ClassificationTaskConfig)
@tf.keras.utils.register_keras_serializable(package="tfts")
class ClassificationModel(TimeSeriesTaskModel):
    task_name = "classification"
    required_output_port = OutputPort.SEQUENCE
    default_objective = "ce"

    def __init__(self, backbone, task_config, capabilities, objective=None, **kwargs):
        if not capabilities.has_port(OutputPort.SEQUENCE):
            raise ValueError(f"{backbone.config.model_type} does not support classification")
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = ClassificationHead(task_config.num_labels, task_config.hidden_units, task_config.dropout)
        self.configure_objective(objective)

    def forward(self, inputs, training=None):
        batch = self.normalize_batch(inputs)
        backbone_output = self.adapter.forward(batch, training=training, require=OutputPort.SEQUENCE)
        logits = self.head(backbone_output.sequence_output, padding_mask=batch.padding_mask, training=training)
        return ClassificationOutput(logits=logits, probabilities=tf.nn.softmax(logits), backbone_output=backbone_output)
