from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from tfts.forecast import ForecastOutput
from tfts.models.base import BackboneOutput, ModelOutput, OutputPort
from tfts.registry import ComponentSpec, register_task, resolve

from .auto_task import ClassificationHead, ReconstructionHead
from .base import TaskConfig, TaskType, TimeSeriesTaskModel


@dataclass(frozen=True)
class ImputationTaskConfig(TaskConfig):
    task: TaskType = TaskType.IMPUTATION
    target_dim: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.target_dim <= 0:
            raise ValueError("target_dim must be positive")


@dataclass(frozen=True)
class ImputationOutput(ModelOutput):
    reconstructed_values: Optional[tf.Tensor] = None
    imputed_values: Optional[tf.Tensor] = None
    mask: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None

    def prediction(self):
        """Return the task's primary prediction tensor."""
        return self.imputed_values


@register_task("imputation", config_class=ImputationTaskConfig)
@tf.keras.utils.register_keras_serializable(package="tfts")
class ImputationModel(TimeSeriesTaskModel):
    task_name = "imputation"
    required_output_port = OutputPort.TEMPORAL_SEQUENCE
    default_objective = "masked_mse"

    def __init__(self, backbone, task_config, capabilities, objective=None, **kwargs):
        if not capabilities.has_port(OutputPort.TEMPORAL_SEQUENCE):
            raise ValueError(f"{backbone.config.model_type} does not support imputation")
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = ReconstructionHead(task_config.target_dim)
        self.configure_objective(objective)

    def forward(self, inputs, training=None):
        batch = self.normalize_batch(inputs)
        backbone_output = self.adapter.forward(batch, training=training, require=OutputPort.TEMPORAL_SEQUENCE)
        temporal = backbone_output.sequence_output[:, : tf.shape(batch.past_values)[1], :]
        reconstructed = self.head(temporal)
        mask = tf.cast(batch.past_observed_mask, reconstructed.dtype)
        imputed = mask * tf.cast(batch.past_values, reconstructed.dtype) + (1.0 - mask) * reconstructed
        return ImputationOutput(
            reconstructed_values=reconstructed,
            imputed_values=imputed,
            mask=batch.past_observed_mask,
            backbone_output=backbone_output,
        )
