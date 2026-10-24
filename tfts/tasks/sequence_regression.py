from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from tfts.forecast import ForecastOutput
from tfts.models.base import BackboneOutput, ModelOutput, OutputPort
from tfts.registry import ComponentSpec, register_task, resolve

from .auto_task import ClassificationHead, ReconstructionHead
from .base import TaskConfig, TaskType, TimeSeriesTaskModel


@dataclass(frozen=True)
class SequenceRegressionTaskConfig(TaskConfig):
    """Predict targets aligned with a complete input window."""

    task: TaskType = TaskType.SEQUENCE_REGRESSION
    output_chunk_length: int = 1
    target_dim: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.output_chunk_length <= 0 or self.target_dim <= 0:
            raise ValueError("output_chunk_length and target_dim must be positive")


@register_task("sequence_regression", config_class=SequenceRegressionTaskConfig)
@tf.keras.utils.register_keras_serializable(package="tfts")
class SequenceRegressionModel(TimeSeriesTaskModel):
    """Aligned regression from any backbone with temporal representations."""

    task_name = "sequence_regression"
    default_objective = "sequence_mse"

    def __init__(self, backbone, task_config, capabilities, objective=None, **kwargs):
        super().__init__(backbone, task_config, capabilities, **kwargs)
        if not capabilities.has_port(OutputPort.TEMPORAL_SEQUENCE):
            raise ValueError(
                f"{backbone.config.model_type} does not support sequence_regression: "
                "a temporal_sequence port is required"
            )
        self.head = ReconstructionHead(task_config.target_dim)
        self.configure_objective(objective)

    def forward(self, inputs, training=None):
        batch = self.normalize_batch(inputs)
        model_batch, restore = self.prepare_backbone_batch(batch)
        output = self.adapter.forward(model_batch, training=training, require=OutputPort.TEMPORAL_SEQUENCE)
        tf.debugging.assert_equal(
            tf.shape(output.sequence_output)[1],
            model_batch.context_length,
            message="Temporal output must align with every input step",
        )
        predictions = self.head(output.sequence_output)
        return ForecastOutput(predictions=restore(predictions))
