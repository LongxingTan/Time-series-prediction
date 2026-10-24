from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from tfts.forecast import ForecastOutput
from tfts.models.base import BackboneOutput, ModelOutput, OutputPort
from tfts.registry import ComponentSpec, register_task, resolve

from .auto_task import ClassificationHead, ReconstructionHead
from .base import TaskConfig, TaskType, TimeSeriesTaskModel


@dataclass(frozen=True)
class ForecastTaskConfig(TaskConfig):
    task: TaskType = TaskType.FORECASTING
    output_chunk_length: int = 1
    target_dim: int = 1
    head: Any = "auto"
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    residual: Optional[str] = None
    spatial_strategy: str = "raise"
    teacher_probability: float = 1.0
    teacher_final_probability: float = 0.0
    teacher_decay_steps: int = 0
    feedback_sampler: str = "point"

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.teacher_probability <= 1 or not 0 <= self.teacher_final_probability <= 1:
            raise ValueError("teacher probabilities must lie in [0, 1]")
        if self.teacher_decay_steps < 0:
            raise ValueError("teacher_decay_steps must be non-negative")
        if self.feedback_sampler not in {"point", "sample"}:
            raise ValueError("feedback_sampler must be 'point' or 'sample'")
        object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles))
        if self.output_chunk_length <= 0 or self.target_dim <= 0:
            raise ValueError("output_chunk_length and target_dim must be positive")
        from tfts.registry import ComponentSpec

        head_name = ComponentSpec.from_value(self.head).name
        if not isinstance(self.head, str):
            object.__setattr__(self, "head", ComponentSpec.from_value(self.head))
        if head_name == "quantile":
            if not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles:
                raise ValueError("quantiles must be unique and sorted")
            if any(q <= 0 or q >= 1 for q in self.quantiles):
                raise ValueError("quantiles must lie strictly between 0 and 1")
        if self.residual not in {None, "last_value", "mean", "last_window"}:
            raise ValueError("Unknown residual %r" % self.residual)
        if self.spatial_strategy not in {"raise", "per_node"}:
            raise ValueError("spatial_strategy must be 'raise' or 'per_node'")


@register_task("forecasting", config_class=ForecastTaskConfig)
@tf.keras.utils.register_keras_serializable(package="tfts")
class ForecastingModel(TimeSeriesTaskModel):
    task_name = "forecasting"
    target_field = "future_values"
    required_output_port = None
    default_objective = "mse"

    def __init__(self, backbone, task_config, capabilities, objective=None, **kwargs):
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = None
        self.output_distribution = (
            getattr(backbone, "output_distribution", None) if capabilities.has_port(OutputPort.DISTRIBUTION) else None
        )
        head_spec = ComponentSpec.from_value(task_config.head)
        head_type = head_spec.name
        if head_type == "auto":
            head_type = "native" if capabilities.has_port(OutputPort.NATIVE_FORECAST) else "point"
        if head_type == "native":
            if not capabilities.has_port(OutputPort.NATIVE_FORECAST):
                raise ValueError("Native forecasting requires the native_forecast output port")
        elif head_type == "distribution" and capabilities.has_port(OutputPort.DISTRIBUTION):
            if self.output_distribution is None:
                raise ValueError("Backbone declares a distribution port but exposes no distribution")
        else:
            head_class = resolve("head", head_type).factory
            required = head_class.required_output_port
            if not capabilities.has_port(required):
                raise ValueError(
                    f"{head_type} requires {required.value}, which {self.backbone_config.model_type} does not expose"
                )
            self.head = head_class.from_task_config(task_config, **head_spec.kwargs)
            self.output_distribution = getattr(self.head, "distribution", None)

        self.generation_probabilistic = self.output_distribution is not None
        self._native_decode = self.head is None and hasattr(backbone, "initialize_decode")
        self.teacher_probability = tf.Variable(
            task_config.teacher_probability,
            trainable=False,
            dtype=tf.float32,
            name="teacher_probability",
        )
        if objective is None:
            if self.head is not None:
                objective = self.head.default_objective
            elif self.output_distribution is not None:
                objective = ComponentSpec("nll", {"distribution": self.output_distribution})
            else:
                objective = self.default_objective
        self.configure_objective(objective)

    def forward(self, inputs, training=None, teacher_probability=None):
        batch = self.normalize_batch(inputs)
        model_batch, restore = self.prepare_backbone_batch(batch)
        if self.head is None:
            if self._native_decode:
                probability = teacher_probability
                if probability is None:
                    probability = (
                        self.teacher_probability if training and model_batch.future_values is not None else 0.0
                    )
                if self.output_distribution is not None and model_batch.future_values is not None and not training:
                    probability = 1.0 if teacher_probability is None else teacher_probability
                if (
                    hasattr(self.backbone, "decode_teacher_forced")
                    and isinstance(probability, (float, int))
                    and probability == 1.0
                    and model_batch.future_observed_mask is None
                ):
                    result = self.backbone.decode_teacher_forced(model_batch, training=training)
                    return ForecastOutput(
                        predictions=restore(result.predictions),
                        distribution_params=restore(result.distribution_params),
                    )
                from tfts.generation.decoders import NativeDecoder
                from tfts.generation.loop import run
                from tfts.generation.processors import Mean, Sample, StepProcessorList, TeacherForcing

                selector = (
                    Sample(self.output_distribution)
                    if training and self.task_config.feedback_sampler == "sample"
                    else Mean(self.output_distribution)
                )
                processors = [selector]
                if model_batch.future_values is not None:
                    processors.append(
                        TeacherForcing(
                            model_batch.future_values,
                            probability,
                            model_batch.future_observed_mask,
                        )
                    )
                result = run(
                    NativeDecoder(self),
                    model_batch,
                    horizon=self.task_config.output_chunk_length,
                    processors=StepProcessorList(processors),
                    training=training,
                )
                return ForecastOutput(
                    predictions=restore(result.predictions),
                    distribution_params=restore(result.distribution_params),
                )
            backbone_output = self.adapter.forward(model_batch, training=training)
            parameters = backbone_output.distribution_params
            return ForecastOutput(
                predictions=restore(backbone_output.native_forecast),
                distribution_params=None if parameters is None else restore(parameters),
                backbone_output=backbone_output,
            )

        backbone_output = self.adapter.forward(model_batch, training=training, require=self.head.required_output_port)
        value = self.head(backbone_output.sequence_output, past_values=model_batch.past_values, training=training)
        return self.head.to_output(restore(value)).replace(backbone_output=backbone_output)

    def generate(self, inputs, config=None, **kwargs):
        from tfts.generation import generate

        return generate(self, inputs, config=config, **kwargs)
