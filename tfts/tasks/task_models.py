"""Task models compose a backbone, one head, and one batch-aware Objective."""

import tensorflow as tf

from tfts.contracts import (
    AnomalyDetectionOutput,
    AnomalyDetectionTaskConfig,
    ClassificationOutput,
    ClassificationTaskConfig,
    ForecastOutput,
    ForecastTaskConfig,
    ImputationOutput,
    ImputationTaskConfig,
    OutputPort,
    SequenceRegressionTaskConfig,
)
from tfts.registry import ComponentSpec, register_task, resolve

from .anomaly import QuantileCalibrator, make_anomaly_scorer
from .auto_task import ClassificationHead, ReconstructionHead
from .base import TimeSeriesTaskModel


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


@register_task("anomaly_detection", config_class=AnomalyDetectionTaskConfig)
@tf.keras.utils.register_keras_serializable(package="tfts")
class AnomalyDetectionModel(TimeSeriesTaskModel):
    task_name = "anomaly_detection"
    default_target_field = "past_values"
    required_output_port = OutputPort.TEMPORAL_SEQUENCE
    default_objective = "masked_mse"

    def __init__(self, backbone, task_config, capabilities, objective=None, **kwargs):
        if not capabilities.has_port(OutputPort.TEMPORAL_SEQUENCE):
            raise ValueError(f"{backbone.config.model_type} does not support anomaly detection")
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = ReconstructionHead(task_config.target_dim)
        self.scorer = make_anomaly_scorer(task_config.scorer)
        self.calibrator = QuantileCalibrator(task_config.threshold_quantile)
        objective = objective or ComponentSpec(
            "masked_mse",
            {"selection": "observed", "target_field": "labels", "prediction_field": "reconstruction"},
        )
        self.configure_objective(objective)

    def forward(self, inputs, training=None):
        batch = self.normalize_batch(inputs)
        backbone_output = self.adapter.forward(batch, training=training, require=OutputPort.TEMPORAL_SEQUENCE)
        temporal = backbone_output.sequence_output[:, : tf.shape(batch.past_values)[1], :]
        reconstruction = self.head(temporal)
        scores = self.scorer(batch.past_values, reconstruction, batch.past_observed_mask)
        return AnomalyDetectionOutput(
            reconstruction=reconstruction,
            scores=scores,
            backbone_output=backbone_output,
        )

    def calibrate(self, inputs):
        return self.calibrator.fit(self.forward(inputs, training=False).scores)

    def detect(self, inputs):
        output = self.forward(inputs, training=False)
        return output.replace(
            labels=self.calibrator.predict(output.scores),
            threshold=tf.identity(self.calibrator.threshold),
        )


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
