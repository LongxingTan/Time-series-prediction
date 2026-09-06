"""Task models compose a backbone, one head, and one batch-aware Objective."""

import tensorflow as tf

from tfts.contracts import AnomalyDetectionOutput, ClassificationOutput, ForecastOutput, ImputationOutput, OutputPort
from tfts.distributions import NormalOutput
from tfts.registry import ComponentSpec, register_task

from .anomaly import QuantileCalibrator, make_anomaly_scorer
from .auto_task import (
    ClassificationHead,
    DistributionForecastHead,
    PointForecastHead,
    QuantileForecastHead,
    ReconstructionHead,
)
from .base import TimeSeriesTaskModel


@register_task("forecasting")
@tf.keras.utils.register_keras_serializable(package="tfts")
class ForecastingModel(TimeSeriesTaskModel):
    task_name = "forecasting"
    required_output_port = None
    default_objective = "mse"

    def __init__(self, backbone, task_config, capabilities, objective=None, **kwargs):
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = None
        self.output_distribution = (
            getattr(backbone, "output_distribution", None) if capabilities.has_port(OutputPort.DISTRIBUTION) else None
        )
        head_type = task_config.head
        if head_type in {"auto", "native"}:
            if not capabilities.has_port(OutputPort.NATIVE_FORECAST):
                raise ValueError("Native forecasting requires the native_forecast output port")
        elif head_type == "point":
            self._require_sequence()
            self.head = PointForecastHead(
                task_config.output_chunk_length,
                task_config.target_dim,
                residual=task_config.residual,
            )
        elif head_type == "quantile":
            self._require_sequence()
            self.head = QuantileForecastHead(
                task_config.output_chunk_length,
                task_config.quantiles,
                task_config.target_dim,
            )
        elif head_type == "distribution":
            if capabilities.has_port(OutputPort.DISTRIBUTION):
                if self.output_distribution is None:
                    raise ValueError("Backbone declares a distribution port but exposes no distribution")
            else:
                self._require_sequence()
                self.output_distribution = NormalOutput(target_dim=task_config.target_dim)
                self.head = DistributionForecastHead(self.output_distribution, task_config.output_chunk_length)

        self.generation_probabilistic = self.output_distribution is not None
        self._native_decode = self.head is None and hasattr(backbone, "initialize_decode")
        self.teacher_probability = tf.Variable(
            task_config.teacher_probability,
            trainable=False,
            dtype=tf.float32,
            name="teacher_probability",
        )
        if objective is None:
            if head_type == "quantile":
                objective = ComponentSpec("quantile", {"quantiles": task_config.quantiles})
            elif self.output_distribution is not None:
                objective = ComponentSpec("nll", {"distribution": self.output_distribution})
            else:
                objective = self.default_objective
        self.configure_objective(objective)

    def _require_sequence(self):
        if not self.capabilities.has_port(OutputPort.SEQUENCE):
            raise ValueError(
                f"{self.backbone_config.model_type} cannot use a learned head because it exposes no sequence port"
            )

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

        backbone_output = self.adapter.forward(model_batch, training=training, require=OutputPort.SEQUENCE)
        if isinstance(self.head, PointForecastHead):
            prediction = self.head(backbone_output.sequence_output, past_values=model_batch.past_values)
            return ForecastOutput(predictions=restore(prediction), backbone_output=backbone_output)
        if isinstance(self.head, QuantileForecastHead):
            values = restore(self.head(backbone_output.sequence_output))
            median = min(range(len(self.task_config.quantiles)), key=lambda i: abs(self.task_config.quantiles[i] - 0.5))
            return ForecastOutput(
                predictions=values[..., median],
                quantile_values=values,
                quantiles=self.task_config.quantiles,
                backbone_output=backbone_output,
            )
        parameters = restore(self.head(backbone_output.sequence_output))
        return ForecastOutput(
            predictions=self.output_distribution.mean(parameters),
            distribution_params=parameters,
            backbone_output=backbone_output,
        )

    def generate(self, inputs, config=None, **kwargs):
        from tfts.generation import generate

        return generate(self, inputs, config=config, **kwargs)


@register_task("classification")
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


@register_task("imputation")
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


@register_task("anomaly_detection")
@tf.keras.utils.register_keras_serializable(package="tfts")
class AnomalyDetectionModel(TimeSeriesTaskModel):
    task_name = "anomaly_detection"
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
