"""Task models compose backbones with small, typed task heads."""

from dataclasses import replace

import tensorflow as tf

from tfts.contracts import (
    AnomalyDetectionOutput,
    ClassificationOutput,
    ForecastMode,
    ForecastOutput,
    ImputationOutput,
    OutputPort,
)
from tfts.distributions import NormalOutput
from tfts.generation.policy import DecodePolicy

from .anomaly import QuantileCalibrator, make_anomaly_scorer
from .auto_task import (
    ClassificationHead,
    DistributionForecastHead,
    PointForecastHead,
    QuantileForecastHead,
    ReconstructionHead,
)
from .base import TimeSeriesTaskModel


@tf.keras.utils.register_keras_serializable(package="tfts")
class ForecastingModel(TimeSeriesTaskModel):
    task_name = "forecasting"

    def __init__(self, backbone, task_config, capabilities, **kwargs):
        super().__init__(backbone, task_config, capabilities, **kwargs)
        head_type = task_config.head
        self.head = None
        self.output_distribution = (
            getattr(backbone, "output_distribution", None) if capabilities.has_port(OutputPort.DISTRIBUTION) else None
        )
        if head_type in {"auto", "native"}:
            if not capabilities.has_port(OutputPort.NATIVE_FORECAST):
                raise ValueError("Native forecasting requires the native_forecast output port")
        elif head_type == "point":
            self._require_sequence()
            self.head = PointForecastHead(
                task_config.prediction_length, task_config.target_dim, residual=task_config.residual
            )
        elif head_type == "quantile":
            self._require_sequence()
            self.head = QuantileForecastHead(
                task_config.prediction_length, task_config.quantiles, task_config.target_dim
            )
        elif head_type == "distribution":
            if capabilities.has_port(OutputPort.DISTRIBUTION):
                if self.output_distribution is None:
                    raise ValueError("Backbone declares a distribution port but exposes no distribution")
            else:
                self._require_sequence()
                self.output_distribution = NormalOutput(target_dim=task_config.target_dim)
                self.head = DistributionForecastHead(self.output_distribution, task_config.prediction_length)
        self.generation_probabilistic = self.output_distribution is not None
        self._decode_policy = (
            DecodePolicy(backbone, capabilities, task_config)
            if self.head is None and ForecastMode.AUTOREGRESSIVE in capabilities.forecast_modes
            else None
        )
        if self.output_distribution is not None:
            self._loss_tracker = tf.keras.metrics.Mean(name="loss")

    def _require_sequence(self):
        if not self.capabilities.has_port(OutputPort.SEQUENCE):
            raise ValueError(
                "%s cannot use a learned forecast head because it exposes no sequence output"
                % self.backbone_config.model_type
            )

    def forward(self, inputs, training=None, teacher_probability=None):
        batch = self.normalize_batch(inputs)
        model_batch, restore = self.prepare_backbone_batch(batch)
        if self.head is None:
            if self._decode_policy is not None:
                probability = teacher_probability
                if probability is None:
                    probability = (
                        self._teacher_probability() if training and model_batch.future_values is not None else 0.0
                    )
                # Likelihood evaluation explicitly conditions on observed targets.
                if self.output_distribution is not None and model_batch.future_values is not None and not training:
                    probability = 1.0 if teacher_probability is None else teacher_probability
                result = self._decode_policy.run(model_batch, training=training, probability=probability)
                return ForecastOutput(
                    predictions=restore(result.predictions), distribution_params=restore(result.distribution_params)
                )
            backbone_output = self.adapter.forward(model_batch, training=training)
            predictions = restore(backbone_output.native_forecast)
            distribution_params = backbone_output.distribution_params
            if distribution_params is not None:
                distribution_params = restore(distribution_params)
            return ForecastOutput(
                predictions=predictions,
                distribution_params=distribution_params,
                backbone_output=backbone_output,
            )

        backbone_output = self.adapter.forward(model_batch, training=training, require=OutputPort.SEQUENCE)
        if isinstance(self.head, PointForecastHead):
            predictions = self.head(backbone_output.sequence_output, past_values=model_batch.past_values)
            predictions = restore(predictions)
            return ForecastOutput(predictions=predictions, backbone_output=backbone_output)
        if isinstance(self.head, QuantileForecastHead):
            values = restore(self.head(backbone_output.sequence_output))
            median_index = min(
                range(len(self.task_config.quantiles)), key=lambda i: abs(self.task_config.quantiles[i] - 0.5)
            )
            return ForecastOutput(
                predictions=values[..., median_index],
                quantile_values=values,
                quantiles=self.task_config.quantiles,
                backbone_output=backbone_output,
            )
        params = self.head(backbone_output.sequence_output)
        params = restore(params)
        return ForecastOutput(
            predictions=self.output_distribution.mean(params),
            distribution_params=params,
            backbone_output=backbone_output,
        )

    def primary_output(self, output):
        if output.quantile_values is not None:
            return output.quantile_values
        return output.predictions

    @property
    def default_loss(self):
        if self.task_config.head == "quantile":
            quantiles = tf.constant(self.task_config.quantiles, dtype=tf.float32)

            def quantile_loss(y_true, y_pred):
                error = tf.expand_dims(tf.cast(y_true, y_pred.dtype), -1) - y_pred
                q = tf.cast(quantiles, y_pred.dtype)
                return tf.reduce_mean(tf.maximum(q * error, (q - 1.0) * error))

            return quantile_loss
        if self.task_config.head == "distribution":
            return tf.keras.losses.MeanSquaredError()
        return tf.keras.losses.MeanSquaredError()

    @property
    def default_metrics(self):
        return (tf.keras.metrics.MeanAbsoluteError(name="mae"),)

    def _distribution_loss(self, batch, target, output):
        if target is None:
            target = batch.future_values
        if target is None:
            raise ValueError("Probabilistic forecasting training requires y or future_values")
        losses = self.output_distribution.loss(
            tf.cast(target, output.predictions.dtype), output.distribution_params, reduction="none"
        )
        if batch.future_observed_mask is None:
            return tf.reduce_mean(losses)
        mask = tf.cast(batch.future_observed_mask, losses.dtype)
        return tf.math.divide_no_nan(tf.reduce_sum(losses * mask), tf.reduce_sum(mask))

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if self._decode_policy is not None and y is not None:
            batch = self.normalize_batch(x)
            x = replace(batch, future_values=y).as_tensor_dict()
            data = (x, y, sample_weight) if sample_weight is not None else (x, y)
        if self.output_distribution is None:
            return super().train_step(data)
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch = self.normalize_batch(x)
        with tf.GradientTape() as tape:
            output = self(batch, training=True, return_dict=True)
            loss = self._distribution_loss(batch, y, output)
            if self.losses:
                loss += tf.add_n(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(pairs)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        if self.output_distribution is None:
            return super().test_step(data)
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch = self.normalize_batch(x)
        output = self(batch, training=False, return_dict=True)
        loss = self._distribution_loss(batch, y, output)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def generate(self, inputs, generation_config=None, **kwargs):
        from tfts.generation import generate

        return generate(self, inputs, generation_config=generation_config, **kwargs)

    def _teacher_probability(self):
        config = self.task_config
        if not config.teacher_decay_steps:
            return config.teacher_probability
        optimizer = getattr(self, "optimizer", None)
        step = tf.cast(optimizer.iterations if optimizer is not None else 0, tf.float32)
        fraction = tf.minimum(step / config.teacher_decay_steps, 1.0)
        return config.teacher_probability + fraction * (config.teacher_final_probability - config.teacher_probability)


@tf.keras.utils.register_keras_serializable(package="tfts")
class ClassificationModel(TimeSeriesTaskModel):
    task_name = "classification"
    required_output_port = OutputPort.SEQUENCE

    def __init__(self, backbone, task_config, capabilities, **kwargs):
        if not capabilities.has_port(OutputPort.SEQUENCE):
            raise ValueError("%s does not support classification" % backbone.config.model_type)
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = ClassificationHead(task_config.num_labels, task_config.hidden_units, task_config.dropout)

    def forward(self, inputs, training=None):
        batch = self.normalize_batch(inputs)
        backbone_output = self.adapter.forward(batch, training=training, require=OutputPort.SEQUENCE)
        logits = self.head(backbone_output.sequence_output, padding_mask=batch.padding_mask, training=training)
        return ClassificationOutput(logits=logits, probabilities=tf.nn.softmax(logits), backbone_output=backbone_output)

    def primary_output(self, output):
        return output.logits

    @property
    def default_loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @property
    def default_metrics(self):
        return (tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),)


@tf.keras.utils.register_keras_serializable(package="tfts")
class ImputationModel(TimeSeriesTaskModel):
    task_name = "imputation"
    required_output_port = OutputPort.TEMPORAL_SEQUENCE

    def __init__(self, backbone, task_config, capabilities, **kwargs):
        if not capabilities.has_port(OutputPort.TEMPORAL_SEQUENCE):
            raise ValueError("%s does not support imputation" % backbone.config.model_type)
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = ReconstructionHead(task_config.target_dim)
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")

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

    def primary_output(self, output):
        return output.imputed_values

    @property
    def default_loss(self):
        return tf.keras.losses.MeanSquaredError()

    def _masked_loss(self, batch, target, reconstruction):
        missing = 1.0 - tf.cast(batch.past_observed_mask, reconstruction.dtype)
        error = tf.square(tf.cast(target, reconstruction.dtype) - reconstruction) * missing
        return tf.math.divide_no_nan(tf.reduce_sum(error), tf.reduce_sum(missing))

    def train_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch = self.normalize_batch(x)
        target = batch.labels if y is None else y
        if target is None:
            raise ValueError("Imputation training requires clean targets as y or batch.labels")
        with tf.GradientTape() as tape:
            output = self(batch, training=True, return_dict=True)
            loss = self._masked_loss(batch, target, output.reconstructed_values)
            if self.losses:
                loss += tf.add_n(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(pairs)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch = self.normalize_batch(x)
        target = batch.labels if y is None else y
        if target is None:
            raise ValueError("Imputation evaluation requires clean targets as y or batch.labels")
        output = self(batch, training=False, return_dict=True)
        loss = self._masked_loss(batch, target, output.reconstructed_values)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}


@tf.keras.utils.register_keras_serializable(package="tfts")
class AnomalyDetectionModel(TimeSeriesTaskModel):
    task_name = "anomaly_detection"
    required_output_port = OutputPort.TEMPORAL_SEQUENCE

    def __init__(self, backbone, task_config, capabilities, **kwargs):
        if not capabilities.has_port(OutputPort.TEMPORAL_SEQUENCE):
            raise ValueError("%s does not support anomaly detection" % backbone.config.model_type)
        super().__init__(backbone, task_config, capabilities, **kwargs)
        self.head = ReconstructionHead(task_config.target_dim)
        self.scorer = make_anomaly_scorer(task_config.scorer)
        self.calibrator = QuantileCalibrator(task_config.threshold_quantile)
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")

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
        output.labels = self.calibrator.predict(output.scores)
        output.threshold = tf.identity(self.calibrator.threshold)
        output["labels"] = output.labels
        output["threshold"] = output.threshold
        return output

    def primary_output(self, output):
        return output.scores

    @property
    def default_loss(self):
        return tf.keras.losses.MeanSquaredError()

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch = self.normalize_batch(x)
        target = batch.past_values if y is None else y
        with tf.GradientTape() as tape:
            output = self(batch, training=True, return_dict=True)
            error = tf.square(tf.cast(target, output.reconstruction.dtype) - output.reconstruction)
            if batch.past_observed_mask is not None:
                mask = tf.cast(batch.past_observed_mask, error.dtype)
                loss = tf.math.divide_no_nan(tf.reduce_sum(error * mask), tf.reduce_sum(mask))
            else:
                loss = tf.reduce_mean(error)
            if self.losses:
                loss += tf.add_n(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(pairs)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch = self.normalize_batch(x)
        target = batch.past_values if y is None else y
        output = self(batch, training=False, return_dict=True)
        error = tf.square(tf.cast(target, output.reconstruction.dtype) - output.reconstruction)
        if batch.past_observed_mask is not None:
            mask = tf.cast(batch.past_observed_mask, error.dtype)
            loss = tf.math.divide_no_nan(tf.reduce_sum(error * mask), tf.reduce_sum(mask))
        else:
            loss = tf.reduce_mean(error)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}
