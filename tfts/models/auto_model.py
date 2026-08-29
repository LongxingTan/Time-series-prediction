"""Task-aware model factories built on explicit backbone capabilities."""

import json
import os
from typing import Optional

import tensorflow as tf

from tfts.contracts import (
    AnomalyDetectionTaskConfig,
    ClassificationTaskConfig,
    ForecastTaskConfig,
    ImputationTaskConfig,
    TaskType,
)
from tfts.tasks.task_models import AnomalyDetectionModel, ClassificationModel, ForecastingModel, ImputationModel

from .registry import RegistryFieldView, get_model_capabilities, get_model_class

MODEL_MAPPING_NAMES = RegistryFieldView("class_name")

_TASK_MODEL_SPECS = {
    TaskType.FORECASTING: (ForecastTaskConfig, ForecastingModel),
    TaskType.CLASSIFICATION: (ClassificationTaskConfig, ClassificationModel),
    TaskType.IMPUTATION: (ImputationTaskConfig, ImputationModel),
    TaskType.ANOMALY_DETECTION: (AnomalyDetectionTaskConfig, AnomalyDetectionModel),
}


def task_config_from_dict(values):
    """Restore the right task dataclass from its serialized values."""
    values = dict(values)
    task_type = TaskType.normalize(values.get("task"))
    task_config_class, _ = _TASK_MODEL_SPECS[task_type]
    return task_config_class(**values)


def build_task_model(config, task_config, model_kwargs=None, spatial_strategy="raise"):
    """Build a task model through the single task-to-model registry."""
    task_type = TaskType.normalize(task_config.task)
    _, model_class = _TASK_MODEL_SPECS[task_type]
    prediction_length = getattr(task_config, "prediction_length", 1)
    backbone = AutoBackbone.from_config(config, prediction_length=prediction_length)
    capabilities = get_model_capabilities(config.model_type)
    return model_class(
        backbone,
        task_config,
        capabilities,
        spatial_strategy=spatial_strategy,
        **(model_kwargs or {}),
    )


class AutoBackbone:
    """Instantiate a registered architecture without attaching task behavior."""

    def __init__(self, *args, **kwargs):
        raise TypeError("AutoBackbone must be constructed with from_config()")

    @classmethod
    def from_config(cls, config, prediction_length: int = 1):
        try:
            backbone_class = get_model_class(config.model_type)
        except (AttributeError, ValueError) as error:
            raise ValueError("Unknown backbone config %r" % type(config).__name__) from error
        return backbone_class(config=config, predict_sequence_length=prediction_length)


class _BaseAutoTaskModel:
    task_type = None
    task_config_class = None

    def __init__(self, *args, **kwargs):
        raise TypeError("%s must be constructed with from_config()" % type(self).__name__)

    @classmethod
    def from_config(cls, config, task_config=None, **task_kwargs):
        spatial_strategy = task_kwargs.pop("spatial_strategy", "raise")
        if task_config is not None and task_kwargs:
            raise ValueError("Pass either task_config or task keyword arguments, not both")
        if "predict_sequence_length" in task_kwargs:
            if "prediction_length" in task_kwargs:
                raise ValueError("Use only prediction_length")
            task_kwargs["prediction_length"] = task_kwargs.pop("predict_sequence_length")
        task_config = task_config or cls.task_config_class(**task_kwargs)
        if TaskType.normalize(task_config.task) != cls.task_type:
            raise ValueError("Expected task %s, got %s" % (cls.task_type.value, task_config.task.value))
        return build_task_model(config, task_config, spatial_strategy=spatial_strategy)


class AutoModelForForecasting(_BaseAutoTaskModel):
    task_type = TaskType.FORECASTING
    task_config_class = ForecastTaskConfig
    model_class = ForecastingModel


class AutoModelForTimeSeriesClassification(_BaseAutoTaskModel):
    task_type = TaskType.CLASSIFICATION
    task_config_class = ClassificationTaskConfig
    model_class = ClassificationModel


class AutoModelForImputation(_BaseAutoTaskModel):
    task_type = TaskType.IMPUTATION
    task_config_class = ImputationTaskConfig
    model_class = ImputationModel


class AutoModelForAnomalyDetection(_BaseAutoTaskModel):
    task_type = TaskType.ANOMALY_DETECTION
    task_config_class = AnomalyDetectionTaskConfig
    model_class = AnomalyDetectionModel


class AutoModel:
    """Single task-aware entry point; explicit AutoModelFor classes are preferred."""

    def __init__(self, *args, **kwargs):
        raise TypeError("AutoModel must be constructed with from_config()")

    @classmethod
    def from_config(cls, config, task="forecasting", task_config=None, **task_kwargs):
        spatial_strategy = task_kwargs.pop("spatial_strategy", "raise")
        task_type = TaskType.normalize(task)
        if task_config is not None and task_kwargs:
            raise ValueError("Pass either task_config or task keyword arguments, not both")
        if "predict_sequence_length" in task_kwargs:
            if "prediction_length" in task_kwargs:
                raise ValueError("Use only prediction_length")
            task_kwargs["prediction_length"] = task_kwargs.pop("predict_sequence_length")
        task_config_class, _ = _TASK_MODEL_SPECS[task_type]
        task_config = task_config or task_config_class(**task_kwargs)
        if TaskType.normalize(task_config.task) != task_type:
            raise ValueError("Expected task %s, got %s" % (task_type.value, task_config.task.value))
        return build_task_model(config, task_config, spatial_strategy=spatial_strategy)

    @classmethod
    def from_pretrained(cls, model_directory, sample_batch=None):
        from tfts.constants import TF2_WEIGHTS_NAME

        from .auto_config import AutoConfig
        from .base import BaseConfig

        task_path = os.path.join(model_directory, "task_config.json")
        if not os.path.isfile(task_path):
            raise FileNotFoundError("Missing task artifact %s" % task_path)
        architecture = BaseConfig.from_pretrained(model_directory).to_dict()
        model_type = architecture.get("model_type")
        config = AutoConfig.for_model(model_type)
        config.update(architecture)
        with open(task_path, "r", encoding="utf-8") as file:
            artifact = json.load(file)
        if artifact.get("schema_version") not in {1, 2}:
            raise ValueError("Unsupported task artifact schema %r" % artifact.get("schema_version"))
        task_config = task_config_from_dict(artifact["task_config"])
        task_type = TaskType.normalize(task_config.task)
        model = build_task_model(config, task_config, spatial_strategy=artifact.get("spatial_strategy", "raise"))
        if sample_batch is None:
            input_shape = getattr(config, "input_shape", None)
            if input_shape is None or isinstance(input_shape, dict):
                raise ValueError("sample_batch is required to restore this model")
            sample_batch = tf.zeros([1] + list(input_shape), tf.float32)
            if task_type == TaskType.IMPUTATION:
                sample_batch = {
                    "past_values": sample_batch,
                    "past_observed_mask": tf.ones_like(sample_batch),
                }
        model(sample_batch)
        model.load_weights(os.path.join(model_directory, TF2_WEIGHTS_NAME))
        return model


# Short aliases that do not change the architecture.
AutoModelForPrediction = AutoModelForForecasting
AutoModelForClassification = AutoModelForTimeSeriesClassification
AutoModelForAnomaly = AutoModelForAnomalyDetection


class AutoModelForQuantile(AutoModelForForecasting):
    @classmethod
    def from_config(cls, config, task_config: Optional[ForecastTaskConfig] = None, **kwargs):
        if task_config is None:
            kwargs["head"] = "quantile"
        return super().from_config(config, task_config=task_config, **kwargs)
