"""Input-specific preparation behind one estimator orchestration path."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.data.arrays import ArrayPreprocessor, input_arrays, target_arrays
from tfts.data.pipeline import FramePipeline
from tfts.models.catalog import get_model_capabilities
from tfts.registry import register, registrations
from tfts.tasks.base import TaskType


@dataclass(frozen=True)
class SourceOptions:
    task: TaskType
    batch_size: int
    seed: int
    schema: Any
    context: int
    horizon: int
    normalize: Any
    calendar: tuple
    freq: Any
    validation: Any
    configuration: Callable
    configuration_for_targets: Callable
    training_targets: Callable


@dataclass(frozen=True)
class PreparedData:
    train: Any
    valid: Any
    preprocessor: Any
    config: Any
    task_config: Any
    report: dict
    input_kind: str


@register("source", "frame", priority=100)
class FrameSource:
    @staticmethod
    def accepts(inputs):
        return isinstance(inputs, pd.DataFrame)

    def prepare(self, inputs, targets, sample_weight, validation_data, options):
        task, batch_size = options.task, options.batch_size
        preprocessor, report = None, {}
        if task != TaskType.FORECASTING:
            raise ValueError("Dataframe fitting is for forecasting; use arrays or a dataset for other tasks")
        if targets is not None or sample_weight is not None or validation_data is not None:
            raise ValueError("Declare dataframe targets in schema and use validation for the temporal split")
        if options.schema is None:
            raise ValueError("Dataframe fitting requires Schema(time_col=..., target_cols=...)")
        if options.context < 1 or options.horizon < 1:
            raise ValueError("context and horizon must be positive")
        config, task_config = options.configuration(options.horizon, len(options.schema.target_cols))
        capabilities = get_model_capabilities(config.model_type)
        preprocessor = FramePipeline(
            options.schema, options.context, options.horizon, options.normalize, options.calendar, options.freq
        )
        prepared, train_windows, valid_windows = preprocessor.fit(inputs, options.validation)
        from .compat import resolve_feature_plan

        resolve_feature_plan(prepared.schema, input_spec=capabilities.input_spec)
        train = preprocessor.dataset(prepared, train_windows, batch_size=batch_size, shuffle=True, seed=options.seed)
        valid = None if valid_windows is None else preprocessor.dataset(prepared, valid_windows, batch_size=batch_size)
        report = preprocessor.fit_report
        input_kind = "frame"
        return PreparedData(train, valid, preprocessor, config, task_config, report, input_kind)


@register("source", "dataset", priority=100)
class DatasetSource:
    @staticmethod
    def accepts(inputs):
        return isinstance(inputs, (tf.data.Dataset, tf.keras.utils.Sequence))

    def prepare(self, inputs, targets, sample_weight, validation_data, options):
        preprocessor, report = None, {}
        if targets is not None or sample_weight is not None:
            raise ValueError("Include targets and weights in the dataset batches")
        train, valid = inputs, validation_data
        if isinstance(train, tf.data.Dataset):
            cardinality = int(tf.data.experimental.cardinality(train))
            if cardinality == 0:
                raise ValueError("Training dataset must not be empty")
            if cardinality == int(tf.data.INFINITE_CARDINALITY):
                raise ValueError("Training dataset must be finite; apply take() before fitting")
        first = next(iter(train)) if isinstance(train, tf.data.Dataset) else train[0]
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(first)
        y = options.training_targets(x, y)
        config, task_config = options.configuration_for_targets(y, dataset=True)
        input_kind = "dataset"
        return PreparedData(train, valid, preprocessor, config, task_config, report, input_kind)


@register("source", "array", priority=0)
class ArraySource:
    @staticmethod
    def accepts(inputs):
        return True

    def prepare(self, inputs, targets, sample_weight, validation_data, options):
        task, batch_size = options.task, options.batch_size
        preprocessor, report = None, {}
        x = input_arrays(inputs)
        y, weight = target_arrays(options.training_targets(x, targets), sample_weight)
        classification = task == TaskType.CLASSIFICATION
        config, task_config = options.configuration_for_targets(y)
        spec = get_model_capabilities(config.model_type).input_spec
        preprocessor = ArrayPreprocessor(options.normalize).fit(
            x,
            y,
            weight,
            scale_target=not classification,
            target_history=spec.requires_target_history or task in {TaskType.IMPUTATION, TaskType.ANOMALY_DETECTION},
            matching_time_features=spec.requires_matching_time_features,
        )
        train = preprocessor.dataset(x, y, weight, batch_size=batch_size, shuffle=True, seed=options.seed)
        valid = None
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) not in (2, 3):
                raise ValueError("Provide validation_data=(X_valid, y_valid[, sample_weight])")
            vx = input_arrays(validation_data[0])
            vy, vw = target_arrays(
                options.training_targets(vx, validation_data[1]),
                validation_data[2] if len(validation_data) == 3 else None,
            )
            if vy.shape[1:] != y.shape[1:]:
                raise ValueError("Training and validation target shapes must match")
            valid = preprocessor.dataset(vx, vy, vw, batch_size=batch_size)
        input_kind = "array"
        return PreparedData(train, valid, preprocessor, config, task_config, report, input_kind)


def source_for(inputs):
    matches = [
        (entry.metadata.get("priority", 0), name, entry.factory)
        for name, entry in registrations("source").items()
        if entry.factory.accepts(inputs)
    ]
    if not matches:
        raise TypeError("No registered source accepts these inputs")
    priority = max(match[0] for match in matches)
    best = [match for match in matches if match[0] == priority]
    if len(best) != 1:
        raise ValueError(f"Ambiguous input sources: {sorted(match[1] for match in best)}")
    return best[0][2]()
