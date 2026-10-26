"""Late 2D and 3D materialization from shared forecast windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from tfts.batch import TimeSeriesBatch
from tfts.data.features.base import PreparedTimeSeries
from tfts.data.schema import FeatureDType, FeaturePlan, FeatureRole, FeatureSelection, resolve_feature_plan

from .windows import WindowIndex


@dataclass(frozen=True)
class TabularBatch:
    """Model-ready matrix and labels for sklearn-compatible estimators."""

    X: np.ndarray
    y: Optional[np.ndarray]
    feature_names: Tuple[str, ...]
    categorical_feature_names: Tuple[str, ...]
    target_names: Tuple[str, ...]
    metadata: Tuple[Mapping[str, Any], ...]


class TabularMaterializer:
    """Flatten shared windows only at a tabular estimator boundary."""

    def __init__(self, strategy: str = "per_horizon", include_target_history: bool = True, fill_value: float = 0.0):
        if strategy not in {"per_horizon", "multioutput"}:
            raise ValueError("strategy must be either 'per_horizon' or 'multioutput'")
        self.strategy = strategy
        self.include_target_history = include_target_history
        self.fill_value = fill_value

    def materialize(
        self,
        prepared: PreparedTimeSeries,
        windows: WindowIndex,
        selection: Optional[FeatureSelection] = None,
        plan: Optional[FeaturePlan] = None,
    ) -> TabularBatch:
        plan = _one_plan(prepared, selection, plan)
        frame, schema = prepared.frame, prepared.schema
        rows: List[List[float]] = []
        labels: List[List[float]] = []
        metadata = []
        names, categorical_names = self._feature_names(schema, plan, windows)
        has_labels = _windows_have_labels(frame, schema, windows)

        for window in windows.windows:
            encoder = frame.iloc[list(window.encoder_indices)]
            decoder = frame.iloc[list(window.decoder_indices)]
            values: List[float] = []
            if self.include_target_history:
                for target in schema.target_cols:
                    values.extend(encoder[target].to_numpy().tolist())
            for feature in plan.selected:
                if feature.role == FeatureRole.STATIC:
                    values.append(encoder[feature.name].iloc[-1])
                else:
                    values.extend(encoder[feature.name].to_numpy().tolist())
                    if feature.role == FeatureRole.KNOWN_FUTURE and self.strategy == "multioutput":
                        if len(decoder) != windows.spec.prediction_length:
                            raise ValueError(
                                f"known-future feature {feature.name!r} requires {windows.spec.prediction_length} "
                                f"decoder rows, found {len(decoder)}"
                            )
                        values.extend(decoder[feature.name].to_numpy().tolist())
            if self.strategy == "multioutput":
                rows.append(values)
                if has_labels:
                    labels.append(decoder[list(schema.target_cols)].to_numpy().reshape(-1).tolist())
                metadata.append({"group_key": window.group_key, "cutoff_index": window.encoder_indices[-1]})
                continue

            known = [feature for feature in plan.selected if feature.role == FeatureRole.KNOWN_FUTURE]
            if known and len(decoder) != windows.spec.prediction_length:
                raise ValueError(
                    f"per_horizon materialization requires {windows.spec.prediction_length} decoder rows, "
                    f"found {len(decoder)}"
                )
            for horizon in range(windows.spec.prediction_length):
                horizon_values = list(values)
                horizon_values.extend(decoder[feature.name].iloc[horizon] for feature in known)
                horizon_values.append(horizon + 1)
                rows.append(horizon_values)
                if has_labels:
                    labels.append(decoder[list(schema.target_cols)].iloc[horizon].to_numpy().tolist())
                metadata.append(
                    {
                        "group_key": window.group_key,
                        "cutoff_index": window.encoder_indices[-1],
                        "horizon": horizon + 1,
                    }
                )

        X = _numeric_array(rows, self.fill_value, "tabular features")
        y = _numeric_array(labels, self.fill_value, "targets") if has_labels else None
        return TabularBatch(X, y, names, categorical_names, schema.target_cols, tuple(metadata))

    def _feature_names(self, schema, plan, windows):
        names, categorical_names = [], []
        context = windows.spec.context_length
        horizon = windows.spec.prediction_length
        if self.include_target_history:
            for target in schema.target_cols:
                names.extend(f"{target}@t-{offset}" for offset in range(context - 1, -1, -1))
        for feature in plan.selected:
            if feature.role == FeatureRole.STATIC:
                names.append(feature.name)
                if feature.dtype == FeatureDType.CATEGORICAL:
                    categorical_names.append(feature.name)
                continue
            past_names = [f"{feature.name}@t-{offset}" for offset in range(context - 1, -1, -1)]
            names.extend(past_names)
            if feature.dtype == FeatureDType.CATEGORICAL:
                categorical_names.extend(past_names)
            if feature.role == FeatureRole.KNOWN_FUTURE and self.strategy == "multioutput":
                future_names = [f"{feature.name}@t+{offset}" for offset in range(1, horizon + 1)]
                names.extend(future_names)
                if feature.dtype == FeatureDType.CATEGORICAL:
                    categorical_names.extend(future_names)
        if self.strategy == "per_horizon":
            for feature in plan.selected:
                if feature.role == FeatureRole.KNOWN_FUTURE:
                    name = f"{feature.name}@forecast"
                    names.append(name)
                    if feature.dtype == FeatureDType.CATEGORICAL:
                        categorical_names.append(name)
            names.append("forecast_horizon")
        return tuple(names), tuple(categorical_names)


class SequenceMaterializer:
    """Create the canonical 3D batch used by TensorFlow models."""

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def materialize(
        self,
        prepared: PreparedTimeSeries,
        windows: WindowIndex,
        selection: Optional[FeatureSelection] = None,
        plan: Optional[FeaturePlan] = None,
    ) -> TimeSeriesBatch:
        plan = _one_plan(prepared, selection, plan)
        dataset = self.dataset(prepared, windows, plan=plan, batch_size=len(windows))
        element = next(iter(dataset))
        if windows.spec.mode == "train":
            inputs, labels = element
            inputs = {**inputs, "future_values": labels, "labels": labels}
        else:
            inputs = element
        feature_names = {
            name: tuple(spec.name for spec in plan.selected if predicate(spec))
            for name, predicate in (
                ("past_real", _past_real),
                ("future_real", _future_real),
                ("past_categorical", _past_categorical),
                ("future_categorical", _future_categorical),
                ("static_real", _static_real),
                ("static_categorical", _static_categorical),
            )
        }
        return TimeSeriesBatch(
            **inputs, metadata={"feature_names": feature_names, "manifest_fingerprint": prepared.manifest.fingerprint}
        )

    def dataset(self, prepared, windows, *, plan=None, batch_size=32, shuffle=False, seed=0):
        """Slice flat role arrays lazily; memory does not grow with window width."""
        import tensorflow as tf

        plan = _one_plan(prepared, None, plan)
        frame, schema = prepared.frame, prepared.schema
        if not len(windows):
            raise ValueError("no valid windows could be constructed")
        columns = {
            "past_values": schema.target_cols,
            "past_time_features": [spec.name for spec in plan.selected if _past_real(spec)],
            "future_time_features": [spec.name for spec in plan.selected if _future_real(spec)],
            "past_categorical_features": [spec.name for spec in plan.selected if _past_categorical(spec)],
            "future_categorical_features": [spec.name for spec in plan.selected if _future_categorical(spec)],
            "static_real_features": [spec.name for spec in plan.selected if _static_real(spec)],
            "static_categorical_features": [spec.name for spec in plan.selected if _static_categorical(spec)],
        }
        arrays, masks = {}, {}
        for name, names in columns.items():
            if names:
                values = frame[list(names)].to_numpy()
                masks[name] = tf.convert_to_tensor(frame[list(names)].notna().to_numpy())
                arrays[name] = tf.convert_to_tensor(
                    _categorical_array(values, name)
                    if "categorical" in name
                    else _numeric_array(values, self.fill_value, name)
                )
        observed = tf.convert_to_tensor(frame[list(schema.target_cols)].notna().to_numpy())
        starts = windows.starts
        context, horizon = windows.spec.context_length, windows.spec.prediction_length
        training = windows.spec.mode == "train"
        if any(name.startswith("future_") for name in arrays):
            for window in windows.windows:
                _require_decoder(window, windows, columns.get("future_time_features", ()))
        dataset = tf.data.Dataset.from_tensor_slices(np.asarray(starts, dtype=np.int64))
        if shuffle:
            dataset = dataset.shuffle(len(starts), seed=seed)

        # Constant offsets preserve context/horizon dimensions during graph tracing.
        past_offsets = tf.range(context, dtype=tf.int64)
        future_offsets = tf.range(context, context + horizon, dtype=tf.int64)

        def gather(start):
            past = start + past_offsets
            future = start + future_offsets
            inputs = {}
            for name, values in arrays.items():
                index = (
                    start + context - 1
                    if name.startswith("static_")
                    else (future if name.startswith("future_") else past)
                )
                inputs[name] = tf.gather(values, index)
                if name != "past_values":
                    inputs[name + "_mask"] = tf.gather(masks[name], index)
            inputs["past_observed_mask"] = tf.gather(observed, past)
            if training:
                inputs["future_observed_mask"] = tf.gather(observed, future)
                return inputs, tf.gather(arrays["past_values"], future)
            return inputs

        return dataset.map(gather).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def as_tf_dataset(
        batch: TimeSeriesBatch,
        batch_size: int = 32,
        shuffle: bool = False,
        seed=None,
        include_future_values: bool = True,
    ):
        """Convert a batch to Keras data, optionally retaining teacher-forcing values."""
        inputs = batch.as_tensor_dict(include_structure=False)
        shared_structure = {}
        if batch.structure is not None:
            per_sample, shared_structure = batch.structure.split_tensor_dict()
            inputs.update(per_sample)
        labels = inputs.pop("labels", None)
        if not include_future_values:
            inputs.pop("future_values", None)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels) if labels is not None else inputs)
        if shuffle:
            dataset = dataset.shuffle(int(batch.past_values.shape[0]), seed=seed)
        dataset = dataset.batch(batch_size)
        if shared_structure:
            if labels is None:

                def attach_structure(values):
                    return {**values, **shared_structure}

            else:

                def attach_structure(values, target):
                    return {**values, **shared_structure}, target

            dataset = dataset.map(attach_structure)
        return dataset.prefetch(tf.data.AUTOTUNE)


def _one_plan(prepared, selection, plan):
    if selection is not None and plan is not None:
        raise ValueError("pass either selection or a resolved plan, not both")
    return plan or resolve_feature_plan(prepared.schema, selection)


def _past_real(spec):
    return spec.role in {FeatureRole.OBSERVED_PAST, FeatureRole.KNOWN_FUTURE} and spec.dtype != FeatureDType.CATEGORICAL


def _future_real(spec):
    return spec.role == FeatureRole.KNOWN_FUTURE and spec.dtype != FeatureDType.CATEGORICAL


def _past_categorical(spec):
    return spec.role in {FeatureRole.OBSERVED_PAST, FeatureRole.KNOWN_FUTURE} and spec.dtype == FeatureDType.CATEGORICAL


def _future_categorical(spec):
    return spec.role == FeatureRole.KNOWN_FUTURE and spec.dtype == FeatureDType.CATEGORICAL


def _static_real(spec):
    return spec.role == FeatureRole.STATIC and spec.dtype != FeatureDType.CATEGORICAL


def _static_categorical(spec):
    return spec.role == FeatureRole.STATIC and spec.dtype == FeatureDType.CATEGORICAL


def _numeric_array(values, fill_value, name):
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric. Encode categorical strings before materialization") from error
    if np.isinf(array).any():
        raise ValueError(f"{name} contains Inf")
    return np.where(np.isnan(array), fill_value, array)


def _categorical_array(values, name):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer category codes") from error
    if np.isinf(array).any() or (array < 0).any() or np.any(np.isfinite(array) & (array != np.floor(array))):
        raise ValueError(f"{name} must contain nonnegative integer category codes")
    if np.nan_to_num(array).max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError(f"{name} category codes exceed int32")
    return np.nan_to_num(array, nan=0).astype(np.int32)


def _stack_or_none(values: Sequence[np.ndarray], dtype):
    return np.stack(values).astype(dtype) if values else None


def _require_decoder(window, windows, features):
    if len(window.decoder_indices) != windows.spec.prediction_length:
        raise ValueError(
            f"known-future features {features} require {windows.spec.prediction_length} decoder rows, "
            f"found {len(window.decoder_indices)}"
        )


def _windows_have_labels(frame, schema, windows):
    for window in windows.windows:
        if len(window.decoder_indices) != windows.spec.prediction_length:
            return False
        targets = frame.iloc[list(window.decoder_indices)][list(schema.target_cols)]
        if targets.isna().any(axis=None):
            return False
    return True
