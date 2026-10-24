"""Late 2D and 3D materialization from shared forecast windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from tfts.contracts import TimeSeriesBatch
from tfts.features.pipeline import PreparedTimeSeries
from tfts.features.schema import FeatureDType, FeaturePlan, FeatureRole, FeatureSelection, resolve_feature_plan

from .windowing import WindowIndex


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
        frame, schema = prepared.frame, prepared.schema
        past_real = [spec.name for spec in plan.selected if _past_real(spec)]
        future_real = [spec.name for spec in plan.selected if _future_real(spec)]
        past_cat = [spec.name for spec in plan.selected if _past_categorical(spec)]
        future_cat = [spec.name for spec in plan.selected if _future_categorical(spec)]
        static_real = [spec.name for spec in plan.selected if _static_real(spec)]
        static_cat = [spec.name for spec in plan.selected if _static_categorical(spec)]

        past_values, observed_masks, labels = [], [], []
        past_reals, future_reals, past_cats, future_cats = [], [], [], []
        static_reals, static_cats = [], []
        has_labels = _windows_have_labels(frame, schema, windows)

        for window in windows.windows:
            encoder = frame.iloc[list(window.encoder_indices)]
            decoder = frame.iloc[list(window.decoder_indices)]
            target = _numeric_array(encoder[list(schema.target_cols)].to_numpy(), self.fill_value, "targets")
            past_values.append(target)
            observed_masks.append(~encoder[list(schema.target_cols)].isna().to_numpy())
            if has_labels:
                labels.append(
                    _numeric_array(decoder[list(schema.target_cols)].to_numpy(), self.fill_value, "future targets")
                )
            if past_real:
                past_reals.append(_numeric_array(encoder[past_real].to_numpy(), self.fill_value, "past real features"))
            if past_cat:
                past_cats.append(_categorical_array(encoder[past_cat].to_numpy(), "past categorical features"))
            if future_real:
                _require_decoder(window, windows, future_real)
                future_reals.append(
                    _numeric_array(decoder[future_real].to_numpy(), self.fill_value, "future real features")
                )
            if future_cat:
                _require_decoder(window, windows, future_cat)
                future_cats.append(_categorical_array(decoder[future_cat].to_numpy(), "future categorical features"))
            if static_real:
                static_reals.append(
                    _numeric_array(encoder[static_real].iloc[-1].to_numpy(), self.fill_value, "static real")
                )
            if static_cat:
                static_cats.append(_categorical_array(encoder[static_cat].iloc[-1].to_numpy(), "static categorical"))

        label_values = np.stack(labels).astype(np.float32) if has_labels else None
        metadata = {
            "feature_names": {
                "past_real": tuple(past_real),
                "future_real": tuple(future_real),
                "past_categorical": tuple(past_cat),
                "future_categorical": tuple(future_cat),
                "static_real": tuple(static_real),
                "static_categorical": tuple(static_cat),
            },
            "manifest_fingerprint": prepared.manifest.fingerprint,
        }
        return TimeSeriesBatch(
            past_values=np.stack(past_values).astype(np.float32),
            future_values=label_values,
            past_time_features=_stack_or_none(past_reals, np.float32),
            future_time_features=_stack_or_none(future_reals, np.float32),
            past_categorical_features=_stack_or_none(past_cats, np.int32),
            future_categorical_features=_stack_or_none(future_cats, np.int32),
            static_real_features=_stack_or_none(static_reals, np.float32),
            static_categorical_features=_stack_or_none(static_cats, np.int32),
            past_observed_mask=np.stack(observed_masks),
            labels=label_values,
            metadata=metadata,
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
        arrays = {}
        for name, names in columns.items():
            if names:
                values = frame[list(names)].to_numpy()
                arrays[name] = tf.convert_to_tensor(
                    _categorical_array(values, name)
                    if "categorical" in name
                    else _numeric_array(values, self.fill_value, name)
                )
        observed = tf.convert_to_tensor(frame[list(schema.target_cols)].notna().to_numpy())
        starts = []
        for window in windows.windows:
            indices = window.encoder_indices
            if indices[-1] - indices[0] + 1 != len(indices):
                raise ValueError("Lazy windows require rows sorted by group and time")
            if window.decoder_indices and window.decoder_indices[0] != indices[-1] + 1:
                raise ValueError("Decoder rows must immediately follow the context")
            starts.append(indices[0])
        context, horizon = windows.spec.context_length, windows.spec.prediction_length
        training = windows.spec.mode == "train"
        if any(name.startswith("future_") for name in arrays):
            for window in windows.windows:
                _require_decoder(window, windows, columns.get("future_time_features", ()))
        dataset = tf.data.Dataset.from_tensor_slices(np.asarray(starts, dtype=np.int32))
        if shuffle:
            dataset = dataset.shuffle(len(starts), seed=seed)

        def gather(start):
            past = tf.range(start, start + context)
            future = tf.range(start + context, start + context + horizon)
            inputs = {}
            for name, values in arrays.items():
                index = (
                    start + context - 1
                    if name.startswith("static_")
                    else (future if name.startswith("future_") else past)
                )
                inputs[name] = tf.gather(values, index)
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
        array = np.asarray(values, dtype=np.int32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer category codes") from error
    return array


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
