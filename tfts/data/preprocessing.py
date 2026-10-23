"""Fitted preprocessing used by the estimator; no model-specific branches."""

from dataclasses import replace

import numpy as np
import tensorflow as tf

from tfts.contracts import TimeSeriesBatch

from .scaling import ChannelScaler

_REAL_INPUTS = ("past_values", "past_time_features", "future_time_features", "static_real_features")
_CATEGORICAL_INPUTS = ("past_categorical_features", "future_categorical_features", "static_categorical_features")
_MASK_INPUTS = ("past_observed_mask", "padding_mask")


def input_arrays(inputs):
    """Validate prepared inputs without changing feature roles or channel widths."""
    if isinstance(inputs, TimeSeriesBatch):
        inputs = inputs.as_tensor_dict()
    inputs = inputs if isinstance(inputs, dict) else {"past_values": inputs}
    allowed = set(_REAL_INPUTS + _CATEGORICAL_INPUTS + _MASK_INPUTS)
    if "past_values" not in inputs or set(inputs) - allowed:
        raise ValueError(f"Prepared inputs require past_values. Unsupported fields: {sorted(set(inputs) - allowed)}")
    arrays = {}
    for key, value in inputs.items():
        value = np.asarray(value)
        rank = 2 if key.startswith("static_") or key == "padding_mask" else 3
        if value.ndim != rank or not all(value.shape) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be a nonempty finite rank-{rank} array")
        if key in _CATEGORICAL_INPUTS:
            if not np.equal(value, np.floor(value)).all() or (value < 0).any():
                raise ValueError(f"{key} must contain nonnegative integer category ids")
            arrays[key] = value.astype(np.int32)
        else:
            arrays[key] = value.astype(np.float32)
    count = len(arrays["past_values"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("All input fields must have the same sample count")
    TimeSeriesBatch.from_inputs(arrays).validate_for("forecasting")
    return arrays


def target_arrays(targets, sample_weight=None):
    target = np.asarray(targets)
    if target.ndim not in (1, 2, 3) or not all(target.shape) or not np.isfinite(target).all():
        raise ValueError("Targets must be nonempty finite arrays")
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    if weight is not None:
        if weight.ndim < 1 or weight.ndim > target.ndim or weight.shape != target.shape[: weight.ndim]:
            raise ValueError("sample_weight must match leading target dimensions, e.g. (B,), (B,T), or (B,T,D)")
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("sample_weight must be finite and nonnegative")
    return target, weight


def inverse_output(output, scaler):
    """Restore task values with one fitted target scaler."""
    if scaler is None:
        return output
    changes = {}
    for name in ("predictions", "reconstructed_values", "imputed_values", "reconstruction", "samples"):
        value = getattr(output, name, None)
        if value is not None:
            changes[name] = scaler.inverse_transform(value)
    quantiles = getattr(output, "quantile_values", None)
    if quantiles is not None:
        changes["quantile_values"] = quantiles * scaler.scale[:, None] + scaler.offset[:, None]
    parameters = getattr(output, "distribution_params", None)
    if parameters is not None:
        if set(parameters) != {"loc", "scale"}:
            raise ValueError("Inverse scaling requires a location/scale distribution; use normalize=None")
        changes["distribution_params"] = {
            "loc": scaler.inverse_transform(parameters["loc"]),
            "scale": parameters["scale"] * scaler.scale,
        }
    return replace(output, **changes)


class ArrayPreprocessor:
    """Training-fitted scaling and a fixed input layout for prepared arrays."""

    def __init__(self, normalize="standard"):
        self.normalize = normalize

    def fit(
        self, inputs, target, weight=None, *, scale_target=True, target_history=False, matching_time_features=False
    ):
        self.shapes = {key: list(value.shape[1:]) for key, value in inputs.items()}
        self.scalers = {
            key: ChannelScaler(self.normalize).fit(
                value, inputs.get("past_observed_mask") if key == "past_values" else None
            )
            for key, value in inputs.items()
            if key in _REAL_INPUTS
        }
        # Known covariates must use the same units on both sides of the cutoff.
        if matching_time_features and "future_time_features" in inputs:
            past = inputs.get("past_time_features")
            future = inputs["future_time_features"]
            if past is not None and past.shape[-1] == future.shape[-1]:
                scaler = ChannelScaler(self.normalize).fit(np.concatenate([past, future], axis=1))
                self.scalers.update(past_time_features=scaler, future_time_features=scaler)
        self.target_scaler = ChannelScaler(self.normalize).fit(target, weight) if scale_target else None
        if target_history:
            if self.target_scaler is None or inputs["past_values"].shape[-1] != target.shape[-1]:
                raise ValueError("This backbone requires target histories with the same width as targets")
            self.scalers["past_values"] = self.target_scaler
        return self

    def transform(self, inputs):
        if set(inputs) != set(self.shapes) or any(
            list(value.shape[1:]) != self.shapes[key] for key, value in inputs.items()
        ):
            raise ValueError("Prediction inputs must match the fitted fields and window shapes")
        return {
            key: self.scalers[key].transform(value) if key in self.scalers else value for key, value in inputs.items()
        }

    def dataset(self, inputs, target=None, weight=None, *, batch_size=32, shuffle=False, seed=0):
        inputs = self.transform(inputs)
        if target is None:
            values = inputs
        else:
            if len(target) != len(inputs["past_values"]):
                raise ValueError("Inputs and targets must have the same sample count")
            target = self.target_scaler.transform(target) if self.target_scaler is not None else target
            values = (inputs, target) if weight is None else (inputs, target, weight)
        dataset = tf.data.Dataset.from_tensor_slices(values)
        if shuffle:
            dataset = dataset.shuffle(len(inputs["past_values"]), seed=seed)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def inverse_output(self, output):
        return inverse_output(output, self.target_scaler)

    def to_dict(self):
        return {
            "normalize": self.normalize,
            "shapes": self.shapes,
            "scalers": {key: scaler.to_dict() for key, scaler in self.scalers.items()},
            "target_scaler": None if self.target_scaler is None else self.target_scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, state):
        result = cls(state["normalize"])
        result.shapes = state["shapes"]
        result.scalers = {key: ChannelScaler.from_dict(value) for key, value in state["scalers"].items()}
        result.target_scaler = (
            None if state["target_scaler"] is None else ChannelScaler.from_dict(state["target_scaler"])
        )
        return result
