"""Serializable, training-fitted channel scaling for prepared windows."""

import numpy as np


class ChannelScaler:
    """Scale each channel across samples and time, preserving constant signals."""

    def __init__(self, method="standard"):
        if method not in {None, "standard", "minmax", "robust", "log1p"}:
            raise ValueError("normalize must be standard, minmax, robust, log1p, or None")
        self.method = method
        self.offset = None
        self.scale = None

    def fit(self, values, sample_weight=None):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 1 or not values.size:
            raise ValueError("Scaler values must be nonempty with a channel axis")
        if np.isinf(values).any():
            raise ValueError("Scaler values must not contain Inf")
        valid = np.isfinite(values)
        if sample_weight is not None:
            weight = np.asarray(sample_weight)
            if weight.ndim < 1 or weight.ndim > values.ndim or weight.shape != values.shape[: weight.ndim]:
                raise ValueError("sample_weight must match leading value dimensions")
            if not np.isfinite(weight).all() or (weight < 0).any():
                raise ValueError("sample_weight must be finite and nonnegative")
            valid &= (
                np.broadcast_to(weight.reshape(weight.shape + (1,) * (values.ndim - weight.ndim)), values.shape) > 0
            )
        offsets, scales = [], []
        for channel in range(values.shape[-1]):
            observed = values[..., channel][valid[..., channel]]
            if not observed.size:
                raise ValueError("Each target channel needs at least one positive-weight training label")
            if self.method == "standard":
                offset, scale = observed.mean(), observed.std()
            elif self.method == "minmax":
                offset, scale = observed.min(), np.ptp(observed)
            elif self.method == "robust":
                offset, scale = np.median(observed), np.subtract(*np.percentile(observed, [75, 25]))
            else:
                if self.method == "log1p" and (observed <= -1).any():
                    raise ValueError("log1p requires values greater than -1")
                offset, scale = 0.0, 1.0
            offsets.append(offset)
            scales.append(scale if scale > 1e-8 else 1.0)
        self.offset = np.asarray(offsets, dtype=np.float32)
        self.scale = np.asarray(scales, dtype=np.float32)
        return self

    def transform(self, values):
        if self.scale is None:
            raise RuntimeError("Scaler is not fitted")
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != len(self.scale) or np.isinf(values).any():
            raise ValueError("Scaler input width must match fitted channels and contain no Inf")
        if self.method == "log1p":
            if (values <= -1).any():
                raise ValueError("log1p requires values greater than -1")
            return np.log1p(values)
        return (values - self.offset) / self.scale

    def inverse_transform(self, values):
        if self.scale is None:
            raise RuntimeError("Scaler is not fitted")
        if self.method == "log1p":
            return np.expm1(values)
        return values * self.scale + self.offset

    def to_dict(self):
        return dict(method=self.method, offset=self.offset.tolist(), scale=self.scale.tolist())

    @classmethod
    def from_dict(cls, values):
        scaler = cls(values["method"])
        scaler.offset = np.asarray(values["offset"], dtype=np.float32)
        scaler.scale = np.asarray(values["scale"], dtype=np.float32)
        if (
            scaler.offset.ndim != 1
            or scaler.scale.shape != scaler.offset.shape
            or not np.isfinite(scaler.offset).all()
            or not np.isfinite(scaler.scale).all()
            or (scaler.scale <= 0).any()
        ):
            raise ValueError("Invalid saved scaler: finite offsets and positive scales are required")
        return scaler
