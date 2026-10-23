"""Serializable, training-fitted channel scaling for prepared windows."""

import numpy as np


class ChannelScaler:
    """Scale each channel across samples and time, preserving constant signals."""

    def __init__(self, method="standard"):
        if method not in {None, "standard", "minmax"}:
            raise ValueError("normalize must be 'standard', 'minmax', or None")
        self.method = method
        self.offset = None
        self.scale = None

    def fit(self, values, sample_weight=None):
        values = np.asarray(values, dtype=np.float64)
        valid = np.ones_like(values, dtype=bool)
        if sample_weight is not None:
            weight = np.asarray(sample_weight)
            valid = np.broadcast_to(weight.reshape(weight.shape + (1,) * (values.ndim - weight.ndim)), values.shape) > 0
        offsets, scales = [], []
        for channel in range(values.shape[-1]):
            observed = values[..., channel][valid[..., channel]]
            if not observed.size:
                raise ValueError("Each target channel needs at least one positive-weight training label")
            if self.method == "standard":
                offset, scale = observed.mean(), observed.std()
            elif self.method == "minmax":
                offset, scale = observed.min(), np.ptp(observed)
            else:
                offset, scale = 0.0, 1.0
            offsets.append(offset)
            scales.append(scale if scale > 1e-8 else 1.0)
        self.offset = np.asarray(offsets, dtype=np.float32)
        self.scale = np.asarray(scales, dtype=np.float32)
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.offset) / self.scale

    def inverse_transform(self, values):
        return values * self.scale + self.offset

    def to_dict(self):
        return dict(method=self.method, offset=self.offset.tolist(), scale=self.scale.tolist())

    @classmethod
    def from_dict(cls, values):
        scaler = cls(values["method"])
        scaler.offset = np.asarray(values["offset"], dtype=np.float32)
        scaler.scale = np.asarray(values["scale"], dtype=np.float32)
        return scaler
