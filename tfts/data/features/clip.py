"""Training-fitted clipping without changing original observations."""

import numpy as np

from tfts.data.schema import FeatureSpec
from tfts.registry import register_transform

from .base import FeatureTransform


@register_transform("clip")
class ClipTransform(FeatureTransform):
    def __init__(self, columns, lower=0.01, upper=0.99, suffix="_clipped"):
        self.columns = (columns,) if isinstance(columns, str) else tuple(columns)
        if not self.columns or not 0 <= lower < upper <= 1 or not suffix:
            raise ValueError("Clipping requires columns, a suffix and 0 <= lower < upper <= 1")
        self.lower, self.upper, self.suffix = lower, upper, suffix
        self.limits_ = {}

    def fit(self, frame, schema):
        self.limits_ = {}
        for column in self.columns:
            values = frame[column].to_numpy(dtype=float)
            if np.isinf(values).any() or not np.isfinite(values).any():
                raise ValueError(f"{column!r} requires finite training observations")
            self.limits_[column] = np.nanquantile(values, [self.lower, self.upper]).tolist()
        return self.state_dict()

    def state_dict(self):
        return dict(self.limits_)

    def load_state(self, state):
        if set(state) != set(self.columns):
            raise ValueError("Clipping state must match columns")
        self.limits_ = {}
        for name, limits in state.items():
            if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] > limits[1]:
                raise ValueError("Invalid clipping limits")
            self.limits_[name] = list(limits)

    def transform(self, frame, schema):
        if set(self.limits_) != set(self.columns):
            raise RuntimeError("ClipTransform is not fitted")
        result, specs = frame.copy(), []
        for column in self.columns:
            source = schema.get(column)
            name = column + self.suffix
            result[name] = frame[column].clip(*self.limits_[column])
            # A transformed target is observed history, never another target or known future.
            from tfts.data.schema import FeatureRole

            role = FeatureRole.OBSERVED_PAST if source.role == FeatureRole.TARGET else source.role
            specs.append(FeatureSpec(name, role, source.dtype, source=column, transform="clip"))
        return result, specs
