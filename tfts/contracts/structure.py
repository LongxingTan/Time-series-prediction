"""Spatial layouts and topology carried by :class:`TimeSeriesBatch`."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

import tensorflow as tf


class SpatialLayout(str, Enum):
    """Spatial axes present between time and feature dimensions."""

    NONE = "none"
    NODES = "nodes"
    GRID = "grid"

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


EXPECTED_RANK = {
    SpatialLayout.NONE: 3,
    SpatialLayout.NODES: 4,
    SpatialLayout.GRID: 5,
}


@dataclass(frozen=True)
class SpatialStructure:
    """Base class for immutable spatial metadata.

    Masks use the TFTS convention: ``True`` means valid. Structure objects are
    public Python contracts; model boundaries receive only their tensor fields.
    """

    @property
    def layout(self) -> SpatialLayout:
        raise NotImplementedError

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    def set_size(self) -> int:
        result = 1
        for dimension in self.spatial_shape:
            result *= dimension
        return result

    def validate(self, values: tf.Tensor) -> None:
        raise NotImplementedError

    def to_tensor_dict(self, prefix: str = "structure.") -> Dict[str, tf.Tensor]:
        """Return tensor fields only, suitable for tracing and model calls."""
        return {
            prefix + field.name: tf.convert_to_tensor(value)
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None and tf.is_tensor(value)
        }


@dataclass(frozen=True)
class GraphStructure(SpatialStructure):
    """Topology for node-set values shaped ``[B, T, N, C]``."""

    num_nodes: int
    adjacency: Optional[tf.Tensor] = None
    edge_index: Optional[tf.Tensor] = None
    edge_weight: Optional[tf.Tensor] = None
    edge_features: Optional[tf.Tensor] = None
    node_features: Optional[tf.Tensor] = None
    node_coordinates: Optional[tf.Tensor] = None
    node_mask: Optional[tf.Tensor] = None
    node_ids: Tuple[str, ...] = ()

    def __post_init__(self):
        if int(self.num_nodes) <= 0:
            raise ValueError("num_nodes must be positive")
        for name in (
            "adjacency",
            "edge_index",
            "edge_weight",
            "edge_features",
            "node_features",
            "node_coordinates",
            "node_mask",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tf.convert_to_tensor(value))
        object.__setattr__(self, "node_ids", tuple(str(value) for value in self.node_ids))
        if self.node_ids and len(self.node_ids) != self.num_nodes:
            raise ValueError("node_ids length must equal num_nodes")

    @property
    def layout(self):
        return SpatialLayout.NODES

    @property
    def spatial_shape(self):
        return (int(self.num_nodes),)

    @property
    def is_shared(self):
        return self.adjacency is not None and self.adjacency.shape.rank == 2

    @property
    def is_dynamic(self):
        return self.adjacency is not None and self.adjacency.shape.rank == 4

    def validate(self, values):
        if values.shape.rank != 4:
            raise ValueError("nodes layout expects past_values shaped [batch, time, node, feature]")
        nodes = values.shape[2]
        if nodes is not None and nodes != self.num_nodes:
            raise ValueError(f"past_values has {nodes} nodes but structure declares {self.num_nodes}")
        if self.adjacency is not None:
            if self.adjacency.shape.rank not in (2, 3, 4):
                raise ValueError("adjacency must be [N,N], [B,N,N], or [B,T,N,N]")
            for dimension in self.adjacency.shape[-2:]:
                if dimension is not None and dimension != self.num_nodes:
                    raise ValueError("adjacency trailing dimensions must equal num_nodes")
        if self.edge_index is not None:
            if self.edge_index.dtype not in (tf.int32, tf.int64):
                raise ValueError("edge_index must use an integer dtype")
            if self.edge_index.shape.rank not in (2, 3) or self.edge_index.shape[-2] != 2:
                raise ValueError("edge_index must be [2,E] or [B,2,E]")
        if self.node_mask is not None and self.node_mask.shape.rank not in (1, 2, 3):
            raise ValueError("node_mask must be [N], [B,N], or [B,T,N]")
        for name in ("node_features", "node_coordinates"):
            value = getattr(self, name)
            if value is not None and value.shape.rank not in (2, 3):
                raise ValueError(f"{name} must be [N,F] or [B,N,F]")

    @classmethod
    def from_tensor_dict(cls, values: Mapping[str, tf.Tensor], prefix="structure.graph."):
        kwargs = {key[len(prefix) :]: value for key, value in values.items() if key.startswith(prefix)}
        if "num_nodes" not in kwargs:
            candidate = kwargs.get("adjacency", kwargs.get("node_mask"))
            if candidate is None or candidate.shape[-1] is None:
                raise ValueError("num_nodes cannot be inferred from structure tensors")
            kwargs["num_nodes"] = int(candidate.shape[-1])
        return cls(**kwargs)


@dataclass(frozen=True)
class GridStructure(SpatialStructure):
    """Geometry for grid values shaped ``[B, T, height, width, C]``."""

    height: int
    width: int
    coordinates: Optional[tf.Tensor] = None
    valid_mask: Optional[tf.Tensor] = None
    periodic_axes: Tuple[str, ...] = ()

    def __post_init__(self):
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise ValueError("height and width must be positive")
        for name in ("coordinates", "valid_mask"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tf.convert_to_tensor(value))
        axes = tuple(str(axis) for axis in self.periodic_axes)
        if any(axis not in {"height", "width"} for axis in axes):
            raise ValueError("periodic_axes entries must be 'height' or 'width'")
        object.__setattr__(self, "periodic_axes", axes)

    @property
    def layout(self):
        return SpatialLayout.GRID

    @property
    def spatial_shape(self):
        return (int(self.height), int(self.width))

    def validate(self, values):
        if values.shape.rank != 5:
            raise ValueError("grid layout expects past_values shaped [batch, time, height, width, feature]")
        for axis, expected, name in ((2, self.height, "height"), (3, self.width, "width")):
            actual = values.shape[axis]
            if actual is not None and actual != expected:
                raise ValueError(f"past_values {name}={actual} but structure declares {expected}")
        if self.valid_mask is not None and self.valid_mask.shape.rank not in (2, 3):
            raise ValueError("valid_mask must be [height,width] or [batch,height,width]")

    @classmethod
    def from_tensor_dict(cls, values: Mapping[str, tf.Tensor], prefix="structure.grid."):
        kwargs = {key[len(prefix) :]: value for key, value in values.items() if key.startswith(prefix)}
        mask = kwargs.get("valid_mask")
        coordinates = kwargs.get("coordinates")
        candidate = mask if mask is not None else coordinates
        if candidate is None:
            raise ValueError("grid dimensions cannot be inferred from structure tensors")
        kwargs.setdefault("height", int(candidate.shape[-3] if coordinates is not None else candidate.shape[-2]))
        kwargs.setdefault("width", int(candidate.shape[-2] if coordinates is not None else candidate.shape[-1]))
        return cls(**kwargs)
