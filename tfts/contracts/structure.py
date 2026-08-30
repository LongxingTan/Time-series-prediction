"""Spatial layouts and topology carried by :class:`TimeSeriesBatch`."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar, Dict, Mapping, Optional, Tuple, Type

import tensorflow as tf


class SpatialArrangement(str, Enum):
    """Physical arrangement encoded by the rank of time-series values."""

    NONE = "none"
    SET = "set"
    GRID = "grid"

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


class TopologyInput(str, Enum):
    """Relational information consumed by a spatial model."""

    NONE = "none"
    DENSE_ADJACENCY = "dense_adjacency"

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


EXPECTED_RANK = {
    SpatialArrangement.NONE: 3,
    SpatialArrangement.SET: 4,
    SpatialArrangement.GRID: 5,
}

ARRANGEMENT_BY_RANK = {rank: arrangement for arrangement, rank in EXPECTED_RANK.items()}


@dataclass(frozen=True)
class SpatialStructure:
    """Base class for immutable spatial metadata.

    Masks use the TFTS convention: ``True`` means valid. Structure objects are
    public Python contracts; model boundaries receive only their tensor fields.
    """

    SHARED_FIELD_RANKS: ClassVar[Mapping[str, Tuple[int, ...]]] = {}
    TENSOR_PREFIX: ClassVar[str]
    _TYPES: ClassVar[Dict[str, Type["SpatialStructure"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prefix = getattr(cls, "TENSOR_PREFIX", None)
        if prefix:
            SpatialStructure._TYPES[prefix] = cls

    @property
    def arrangement(self) -> SpatialArrangement:
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

    def to_tensor_dict(self) -> Dict[str, tf.Tensor]:
        """Return tensor fields only, suitable for tracing and model calls."""
        return {
            self.TENSOR_PREFIX + field.name: tf.convert_to_tensor(value)
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None and tf.is_tensor(value)
        }

    def split_tensor_dict(self):
        """Return ``(per_sample, shared)`` tensors using type-owned semantics."""
        per_sample, shared = {}, {}
        for name, value in self.to_tensor_dict().items():
            field_name = name[len(self.TENSOR_PREFIX) :]
            ranks = self.SHARED_FIELD_RANKS.get(field_name, ())
            target = shared if value.shape.rank in ranks else per_sample
            target[name] = value
        return per_sample, shared

    @classmethod
    def from_tensor_dict(cls, values: Mapping[str, tf.Tensor]):
        """Dispatch a flat tensor mapping to its declared structure type."""
        matches = [
            (prefix, structure_type)
            for prefix, structure_type in cls._TYPES.items()
            if any(key.startswith(prefix) for key in values)
        ]
        if len(matches) != 1:
            raise ValueError("Structure tensors must use exactly one recognized prefix")
        _, structure_type = matches[0]
        return structure_type.from_tensor_dict(values)

    @property
    def topology_inputs(self):
        return frozenset()


@dataclass(frozen=True)
class GraphStructure(SpatialStructure):
    """Topology for node-set values shaped ``[B, T, N, C]``."""

    TENSOR_PREFIX: ClassVar[str] = "structure.graph."
    SHARED_FIELD_RANKS: ClassVar[Mapping[str, Tuple[int, ...]]] = {
        "num_nodes": (0,),
        "adjacency": (2,),
        "node_features": (2,),
        "node_mask": (1,),
        "node_ids": (1,),
    }

    num_nodes: int
    adjacency: Optional[tf.Tensor] = None
    node_features: Optional[tf.Tensor] = None
    node_mask: Optional[tf.Tensor] = None
    node_ids: Tuple[str, ...] = ()

    def __post_init__(self):
        if int(self.num_nodes) <= 0:
            raise ValueError("num_nodes must be positive")
        for name in (
            "adjacency",
            "node_features",
            "node_mask",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tf.convert_to_tensor(value))
        object.__setattr__(self, "node_ids", tuple(str(value) for value in self.node_ids))
        if self.node_ids and len(self.node_ids) != self.num_nodes:
            raise ValueError("node_ids length must equal num_nodes")

    @property
    def arrangement(self):
        return SpatialArrangement.SET

    @property
    def spatial_shape(self):
        return (int(self.num_nodes),)

    @property
    def is_shared(self):
        return self.adjacency is not None and self.adjacency.shape.rank == 2

    @property
    def is_dynamic(self):
        return self.adjacency is not None and self.adjacency.shape.rank == 4

    @property
    def topology_inputs(self):
        inputs = set()
        if self.adjacency is not None:
            inputs.add(TopologyInput.DENSE_ADJACENCY)
        return frozenset(inputs)

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
        if self.node_mask is not None and self.node_mask.shape.rank not in (1, 2, 3):
            raise ValueError("node_mask must be [N], [B,N], or [B,T,N]")
        for name in ("node_features",):
            value = getattr(self, name)
            if value is not None and value.shape.rank not in (2, 3):
                raise ValueError(f"{name} must be [N,F] or [B,N,F]")

    def to_tensor_dict(self):
        values = super().to_tensor_dict()
        values[self.TENSOR_PREFIX + "num_nodes"] = tf.convert_to_tensor(self.num_nodes, tf.int32)
        if self.node_ids:
            values[self.TENSOR_PREFIX + "node_ids"] = tf.convert_to_tensor(self.node_ids, tf.string)
        return values

    @classmethod
    def from_tensor_dict(cls, values: Mapping[str, tf.Tensor]):
        prefix = cls.TENSOR_PREFIX
        kwargs = {key[len(prefix) :]: value for key, value in values.items() if key.startswith(prefix)}
        if "num_nodes" in kwargs:
            stored_num_nodes = tf.get_static_value(kwargs["num_nodes"])
            if stored_num_nodes is not None:
                kwargs["num_nodes"] = int(stored_num_nodes)
            else:
                kwargs.pop("num_nodes")
        if "num_nodes" not in kwargs:
            candidate = kwargs.get("adjacency", kwargs.get("node_mask", kwargs.get("node_ids")))
            if candidate is None or candidate.shape[-1] is None:
                raise ValueError("num_nodes cannot be inferred from structure tensors")
            kwargs["num_nodes"] = int(candidate.shape[-1])
        if "node_ids" in kwargs:
            node_ids = tf.get_static_value(kwargs["node_ids"])
            if node_ids is None:
                kwargs["node_ids"] = ("",) * int(kwargs["node_ids"].shape[-1])
            else:
                kwargs["node_ids"] = tuple(
                    value.decode() if isinstance(value, bytes) else str(value) for value in node_ids
                )
        return cls(**kwargs)
