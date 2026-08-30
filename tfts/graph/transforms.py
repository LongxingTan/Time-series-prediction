"""Composable adjacency transforms."""

from dataclasses import replace

import numpy as np

from tfts.contracts.structure import GraphStructure


def _adjacency(structure: GraphStructure):
    if structure.adjacency is None:
        raise ValueError("the graph has no dense adjacency")
    adjacency = np.asarray(structure.adjacency)
    if adjacency.ndim != 2:
        raise ValueError("this transform requires shared [N,N] adjacency")
    return adjacency.astype(np.float32)


def add_self_loops(structure, weight=1.0):
    adjacency = _adjacency(structure).copy()
    np.fill_diagonal(adjacency, float(weight))
    return replace(structure, adjacency=adjacency)


def symmetric_normalize(structure, epsilon=1e-12):
    adjacency = _adjacency(structure)
    degree = np.sum(adjacency, axis=1)
    inverse_root = np.where(degree > epsilon, degree**-0.5, 0.0)
    normalized = inverse_root[:, None] * adjacency * inverse_root[None, :]
    return replace(structure, adjacency=normalized.astype(np.float32))


def random_walk_normalize(structure, epsilon=1e-12):
    adjacency = _adjacency(structure)
    degree = np.sum(adjacency, axis=1, keepdims=True)
    normalized = np.divide(adjacency, degree, out=np.zeros_like(adjacency), where=degree > epsilon)
    return replace(structure, adjacency=normalized.astype(np.float32))


def laplacian(structure, normalized=True):
    """Return a Laplacian matrix; unlike adjacency transforms, this returns an array."""
    adjacency = _adjacency(structure)
    if normalized:
        normalized_adjacency = np.asarray(symmetric_normalize(structure).adjacency)
        return np.eye(structure.num_nodes, dtype=np.float32) - normalized_adjacency
    return np.diag(np.sum(adjacency, axis=1)) - adjacency
