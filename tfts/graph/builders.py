"""Build graph structures from explicit data, coordinates, or grids."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from tfts.contracts.structure import GraphStructure


def from_adjacency(matrix, node_ids: Optional[Sequence[str]] = None) -> GraphStructure:
    """Build a graph from a finite, non-negative square adjacency matrix."""
    adjacency = np.asarray(matrix, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.all(np.isfinite(adjacency)) or np.any(adjacency < 0):
        raise ValueError("adjacency must contain finite non-negative values")
    return GraphStructure(num_nodes=adjacency.shape[0], adjacency=adjacency, node_ids=tuple(node_ids or ()))


def from_knn(
    coords,
    k: int,
    sigma=None,
    symmetric=True,
    metric="euclidean",
    kernel="gaussian",
    epsilon=0.0,
) -> GraphStructure:
    """Build a k-nearest-neighbor graph from ``[node, coordinate]`` values.

    ``kernel='gaussian'`` assigns ``exp(-(distance / sigma) ** 2)`` weights;
    when ``sigma`` is omitted it is the standard deviation of retained
    distances. ``kernel='binary'`` assigns weight one. Weights at or below
    ``epsilon`` are removed. Distances are computed densely, so this builder is
    intended for modest graph sizes.
    """
    coordinates = _coordinates(coords)
    nodes = coordinates.shape[0]
    if not 0 < int(k) < nodes:
        raise ValueError("k must be between 1 and num_nodes - 1")
    distances = _pairwise_distance(coordinates, metric)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, : int(k)]
    rows = np.repeat(np.arange(nodes), int(k))
    columns = neighbors.reshape(-1)
    retained = distances[rows, columns]
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    adjacency[rows, columns] = _kernel_weights(retained, sigma, kernel)
    adjacency[adjacency <= _epsilon(epsilon)] = 0.0
    if symmetric:
        adjacency = np.maximum(adjacency, adjacency.T)
    return GraphStructure(num_nodes=nodes, adjacency=adjacency)


def from_radius(coords, radius: float, sigma=None, metric="euclidean", kernel="gaussian", epsilon=0.0):
    """Build a radius graph using the same configurable kernel as :func:`from_knn`.

    Pairwise distances are materialized as an ``N x N`` array. Gaussian
    ``sigma`` defaults to the standard deviation of distances within ``radius``.
    Weights at or below ``epsilon`` are removed.
    """
    coordinates = _coordinates(coords)
    if radius <= 0:
        raise ValueError("radius must be positive")
    distances = _pairwise_distance(coordinates, metric)
    keep = (distances <= radius) & (distances > 0)
    adjacency = np.zeros_like(distances, dtype=np.float32)
    adjacency[keep] = _kernel_weights(distances[keep], sigma, kernel)
    adjacency[adjacency <= _epsilon(epsilon)] = 0.0
    return GraphStructure(num_nodes=coordinates.shape[0], adjacency=adjacency)


def from_correlation(values, threshold=0.3) -> GraphStructure:
    """Build an absolute-correlation graph from ``[time, node]`` observations."""
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("values must be [time, node]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    correlation = np.nan_to_num(np.corrcoef(observations, rowvar=False), nan=0.0)
    adjacency = np.where(np.abs(correlation) >= threshold, np.abs(correlation), 0.0)
    np.fill_diagonal(adjacency, 0.0)
    return from_adjacency(adjacency)


def from_grid(height: int, width: int, connectivity=4, periodic_axes=()) -> GraphStructure:
    """Build an unweighted 4- or 8-neighbor graph for a rectangular grid.

    Nodes use row-major order. ``periodic_axes`` may contain ``'height'`` or
    ``'width'`` to wrap neighbors across that boundary. The dense construction
    uses ``O((height * width) ** 2)`` memory and is intended for small grids.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    periodic_axes = frozenset(periodic_axes)
    if not periodic_axes <= {"height", "width"}:
        raise ValueError("periodic_axes entries must be 'height' or 'width'")
    adjacency = np.zeros((height * width, height * width), dtype=np.float32)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for row in range(height):
        for column in range(width):
            source = row * width + column
            for delta_row, delta_column in offsets:
                target_row, target_column = row + delta_row, column + delta_column
                if "height" in periodic_axes:
                    target_row %= height
                if "width" in periodic_axes:
                    target_column %= width
                if 0 <= target_row < height and 0 <= target_column < width:
                    target = target_row * width + target_column
                    if target != source:
                        adjacency[source, target] = 1.0
    return GraphStructure(num_nodes=height * width, adjacency=adjacency)


def _coordinates(values):
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("coords must be [node, coordinate] with at least two nodes")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coords must contain only finite values")
    return coordinates.astype(np.float32)


def _kernel_weights(distances, sigma, kernel):
    if kernel == "binary":
        return np.ones_like(distances, dtype=np.float32)
    if kernel != "gaussian":
        raise ValueError("kernel must be 'gaussian' or 'binary'")
    scale = _distance_scale(distances, sigma)
    return np.exp(-np.square(distances / scale)).astype(np.float32)


def _epsilon(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("epsilon must be finite and non-negative")
    return value


def _distance_scale(distances, sigma):
    if sigma is not None:
        scale = float(sigma)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("sigma must be finite and positive")
        return scale
    scale = float(np.std(distances)) if distances.size else 1.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _pairwise_distance(coordinates, metric):
    if metric == "euclidean":
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        return np.sqrt(np.sum(np.square(differences), axis=-1))
    if metric == "haversine":
        if coordinates.shape[1] != 2:
            raise ValueError("haversine coordinates must contain latitude and longitude")
        radians = np.deg2rad(coordinates)
        lat1, lon1 = radians[:, 0][:, None], radians[:, 1][:, None]
        lat2, lon2 = lat1.T, lon1.T
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    raise ValueError("metric must be 'euclidean' or 'haversine'")
