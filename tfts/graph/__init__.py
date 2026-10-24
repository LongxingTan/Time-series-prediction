"""Public exports, loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "from_adjacency": ("tfts.graph.builders", "from_adjacency"),
    "from_correlation": ("tfts.graph.builders", "from_correlation"),
    "from_grid": ("tfts.graph.builders", "from_grid"),
    "from_knn": ("tfts.graph.builders", "from_knn"),
    "from_radius": ("tfts.graph.builders", "from_radius"),
    "add_self_loops": ("tfts.graph.transforms", "add_self_loops"),
    "laplacian": ("tfts.graph.transforms", "laplacian"),
    "random_walk_normalize": ("tfts.graph.transforms", "random_walk_normalize"),
    "symmetric_normalize": ("tfts.graph.transforms", "symmetric_normalize"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
