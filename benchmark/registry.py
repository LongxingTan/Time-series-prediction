"""Registry for datasets and models in the TFTS benchmark system."""

import logging
from typing import Callable, Dict, List, Optional, Type, Union

from benchmark.base import Dataset
from tfts.models.auto_config import CONFIG_MAPPING_NAMES
from tfts.models.auto_model import MODEL_MAPPING_NAMES

logger = logging.getLogger(__name__)


class _Registry:
    """Internal registry base class."""

    def __init__(self):
        self._items: Dict[str, type] = {}

    def register(self, name: str, item: type) -> None:
        if name in self._items:
            logger.warning("Overwriting existing registration: %s", name)
        self._items[name] = item
        logger.debug("Registered: %s", name)

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(f"'{name}' not found. Available: {list(self._items.keys())}")
        return self._items[name]

    def list_items(self) -> Dict[str, type]:
        return dict(self._items)

    def __contains__(self, item: str) -> bool:
        return item in self._items


class DatasetRegistry(_Registry):
    """Registry for benchmark datasets.

    Usage::

        from tfts.benchmark.registry import DatasetRegistry
        from tfts.benchmark.datasets import SineDataset

        registry = DatasetRegistry()
        registry.register("sine", SineDataset)

        ds_cls = registry.get("sine")
        ds = ds_cls()
        print(ds.name)
    """

    def __init__(self):
        super().__init__()
        self._lazy: Dict[str, Callable[[], Dataset]] = {}

    def register_lazy(self, name: str, factory: Callable[[], Dataset]) -> None:
        """Register a lazy factory so datasets are only instantiated on demand."""
        self._lazy[name] = factory

    def get(self, name: str) -> Type[Dataset]:
        if name in self._lazy:
            # Return a tiny wrapper that calls the factory
            factory = self._lazy[name]

            class _LazyDataset(Dataset):
                """Lazy-loaded dataset wrapper."""

                name = name

                def prepare_data(self, **kwargs):
                    return factory().prepare_data(**kwargs)

                def get_train_valid_split(self, **kwargs):
                    return factory().get_train_valid_split(**kwargs)

            return _LazyDataset
        return super().get(name)

    def list_datasets(self) -> List[str]:
        return sorted(set(list(self._items.keys())) | set(self._lazy.keys()))


class ModelRegistry:
    """Registry for models available in the benchmark.

    Wraps the existing :mod:`tfts.models` mapping.
    """

    def __init__(self):
        # Synchronized with tfts.models.auto_model.MODEL_MAPPING_NAMES
        self._models: Dict[str, str] = dict(MODEL_MAPPING_NAMES)

    @property
    def available_models(self) -> List[str]:
        return list(self._models.keys())

    def get(self, name: str) -> str:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {self.available_models}")
        return self._models[name]

    def register(self, name: str, class_name: str) -> None:
        """Register a custom model."""
        self._models[name] = class_name
        logger.debug("Registered model: %s -> %s", name, class_name)

    def __contains__(self, item: str) -> bool:
        return item in self._models
