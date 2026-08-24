"""AutoConfig to set up models custom config"""

import importlib
from typing import Dict

from .base import BaseConfig
from .registry import MODEL_REGISTRY

CONFIG_MAPPING_NAMES = {name: metadata["config_class"] for name, metadata in MODEL_REGISTRY.items()}


class AutoConfig(BaseConfig):
    """AutoConfig for tfts model"""

    def __init__(self, **kwargs: Dict[str, object]):
        super().__init__(**kwargs)

    @classmethod
    def for_model(cls, model_name: str):

        if model_name in CONFIG_MAPPING_NAMES:
            class_name = CONFIG_MAPPING_NAMES[model_name]
            module = importlib.import_module(f".{model_name}", "tfts.models")
            mapping = getattr(module, class_name)

            return mapping()
        raise ValueError(
            f"Unrecognized model: {model_name}. Should contain one of {', '.join(CONFIG_MAPPING_NAMES.keys())}"
        )

    def __call__(self, model_name: str):
        return self.for_model(model_name)
