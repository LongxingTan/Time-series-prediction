"""AutoConfig to set up models custom config"""

from typing import Dict

from .base import BaseConfig
from .registry import RegistryFieldView, get_config_class

CONFIG_MAPPING_NAMES = RegistryFieldView("config_class")


class AutoConfig(BaseConfig):
    """AutoConfig for tfts model"""

    def __init__(self, **kwargs: Dict[str, object]):
        super().__init__(**kwargs)

    @classmethod
    def for_model(cls, model_name: str):

        if model_name in CONFIG_MAPPING_NAMES:
            return get_config_class(model_name)()
        raise ValueError(
            f"Unrecognized model: {model_name}. Should contain one of {', '.join(CONFIG_MAPPING_NAMES.keys())}"
        )

    def __call__(self, model_name: str):
        return self.for_model(model_name)
