"""Serializable architecture configuration."""

from abc import ABC
from collections.abc import Mapping
from functools import wraps
import json
import logging
import os
from typing import Any, Dict, Optional, Union

from .constants import CONFIG_NAME

logger = logging.getLogger(__name__)


class BaseConfig(ABC):
    """Base class for tfts config."""

    attribute_map: Dict[str, str] = {}
    model_type: str

    @staticmethod
    def validate_checkpoint_config(values):
        """Validate persisted architecture versions before constructing layers."""

    def __init_subclass__(cls, **kwargs):
        """Validate concrete configs once their complete constructor returns."""
        super().__init_subclass__(**kwargs)
        original_init = cls.__dict__.get("__init__")
        if original_init is None or getattr(original_init, "_tfts_validated_init", False):
            return

        @wraps(original_init)
        def validated_init(self, *args, **init_kwargs):
            depth = getattr(self, "_config_init_depth", 0)
            object.__setattr__(self, "_config_init_depth", depth + 1)
            try:
                original_init(self, *args, **init_kwargs)
            finally:
                object.__setattr__(self, "_config_init_depth", depth)
            if depth == 0:
                self.validate()

        validated_init._tfts_validated_init = True
        cls.__init__ = validated_init

    def __init__(self, **kwargs):
        self.update(kwargs)

    @classmethod
    def _attribute_mapping(cls) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for base in reversed(cls.__mro__):
            mapping.update(base.__dict__.get("attribute_map", {}))
        return mapping

    def __setattr__(self, key: str, value):
        mapped_key = type(self)._attribute_mapping().get(key, key)
        super().__setattr__(mapped_key, value)

    def __getattr__(self, key: str):
        mapped_key = type(self)._attribute_mapping().get(key)
        if mapped_key is None or mapped_key == key:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {key!r}")
        return object.__getattribute__(self, mapped_key)

    def update(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err
        if getattr(self, "_config_init_depth", 0) == 0:
            self.validate()

    def validate(self) -> None:
        """Validate common fields and any model-specific ``__post_init__``."""
        for field in ("hidden_size", "num_layers", "num_attention_heads", "intermediate_size", "target_dim"):
            value = getattr(self, field, None)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{field} must be a positive integer, got {value!r}")

        dropout = getattr(self, "dropout", None)
        if dropout is not None and not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

        model_validator = type(self).__dict__.get("__post_init__")
        if model_validator is not None:
            model_validator(self)

    def to_dict(self):
        instance_attributes = {key: getattr(self, key) for key in self.__dict__ if not key.startswith("_")}

        if hasattr(self, "model_type"):
            instance_attributes["model_type"] = self.model_type
        return instance_attributes

    def to_json(self, json_file: Union[str, os.PathLike]):
        config_dict = self.to_dict()
        with open(json_file, "w", encoding="utf-8") as file:
            json.dump(config_dict, file, indent=4)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        config = cls()
        config.update(config_dict)
        return config

    @classmethod
    def from_json(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        return cls.from_dict(config_dict)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
    ):
        path = os.fspath(pretrained_model_name_or_path)
        if os.path.isdir(path):
            path = os.path.join(path, CONFIG_NAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found at {path}")

        # These arguments are retained for API compatibility. Remote artifact
        # downloads are not supported by BaseConfig yet.
        del cache_dir, force_download
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) must be a directory")
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json(output_config_file)
        logger.info(f"Model config successfully saved in {output_config_file}")

    def __str__(self):
        """Convert config to string representation in dictionary format"""
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})


class CommonConfig(BaseConfig):
    """Shared, canonical configuration vocabulary for model configs.

    Legacy names remain readable and writable through ``attribute_map`` while
    serialization uses the canonical names.
    """

    attribute_map = {
        "d_model": "hidden_size",
        "d_ff": "intermediate_size",
        "e_layers": "num_layers",
        "ffn_intermediate_size": "intermediate_size",
        "hidden_dropout_prob": "dropout",
        "dropout_rate": "dropout",
    }

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        target_dim: int = 1,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.target_dim = target_dim
        super().__init__(**kwargs)


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat
