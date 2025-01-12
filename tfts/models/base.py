"""Base class for config and model"""

from abc import ABC, abstractmethod
import collections
import json
import logging
import os
from typing import Any, Dict, Union

import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for tfts model."""

    @classmethod
    def from_config(cls, config: "BaseConfig", predict_sequence_length: int = 1):
        return cls(predict_sequence_length=predict_sequence_length, config=config)

    @classmethod
    def from_pretrained(cls, weights_dir: str, predict_sequence_length: int = 1):
        config_path = os.path.join(weights_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = BaseConfig.from_json(config_path)  # Load config from JSON
        model = cls.from_config(
            config, predict_sequence_length=predict_sequence_length
        )  # Use from_config to create the model
        weights_dir = os.path.join(weights_dir, "model.h5")
        model.load_pretrained_weights(weights_dir)  # Load weights
        return model

    def build_model(self, inputs):
        outputs = self.model(inputs)
        # to handles the Keras symbolic tensors for tf2.3.1
        return tf.keras.Model([inputs], [outputs])

    def load_pretrained_weights(self, weights_dir: str):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f"Weights file not found at {weights_dir}")
        self.model = tf.keras.models.load_model(weights_dir)

    def save_weights(self, weights_dir: str):
        self.model.save_weights(weights_dir)
        logging.info(f"Model weights successfully saved in {weights_dir}")

    def save_model(self, weights_dir: str):
        self.model.save(weights_dir)
        logging.info(f"Protobuf model successfully saved in {weights_dir}")

    def summary(self):
        if hasattr(self, "model"):
            self.model.summary()
        else:
            raise RuntimeError("Model has not been built yet. Please build the model first.")


class BaseConfig(ABC):
    """Base class for tfts config."""

    attribute_map: Dict[str, str] = {}

    def __init__(self, **kwargs):
        self.update(kwargs)

    def __setattr__(self, key, value):
        mapped_key = self.attribute_map.get(key, key)
        super().__setattr__(mapped_key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def update(self, config_dict):
        for key, value in config_dict.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dict__ if not key.startswith("_")}

    def to_json(self, json_file: Union[str, os.PathLike]):
        config_dict = self.to_dict()
        with open(json_file, "w", encoding="utf-8") as file:
            json.dump(config_dict, file, indent=4)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        with open(pretrained_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_pretrained(self, save_path):
        with open(save_path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def __str__(self):
        """Convert config to string representation in dictionary format"""
        return str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat
