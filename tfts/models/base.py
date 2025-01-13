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

    def __init__(self, config=None, predict_sequence_length=1):
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.model = None  # Model should be defined later

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
        model.load_weights(weights_dir, os.path.join(weights_dir, "model.h5"))  # Load weights
        return model

    def build_model(self, inputs):
        outputs = self.model(inputs)
        # to handles the Keras symbolic tensors for tf2.3.1
        self.model = tf.keras.Model([inputs], [outputs])
        return self.model

    def to_model(self):
        inputs = tf.keras.Input(shape=(self.config.input_shape))
        return self.build_model(inputs)

    def load_pretrained_weights(self, weights_dir: str):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f"Weights file not found at {weights_dir}")
        self.model = tf.keras.models.load_model(weights_dir)
        # self.model = model.load_weights(os.path.join(weights_dir, "weights.h5"))

    def save_weights(self, weights_dir: str):
        if os.path.isdir(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)
            parent_dir = os.path.dirname(weights_dir)
            weights_dir = os.path.join(parent_dir, "weights.h5")
            config_dir = os.path.join(parent_dir, "config.json")
        else:
            parent_dir = os.path.normpath(weights_dir)  # Get the parent directory
            os.makedirs(parent_dir, exist_ok=True)
            weights_dir = os.path.join(parent_dir, "weights.h5")  # Save weights in the same directory
            config_dir = os.path.join(parent_dir, "config.json")

        self.model.save_weights(weights_dir)
        self.config.to_json(config_dir)
        logger.info(f"Model weights successfully saved in {weights_dir}")

    def save_model(self, weights_dir: str):
        self.model.save(weights_dir)
        logger.info(f"Protobuf model successfully saved in {weights_dir}")

    def summary(self):
        if hasattr(self, "model"):
            self.model.summary()
        else:
            raise RuntimeError("Model has not been built yet. Please build the model first.")


class BaseConfig(ABC):
    """Base class for tfts config."""

    attribute_map: Dict[str, str] = {}
    model_type: str

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
