from abc import ABC, abstractmethod
import collections
import json
import logging
import os
from typing import Any, Dict, Union

import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, config, predict_length):
        model = cls()
        model.build_model(config, predict_length)
        return model

    @classmethod
    def from_pretrained(cls, name: str):
        model = cls()
        model.load_pretrained_weights(name)
        return model

    def build_model(self, inputs):
        outputs = self.model(inputs)
        return tf.keras.Model([inputs], [outputs])  # to handles the Keras symbolic tensors for tf2.3.1

    # def build_model(self, config, predict_length):
    #     inputs = tf.keras.Input(shape=(config["input_shape"],))
    #     x = inputs
    #     for layer_units in config["layers"]:
    #         x = tf.keras.layers.Dense(layer_units, activation="relu")(x)
    #     outputs = tf.keras.layers.Dense(predict_length, activation="softmax")(x)
    #     self.model = tf.keras.Model(inputs, outputs)
    #     self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def load_pretrained_weights(self, name: str):
        self.model = tf.keras.models.load_model(name)


class BaseConfig(ABC):
    attribute_map: Dict[str, str] = {}

    def __init__(self, **kwargs):
        self.update(kwargs)

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

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
        # output_dict = {}
        # for key, value in self.__dict__.items():
        #     output_dict[key] = value
        # return flatten_dict(output_dict)
        return {key: getattr(self, key) for key in self.__dict__}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
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
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


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
