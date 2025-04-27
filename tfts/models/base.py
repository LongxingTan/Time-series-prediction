"""Base class for config and model"""

from abc import ABC, abstractmethod
import collections
import json
import logging
import os
from typing import Any, Dict, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Lambda

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Bert model for time series forecasting.

    This model implements a transformer-based architecture (BERT) adapted for time series data.
    It processes time series inputs through a transformer encoder and produces predictions
    for future time steps.

    Parameters
    ----------
    predict_sequence_length : int, optional
        Number of future time steps to predict, by default 1
    config : BertConfig, optional
        Configuration parameters for the model, by default None
    """

    def __init__(self, predict_sequence_length: int = 1, config: Optional["BaseConfig"] = None):
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.model = None  # Model should be defined later (may not be directly used in all subclasses)

    @classmethod
    def from_pretrained(cls, weights_dir: Union[str, os.PathLike], predict_sequence_length: int = 1):
        config_path = os.path.join(weights_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        model = tf.keras.models.load_model(weights_dir)
        return model

    def build_model(self, inputs: tf.keras.layers.Input) -> tf.keras.Model:
        # only accept the inputs parameters after built
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

    def _prepare_3d_inputs(self, inputs, ignore_decoder_inputs=True):
        """
        Prepares 3D inputs for model processing by extracting and formatting features from various input types.

        Args:
            inputs: Input data that can be a tuple/list, dictionary, or tensor.
                - If tuple/list: Expected to be [x, encoder_feature, decoder_feature]
                - If dictionary: Expected to have keys "x", "encoder_feature", and "decoder_feature"
                - If tensor: Used directly as both x and encoder_feature

        Returns:
            tuple: (x, encoder_feature, decoder_feature) properly formatted for model processing
        """
        logger.debug(f"Preparing 3D inputs with shape: {inputs.shape}")

        decoder_feature = None
        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
            if "decoder_feature" in inputs:
                decoder_feature = inputs["decoder_feature"]
        else:
            encoder_feature = x = inputs
            if not ignore_decoder_inputs:
                decoder_feature = Lambda(
                    lambda encoder_feature: tf.cast(
                        tf.tile(
                            tf.reshape(tf.range(self.predict_sequence_length), (1, self.predict_sequence_length, 1)),
                            (tf.shape(encoder_feature)[0], 1, 1),
                        ),
                        tf.float32,
                    )
                )(encoder_feature)
        return x, encoder_feature, decoder_feature

    def save_weights(self, weights_path: str):
        if weights_path.endswith(".h5"):
            # User passed a full filepath
            weights_file = weights_path
            config_file = weights_path.replace(".h5", ".config.json")
            os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        else:
            # User passed a directory
            os.makedirs(weights_path, exist_ok=True)
            weights_file = os.path.join(weights_path, "model.weights.h5")
            config_file = os.path.join(weights_path, "config.json")

        self.model.save_weights(weights_file)
        self.config.to_json(config_file)
        logger.info(f"Model weights successfully saved in {weights_file}")

    def save_model(self, weights_dir: str):
        self.model.save(weights_dir)
        logger.info(f"Protobuf model successfully saved in {weights_dir}")

    def summary(self):
        if hasattr(self, "model") and self.model is not None:
            self.model.summary()
        else:
            raise RuntimeError("Model has not been built yet. Please build the model first.")


class BaseConfig(ABC):
    """Base class for tfts config."""

    attribute_map: Dict[str, str] = {}
    model_type: str

    def __init__(self, **kwargs):
        self.update(kwargs)

    def __setattr__(self, key: str, value):
        mapped_key = self.attribute_map.get(key, key)
        super().__setattr__(mapped_key, value)

    def __getattribute__(self, key: str):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def update(self, config_dict: Dict[str, Any]):
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
    def from_pretrained(cls, pretrained_path: Union[str, os.PathLike]):
        with open(pretrained_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_pretrained(self, save_path: Union[str, os.PathLike]):
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
