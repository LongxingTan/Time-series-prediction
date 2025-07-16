"""Base class for config and model"""

from abc import ABC, abstractmethod
import collections
import json
import logging
import os
from typing import Any, Dict, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Lambda

from ..constants import CONFIG_NAME, TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME

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

    def build_model(self, inputs: tf.keras.layers.Input) -> tf.keras.Model:
        if hasattr(self, "config"):
            if isinstance(inputs, dict):
                self.config.input_shape = {k: tuple(v.shape[1:]) for k, v in inputs.items()}
            elif isinstance(inputs, (list, tuple)):
                # multiple input
                self.config.input_shape = [tuple(v.shape[1:]) for v in inputs]
            else:
                self.config.input_shape = tuple(inputs.shape[1:])

        if self.model is not None:
            # only accept the inputs parameters after built
            outputs = self.model(inputs)
            # to handles the Keras symbolic tensors for tf2.3.1, use []
            self.model = tf.keras.Model(inputs, outputs)
            return self.model
        else:
            outputs = self(inputs)
            return tf.keras.Model(inputs, outputs)

    def to_model(self):
        inputs = tf.keras.Input(shape=(self.config.input_shape))
        return self.build_model(inputs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

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

        decoder_feature = None
        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = Concatenate(axis=-1)([x, encoder_feature])
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = Concatenate(axis=-1)([x, encoder_feature])
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

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        max_shard_size: Union[int, str] = "8GB",
        safe_serialization: bool = False,
    ):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        self.config.architectures = [self.__class__.__name__[2:]]
        self.config.save_pretrained(save_directory)

        weights_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)  # Or the appropriate extension

        try:
            self.model.save_weights(weights_file)
            logging.info(f"Model weights successfully saved in {weights_file}")
        except Exception as e:
            logging.error(f"Failed to save model weights to {weights_file}: {e}")
            return

    def save_weights(self, weights_path: str):
        if weights_path.endswith(".h5"):
            # User passed a full filepath
            weights_file = weights_path
            config_file = weights_path.replace(".h5", ".config.json")
            os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        else:
            # User passed a directory
            os.makedirs(weights_path, exist_ok=True)
            weights_file = os.path.join(weights_path, TF2_WEIGHTS_NAME)
            config_file = os.path.join(weights_path, CONFIG_NAME)

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

    def get_config(self):
        return self.config.to_dict() if self.config else {}

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        output_dim = self.config.hidden_size if self.config and hasattr(self.config, "hidden_size") else 1
        return (batch_size, self.predict_sequence_length, output_dim)


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
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
    ):
        with open(pretrained_model_name_or_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        try:
            with open(output_config_file, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Model config successfully saved in {output_config_file}")
        except Exception as e:
            logger.warning(f"Error saving model config to {output_config_file}: {e}")

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
