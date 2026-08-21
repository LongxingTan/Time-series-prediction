"""Base class for config and model"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
import json
import logging
import os
from typing import Any, Dict, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Lambda

from ..constants import CONFIG_NAME, TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME
from ..layers.util_layer import CreateDecoderFeature

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base model for time series forecasting.

    Abstract base class that all tfts models inherit from.
    Subclasses must implement __call__ and can optionally override build_model.

    Parameters
    ----------
    predict_sequence_length : int, optional
        Number of future time steps to predict, by default 1
    config : BaseConfig, optional
        Configuration parameters for the model, by default None
    """

    def __init__(self, predict_sequence_length: int = 1, config: Optional["BaseConfig"] = None):
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        if isinstance(self.config, dict):
            self.config["predict_sequence_length"] = predict_sequence_length
        elif self.config is not None:
            self.config.predict_sequence_length = predict_sequence_length
        self.model = None  # Model should be defined later (may not be directly used in all subclasses)
        # ``core_model`` keeps the object that carries any generation hooks (e.g. DeepAR's
        # ``generate``); ``keras_model`` is the compiled teacher-forced graph. Both are
        # needed because ``build_model`` replaces ``self.model`` with a ``tf.keras.Model``
        # that no longer exposes subclass methods like ``generate``.
        self.core_model: Optional["BaseModel"] = None
        self.keras_model: Optional[tf.keras.Model] = None

    def build_model(self, inputs: tf.keras.layers.Input) -> tf.keras.Model:
        # Retain the object that carries generation hooks across the ``self.model``
        # replacement below (first call only; never clobber an already-captured core).
        if self.core_model is None:
            candidate = getattr(self, "model", None)
            self.core_model = candidate if isinstance(candidate, BaseModel) else self

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
            self.keras_model = tf.keras.Model(inputs, outputs)
            self.model = self.keras_model
            return self.model
        else:
            outputs = self(inputs)
            self.keras_model = tf.keras.Model(inputs, outputs)
            self.model = self.keras_model
            return self.model

    def _keras_model_for_saving(self) -> tf.keras.Model:
        if isinstance(self.model, tf.keras.Model):
            return self.model
        if isinstance(self.model, BaseModel) and isinstance(self.model.model, tf.keras.Model):
            return self.model.model
        raise ValueError(
            "Model weights cannot be saved before the model is built. "
            "Call `build_model(...)` or train the model before saving weights."
        )

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
                decoder_feature = CreateDecoderFeature(self.predict_sequence_length)(encoder_feature)
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

        keras_model = self._keras_model_for_saving()

        os.makedirs(save_directory, exist_ok=True)
        # Use model_type from config if available, otherwise derive from class name
        name = self.__class__.__name__
        architecture = getattr(self.config, "model_type", name)
        self.config.architectures = [architecture]
        self.config.save_pretrained(save_directory)

        weights_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)  # Or the appropriate extension

        keras_model.save_weights(weights_file)
        logging.info(f"Model weights successfully saved in {weights_file}")

    def save_weights(self, weights_path: str):
        if weights_path.endswith(".h5"):
            # User passed a full filepath
            weights_file = weights_path
            config_file = weights_path.replace(".h5", ".config.json")
            weights_dir = os.path.dirname(weights_file)
            if weights_dir:
                os.makedirs(weights_dir, exist_ok=True)
        else:
            # User passed a directory
            os.makedirs(weights_path, exist_ok=True)
            weights_file = os.path.join(weights_path, TF2_WEIGHTS_NAME)
            config_file = os.path.join(weights_path, CONFIG_NAME)

        self._keras_model_for_saving().save_weights(weights_file)
        self.config.to_json(config_file)
        logger.info(f"Model weights successfully saved in {weights_file}")

    def save_model(self, weights_dir: str):
        self._keras_model_for_saving().save(weights_dir)
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
