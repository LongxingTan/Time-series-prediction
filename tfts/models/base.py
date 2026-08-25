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


class BaseModel(tf.keras.Model, ABC):
    """Base model for time series forecasting.

    This is a regular :class:`tf.keras.Model`: subclasses define layers in
    ``__init__`` (or ``build``) and implement ``call``.

    Parameters
    ----------
    predict_sequence_length : int, optional
        Number of future time steps to predict, by default 1
    config : BaseConfig, optional
        Configuration parameters for the model, by default None
    """

    def __init__(
        self,
        predict_sequence_length: int = 1,
        config: Optional["BaseConfig"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        if isinstance(self.config, dict):
            self.config["predict_sequence_length"] = predict_sequence_length
        elif self.config is not None:
            self.config.predict_sequence_length = predict_sequence_length

    def build(self, input_shape):
        """Record the serializable input contract and use Keras' build state."""
        if self.config is not None:
            if isinstance(input_shape, dict):
                self.config.input_shape = {key: tuple(shape[1:]) for key, shape in input_shape.items()}
            elif (
                isinstance(input_shape, (list, tuple))
                and input_shape
                and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
            ):
                self.config.input_shape = [tuple(shape[1:]) for shape in input_shape]
            else:
                self.config.input_shape = tuple(tf.TensorShape(input_shape)[1:])
        super().build(input_shape)

    def build_from_config(self, config):
        """Create child variables before Keras restores their saved values.

        Most TFTS models create child-layer variables on their first forward
        pass. Calling ``build(input_shape)`` alone would mark the outer model
        built while leaving those children empty, so use one shape-only dummy
        forward when loading a full ``.keras`` archive.
        """

        def make_dummy(shape):
            shape = tf.TensorShape(shape).as_list()
            return tf.zeros([dimension if dimension is not None else 1 for dimension in shape], dtype=self.dtype)

        def make_inputs(shape):
            if isinstance(shape, dict):
                return {key: make_inputs(value) for key, value in shape.items()}
            if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple, tf.TensorShape)):
                return [make_inputs(value) for value in shape]
            return make_dummy(shape)

        input_shape = config.get("input_shape")
        if input_shape is None:
            return
        self(make_inputs(input_shape))

    def build_model(self, inputs) -> tf.keras.Model:
        """Compatibility shim: build this model and return it.

        New code should call the model directly or use ``model.build(shape)``.
        Unlike the legacy implementation, this never creates a second model or
        mutates an attribute from a backbone into a Functional model.
        """
        if hasattr(self, "config"):
            if isinstance(inputs, dict):
                self.config.input_shape = {k: tuple(v.shape[1:]) for k, v in inputs.items()}
            elif isinstance(inputs, (list, tuple)):
                # multiple input
                self.config.input_shape = [tuple(v.shape[1:]) for v in inputs]
            else:
                self.config.input_shape = tuple(inputs.shape[1:])

        self(inputs)
        return self

    def to_model(self):
        inputs = tf.keras.Input(shape=(self.config.input_shape))
        return self.build_model(inputs)

    def predict(self, x, **kwargs):
        return super().predict(x, **kwargs)

    def load_pretrained_weights(self, weights_dir: str):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f"Weights file not found at {weights_dir}")
        loaded = tf.keras.models.load_model(weights_dir)
        self.set_weights(loaded.get_weights())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        predict_sequence_length: Optional[int] = None,
    ):
        """Rebuild a model from its TFTS config and load its saved weights.

        Models returned by this method remain regular TFTS model instances and
        can be passed directly to a trainer for further fine-tuning.
        """
        from .auto_config import AutoConfig

        model_dir = os.fspath(pretrained_model_name_or_path)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Pretrained model directory not found at {model_dir}")

        base_config = BaseConfig.from_pretrained(model_dir)
        model_type = getattr(base_config, "model_type", None)
        if model_type is None:
            raise ValueError(f"Missing 'model_type' in {os.path.join(model_dir, CONFIG_NAME)}")

        config = AutoConfig.for_model(model_type)
        config.update(base_config.to_dict())
        prediction_length = predict_sequence_length or getattr(config, "predict_sequence_length", 1)
        model = cls(config=config, predict_sequence_length=prediction_length)

        input_shape = getattr(config, "input_shape", None)
        if input_shape is None:
            raise ValueError("Missing `input_shape` in the pretrained model config")
        if isinstance(input_shape, dict):
            inputs = {key: tf.keras.Input(shape=shape, name=key) for key, shape in input_shape.items()}
        elif isinstance(input_shape[0], (list, tuple)):
            inputs = [tf.keras.Input(shape=shape, name=f"input_{i}") for i, shape in enumerate(input_shape)]
        else:
            inputs = tf.keras.Input(shape=input_shape, name="input")

        model.build_model(inputs)
        weights_path = os.path.join(model_dir, TF2_WEIGHTS_NAME)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        model.load_weights(weights_path)
        return model

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

    @staticmethod
    def _input_shapes(input_shape):
        """Return ``(value_shape, encoder_shape)`` for supported input nests."""
        if isinstance(input_shape, dict):
            value_shape = tf.TensorShape(input_shape["x"])
            feature_shape = tf.TensorShape(input_shape["encoder_feature"])
            return value_shape, value_shape[:-1].concatenate(value_shape[-1] + feature_shape[-1])
        if (
            isinstance(input_shape, (list, tuple))
            and input_shape
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
        ):
            value_shape = tf.TensorShape(input_shape[0])
            feature_shape = tf.TensorShape(input_shape[1])
            return value_shape, value_shape[:-1].concatenate(value_shape[-1] + feature_shape[-1])
        shape = tf.TensorShape(input_shape)
        return shape, shape

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
        # Use model_type from config if available, otherwise derive from class name
        name = self.__class__.__name__
        architecture = getattr(self.config, "model_type", name)
        self.config.architectures = [architecture]
        self.config.predict_sequence_length = self.predict_sequence_length
        self.config.save_pretrained(save_directory)

        weights_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)  # Or the appropriate extension

        super().save_weights(weights_file)
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

        super().save_weights(weights_file)
        self.config.to_json(config_file)
        logger.info(f"Model weights successfully saved in {weights_file}")

    def save_model(self, weights_dir: str):
        self.save(weights_dir)
        logger.info(f"Protobuf model successfully saved in {weights_dir}")

    def summary(self, *args, **kwargs):
        return super().summary(*args, **kwargs)

    def get_config(self):
        """Return the constructor configuration used by Keras serialization."""
        model_config = self.config.to_dict() if self.config is not None else None
        model_type = model_config.get("model_type") if model_config is not None else None
        return {
            "model_type": model_type,
            "predict_sequence_length": self.predict_sequence_length,
            "config": model_config,
        }

    @classmethod
    def from_config(cls, config):
        """Rebuild a TFTS model from a Keras object configuration."""
        from .auto_config import AutoConfig

        config = config.copy()
        model_type = config.pop("model_type", None)
        model_config = config.pop("config", None)
        if model_config is not None:
            model_type = model_config.get("model_type", model_type)
            if model_type is None:
                raise ValueError("Serialized TFTS model config is missing `model_type`.")
            restored_config = AutoConfig.for_model(model_type)
            restored_config.update(model_config)
            config["config"] = restored_config
        return cls(**config)


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
