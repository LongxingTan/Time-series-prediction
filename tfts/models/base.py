"""Shared model contracts and the base model implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from functools import wraps
import json
import logging
import os
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Lambda

from tfts.config import BaseConfig
from tfts.graph.structure import SpatialArrangement, TopologyInput

from ..constants import CONFIG_NAME, TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME
from ..layers.util_layer import CreateDecoderFeature

logger = logging.getLogger(__name__)


class OutputPort(str, Enum):
    SEQUENCE = "sequence"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    POOLED = "pooled"
    NATIVE_FORECAST = "native_forecast"
    DISTRIBUTION = "distribution"


@dataclass(frozen=True)
class ModelInputSpec:
    """Feature roles and datatypes one model implementation can consume."""

    accepted_roles: FrozenSet[str] = frozenset({"observed_past"})
    supports_categorical: bool = False
    supports_static: bool = False
    supports_multivariate_target: bool = True
    requires_target_history: bool = False
    requires_matching_time_features: bool = False
    accepted_dtypes_by_role: Mapping[str, FrozenSet[str]] = field(default_factory=dict)
    arrangement: SpatialArrangement = SpatialArrangement.NONE
    accepted_topologies: FrozenSet[TopologyInput] = frozenset({TopologyInput.NONE})
    supports_dynamic_graph: bool = False
    supports_node_mask: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_roles",
            frozenset(getattr(role, "value", role) for role in self.accepted_roles),
        )
        unknown_roles = set(self.accepted_roles) - {"observed_past", "known_future", "static"}
        if unknown_roles:
            raise ValueError(f"unknown feature roles: {sorted(unknown_roles)}")
        if "static" in self.accepted_roles and not self.supports_static:
            raise ValueError("accepted static features require supports_static=True")
        accepted_dtypes = {
            getattr(role, "value", role): frozenset(getattr(dtype, "value", dtype) for dtype in dtypes)
            for role, dtypes in self.accepted_dtypes_by_role.items()
        }
        unknown_roles = set(accepted_dtypes) - set(self.accepted_roles)
        if unknown_roles:
            raise ValueError(f"dtype constraints reference unaccepted roles: {sorted(unknown_roles)}")
        unknown_dtypes = set().union(*accepted_dtypes.values()) - {"real", "categorical", "boolean"}
        if unknown_dtypes:
            raise ValueError(f"unknown feature dtypes: {sorted(unknown_dtypes)}")
        if not self.supports_categorical and any("categorical" in dtypes for dtypes in accepted_dtypes.values()):
            raise ValueError("categorical dtype constraints require supports_categorical=True")
        object.__setattr__(self, "accepted_dtypes_by_role", MappingProxyType(accepted_dtypes))
        arrangement = SpatialArrangement.normalize(self.arrangement)
        topologies = frozenset(TopologyInput.normalize(topology) for topology in self.accepted_topologies)
        if not topologies:
            raise ValueError("accepted_topologies cannot be empty")
        if (self.supports_dynamic_graph or self.supports_node_mask) and (arrangement != SpatialArrangement.SET):
            raise ValueError("graph capabilities require the set arrangement")
        object.__setattr__(self, "arrangement", arrangement)
        object.__setattr__(self, "accepted_topologies", topologies)


@dataclass(frozen=True)
class BackboneCapabilities:
    """Output ports and accepted inputs used by task composition."""

    output_ports: FrozenSet[OutputPort] = frozenset({OutputPort.NATIVE_FORECAST})
    input_spec: ModelInputSpec = ModelInputSpec()

    def has_port(self, port: OutputPort) -> bool:
        if port == OutputPort.SEQUENCE and OutputPort.TEMPORAL_SEQUENCE in self.output_ports:
            return True
        return port in self.output_ports


@dataclass(frozen=True)
class ModelOutput:
    """Immutable base for every task and backbone output."""

    loss: Optional[tf.Tensor] = None

    def replace(self, **changes):
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name) for field in fields(self) if getattr(self, field.name) is not None
        }


@dataclass(frozen=True)
class BackboneOutput(ModelOutput):
    sequence_output: Optional[tf.Tensor] = None
    pooled_output: Optional[tf.Tensor] = None
    native_forecast: Optional[tf.Tensor] = None
    distribution_params: Optional[Mapping[str, tf.Tensor]] = None
    state: Any = None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
    attentions: Optional[Tuple[tf.Tensor, ...]] = None


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
        nested_input = isinstance(input_shape, dict) or (
            isinstance(input_shape, (list, tuple))
            and input_shape
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
        )
        if self.config is not None:
            if isinstance(input_shape, dict):
                self.config.input_shape = {key: tuple(shape[1:]) for key, shape in input_shape.items()}
            elif nested_input:
                self.config.input_shape = [tuple(shape[1:]) for shape in input_shape]
            else:
                self.config.input_shape = tuple(tf.TensorShape(input_shape)[1:])

        # These are subclassed models whose child layers are built by the first
        # real call.  Keras 2's ``Model.build`` instead calls the model with
        # symbolic placeholders: nested shapes fail in placeholder creation,
        # while manually-built RNN cells inherit a callable caching device that
        # eager ``tf.device`` rejects.  Record the shape and mark only the outer
        # layer built; the concrete call already in progress builds children.
        self._build_input_shape = input_shape
        tf.keras.layers.Layer.build(self, input_shape)

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
        config.validate_checkpoint_config(base_config.to_dict())
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

    def _validate_target_shape(self, input_shape):
        """Validate models that forecast the leading target channels at build time."""
        value_shape, _ = self._input_shapes(input_shape)
        if value_shape[-1] is not None and value_shape[-1] < self.config.target_dim:
            raise ValueError("Input has fewer channels than target_dim")

    def _split_targets(self, x):
        """Return leading target channels and remaining observed covariates.

        Models using this convention call _validate_target_shape from build().
        """
        return x[..., : self.config.target_dim], x[..., self.config.target_dim :]

    @staticmethod
    def _input_shapes(input_shape):
        """Return ``(value_shape, encoder_shape)`` for supported input nests."""
        if isinstance(input_shape, dict):
            value_shape = tf.TensorShape(input_shape["x"])
            feature_shape = tf.TensorShape(input_shape["encoder_feature"])
            return value_shape, value_shape[:-1].concatenate(
                None if value_shape[-1] is None or feature_shape[-1] is None else value_shape[-1] + feature_shape[-1]
            )
        if (
            isinstance(input_shape, (list, tuple))
            and input_shape
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
        ):
            value_shape = tf.TensorShape(input_shape[0])
            feature_shape = tf.TensorShape(input_shape[1])
            return value_shape, value_shape[:-1].concatenate(
                None if value_shape[-1] is None or feature_shape[-1] is None else value_shape[-1] + feature_shape[-1]
            )
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
            restored_config.validate_checkpoint_config(model_config)
            restored_config.update(model_config)
            config["config"] = restored_config
        return cls(**config)
