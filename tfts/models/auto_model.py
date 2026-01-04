"""AutoModel to choose different models"""

import collections
import importlib
import json
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tfts.losses.loss import MultiQuantileLoss
from tfts.models.base import BaseConfig, BaseModel
from tfts.tasks.auto_task import (
    AnomalyHead,
    ClassificationHead,
    ClassificationOutput,
    GaussianHead,
    PredictionHead,
    PredictionOutput,
)

from ..constants import CONFIG_NAME, TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME
from .auto_config import AutoConfig

logger = logging.getLogger(__name__)


MODEL_MAPPING_NAMES = collections.OrderedDict(
    [
        ("seq2seq", "Seq2seq"),
        ("rnn", "RNN"),
        ("wavenet", "WaveNet"),
        ("tcn", "TCN"),
        ("transformer", "Transformer"),
        ("bert", "Bert"),
        ("informer", "Informer"),
        ("autoformer", "AutoFormer"),
        ("tft", "TFTransformer"),
        ("unet", "Unet"),
        ("nbeats", "NBeats"),
        ("dlinear", "DLinear"),
        ("rwkv", "RWKV"),
        ("patch_tst", "PatchTST"),
        ("deep_ar", "DeepAR"),
    ]
)


class AutoModel(BaseModel):
    """tfts auto model
    input tensor: [batch_size, sequence_length, num_features]
    output tensor: [batch_size, predict_sequence_length, num_labels]
    """

    def __init__(self, model, config):
        super().__init__(config=config)
        self.model = model
        self.config = config

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """auto_model callable

        Parameters
        ----------
        x : tf.data.Dataset, np.array
            model inputs
        return_dict: bool
            if return output a dict

        Returns
        -------
        tf.Tensor
            model output
        """
        if isinstance(x, (list, tuple)):
            if len(x[0].shape) != 3:
                raise ValueError(
                    f"Expected input dimension is 3 (batch_size, train_sequence_length, num_features), "
                    f"but got {len(x[0].shape)}"
                )
        if isinstance(self.model, BaseModel):
            return self.model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)
        else:  # after build_model, the model will become tf.keras.Model
            return self.model(x)

    @classmethod
    def from_config(cls, config, predict_sequence_length: int = 1):
        model_name = config.model_type
        class_name = MODEL_MAPPING_NAMES[model_name]
        module = importlib.import_module(f".{model_name}", "tfts.models")
        model = getattr(module, class_name)(config=config, predict_sequence_length=predict_sequence_length)
        return cls(model, config)

    @classmethod
    def from_pretrained(cls, weights_dir: Union[str, os.PathLike], predict_sequence_length: int = 1):
        config_path = os.path.join(weights_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        except Exception as e:
            raise OSError(f"Error loading config file from {config_path}. Original error: {e}")

        try:
            model_type = config_dict.get("model_type")
            if model_type is None:
                raise ValueError("Missing `model_type` in config.")

            # Dynamically get the correct Config subclass
            config = AutoConfig.for_model(model_type)
            config.update(config_dict)  # update with the saved values

            # Build model and load weights
            model = cls.from_config(config, predict_sequence_length=predict_sequence_length)
            if isinstance(config.input_shape, dict):
                inputs = {k: tf.keras.layers.Input(shape=v, name=k) for k, v in config.input_shape.items()}
            elif isinstance(config.input_shape[0], (list, tuple)):
                inputs = [
                    tf.keras.layers.Input(shape=shape, name=f"input_{i}") for i, shape in enumerate(config.input_shape)
                ]
            else:
                inputs = tf.keras.layers.Input(shape=config.input_shape, name="input")

            model.build_model(inputs)
            model.model.load_weights(os.path.join(weights_dir, TF2_WEIGHTS_NAME))
            return cls(model, config)
        except Exception as e:
            raise OSError(
                f"Error loading model weights from {weights_dir}. "
                f"Ensure weights were saved using model.save_weights(...). Original error: {e}"
            )

    def get_config(self):
        return self.config.to_dict() if self.config else {}


class AutoModelForPrediction(AutoModel):
    """tfts model for prediction"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        model_output = self.model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)

        if self.config.skip_connect_circle:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            model_output = model_output + x_mean
        elif self.config.skip_connect_mean:
            x_mean = tf.tile(tf.reduce_mean(x[..., 0:1], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            model_output = model_output + x_mean
        return model_output


class AutoModelForClassification(BaseModel):
    """tfts model for classification"""

    def __init__(self, model, config):
        super(AutoModelForClassification, self).__init__()
        self.model = model
        self.config = config
        self.head = ClassificationHead(num_labels=config.num_labels)
        self.keras_model: Optional[tf.keras.Model] = None

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if self.keras_model is not None:
            logits = self.keras_model(x)
        else:
            model_output = self.model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)
            logits = self.head(model_output)
        return logits

    @classmethod
    def from_config(cls, config, num_labels: int = 1):
        config.num_labels = num_labels
        model_name = config.model_type
        class_name = MODEL_MAPPING_NAMES[model_name]
        module = importlib.import_module(f".{model_name}", "tfts.models")
        model = getattr(module, class_name)(config=config)
        return cls(model, config)

    def build_model(self, inputs: tf.keras.layers.Input):
        model_output = self.model(inputs)
        logits = self.head(model_output)  # Apply the head layer to the output
        self.keras_model = tf.keras.Model(inputs, logits)  # Create a complete model
        return self.keras_model


class AutoModelForAnomaly(BaseModel):
    """tfts model for anomaly detection"""

    def __init__(self, model, config):
        super().__init__(config=config)
        self.model = model
        self.config = config
        self.head = AnomalyHead(config.train_sequence_length)

    def detect(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        labels=None,
    ):
        model_output = self.model(x)
        dist = self.head(model_output, labels)
        return dist

    @classmethod
    def from_pretrained(cls, weights_dir: Union[str, os.PathLike]):
        model_path = os.path.join(weights_dir, "model.h5")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Load model from {weights_dir}")
        config_path = os.path.join(weights_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = BaseConfig.from_json(config_path)  # Load config from JSON
        return cls(model, config)

    @classmethod
    def from_config(cls, config):
        model_name = config.model_type
        class_name = MODEL_MAPPING_NAMES[model_name]
        module = importlib.import_module(f".{model_name}", "tfts.models")
        model = getattr(module, class_name)(config=config)
        return cls(model, config)


class AutoModelForSegmentation(BaseModel):
    """tfts model for time series segmentation"""

    def __init__(self, model, config):
        super().__init__(config=config)
        self.model = model
        self.config = config

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        model_output = self.model(x, return_dict=return_dict)
        return model_output

    @classmethod
    def from_config(cls, config):
        model_name = config.model_type
        class_name = MODEL_MAPPING_NAMES[model_name]
        module = importlib.import_module(f".{model_name}", "tfts.models")
        model = getattr(module, class_name)(config=config)
        return cls(model, config)


class AutoModelForUncertainty(BaseModel):
    """tfts model for time series uncertainty probabilistic forecasting model, not a point forecasting"""

    def __init__(self, model, config):
        super().__init__(config=config)
        self.model = model
        self.config = config

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        model_output = self.model(x, return_dict=return_dict)
        return model_output

    @classmethod
    def from_config(cls, config):
        model_name = config.model_type
        class_name = MODEL_MAPPING_NAMES[model_name]
        module = importlib.import_module(f".{model_name}", "tfts.models")
        model = getattr(module, class_name)(config=config)
        return cls(model, config)


class AutoModelForQuantile(BaseModel):
    """tfts model for quantile forecasting"""

    def __init__(self, model, config):
        super(AutoModelForQuantile, self).__init__()
        self.model = model
        self.config = config
        self.quantiles = getattr(config, "quantiles", [0.1, 0.5, 0.9])
        self.num_labels = getattr(config, "num_labels", 1)
        self.head = Dense(self.num_labels * len(self.quantiles))
        self.keras_model: Optional[tf.keras.Model] = None

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], List[np.ndarray]],
        output_hidden_states: Optional[bool] = True,
        **kwargs,
    ):
        if self.keras_model is not None:
            return self.keras_model(x)

        model_output = self.model(x, output_hidden_states=output_hidden_states)
        return self.head(model_output)

    @classmethod
    def from_config(cls, config, quantiles: List[float] = [0.1, 0.5, 0.9]):
        config.quantiles = quantiles
        model_name = config.model_type
        class_name = MODEL_MAPPING_NAMES[model_name]
        module = importlib.import_module(f".{model_name}", "tfts.models")
        model = getattr(module, class_name)(config=config)
        return cls(model, config)

    def build_model(self, inputs):
        model_output = self.model(inputs)
        outputs = self.head(model_output)
        self.keras_model = tf.keras.Model(inputs, outputs)
        return self.keras_model

    def compile_model(self, optimizer="adam"):
        """Helper to compile with the correct loss"""
        loss_fn = MultiQuantileLoss(quantiles=self.quantiles)
        self.keras_model.compile(optimizer=optimizer, loss=loss_fn)
