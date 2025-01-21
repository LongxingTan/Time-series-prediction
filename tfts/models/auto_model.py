"""AutoModel to choose different models"""

from collections import OrderedDict
import importlib
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.tasks.auto_task import AnomalyHead, ClassificationHead

from .base import BaseConfig, BaseModel

logger = logging.getLogger(__name__)


MODEL_MAPPING_NAMES = OrderedDict(
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
    ]
)


class AutoModel(BaseModel):
    """tftf auto model
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
        """automodel callable

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
        return self.model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)

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

        # config = BaseConfig.from_json(config_path)  # Load config from JSON
        # model = cls.from_config(config, predict_sequence_length=predict_sequence_length)
        # model.load_weights(os.path.join(weights_dir, "weights.h5"))  # Load weights
        model = tf.keras.models.load_model(weights_dir)
        return model

    def save_pretrained(self):
        pass


class AutoModelForPrediction(AutoModel):
    """tfts model for prediction"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
    ):

        model_output = super().__call__(x, return_dict=return_dict)

        if self.config.skip_connect_circle:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            model_output = model_output + x_mean
        if self.config.skip_connect_mean:
            x_mean = tf.tile(tf.reduce_mean(x[..., 0:1], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            model_output = model_output + x_mean
        return model_output


class AutoModelForClassification(AutoModel):
    """tfts model for classification"""

    def __init__(self, model, config):
        super().__init__(model, config)
        self.head = ClassificationHead(num_labels=config.num_labels)

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        model_output = self.model(x, output_hidden_states=True)
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


class AutoModelForAnomaly(AutoModel):
    """tfts model for anomaly detection"""

    def __init__(self, model, config):
        super().__init__(model, config)
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
        model = tf.keras.models.load_model(weights_dir)
        logger.info(f"Load model from {weights_dir}")
        config_path = os.path.join(weights_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = BaseConfig.from_json(config_path)  # Load config from JSON
        return cls(model, config)


class AutoModelForSegmentation(AutoModel):
    """tfts model for time series segmentation"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        model_output = self.model(x, return_dict=return_dict)
        return model_output


class AutoModelForUncertainty(AutoModel):
    """tfts model for time series uncertainty prediction"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        model_output = self.model(x, return_dict=return_dict)
        return model_output
