"""AutoModel to choose different models"""

from collections import OrderedDict
import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.layers.auto_task import AnomalyHead, ClassificationHead, PredictionHead, SegmentationHead

from .auto_config import AutoConfig
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
    ]
)


class AutoModel(BaseModel):
    """tftf auto model
    input tensor: [batch_size, sequence_length, num_features]
    output tensor: [batch_size, predict_sequence_length, num_labels]
    """

    def __init__(
        self,
        model,
        config,
    ):
        super().__init__()
        self.model = model
        self.config = config

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
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
        return self.model(x, return_dict=return_dict)

    @classmethod
    def from_config(cls, config, predict_sequence_length=1, task="prediction"):
        model_name = config.model_type
        class_name = MODEL_MAPPING_NAMES[model_name]
        module = importlib.import_module(f".{model_name}", "tfts.models")
        model = getattr(module, class_name)(predict_sequence_length, config=config)
        return cls(model, config)

    @classmethod
    def from_pretrained(cls, config, predict_sequence_length, weights_path):
        instance = cls.from_config(config, predict_sequence_length)
        instance.built = True
        instance.load_weights(weights_path)
        return instance


class AutoModelForPrediction(AutoModel):
    """tfts model for prediction"""

    def __call__(self, x):

        model_output = super().__call__(x)

        if self.config.skip_connect_circle:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            model_output = model_output + x_mean
        if self.config.skip_connect_mean:
            x_mean = tf.tile(tf.reduce_mean(x[..., 0:1], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            model_output = model_output + x_mean
        return model_output


class AutoModelForClassification(AutoModel):
    """tfts model for classification"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
    ):
        return super().__call__(x)


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


class AutoModelForSegmentation(AutoModel):
    """tfts model for time series segmentation"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
    ):
        model_output = self.model(x)
        return model_output


class AutoModelForUncertainty(AutoModel):
    """tfts model for time series uncertainty prediction"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
    ):
        model_output = self.model(x)
        return model_output
