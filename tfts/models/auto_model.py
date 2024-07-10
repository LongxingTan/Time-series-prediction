"""AutoModel to choose different models"""

from collections import OrderedDict
import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import tensorflow as tf

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
        model_name: str,
        predict_length: int,
        config=None,
    ):
        super(AutoModel, self).__init__()
        class_name = MODEL_MAPPING_NAMES[model_name]
        module = importlib.import_module(f".{model_name}", "tfts.models")
        if not config:
            config = AutoConfig.for_model(model_name)
        self.model = getattr(module, class_name)(predict_length, config=config)

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

        Returns
        -------
        tf.Tensor
            model output
        """
        # if isinstance(x, (list, tuple)):
        #     assert len(x[0].shape) == 3, "The expected inputs dimension is 3, while get {}".format(len(x[0].shape))
        return self.model(x, return_dict=return_dict)


class AutoModelForPrediction(BaseModel):
    """tfts model for prediction"""

    def __init__(self, model_name):
        super(AutoModelForPrediction, self).__init__()
        self.model = AutoModel(model_name)

    def __call__(self, x):

        model_output = self.model(x)

        if self.config["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            model_output = model_output + x_mean
        if self.config["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x[..., 0:1], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            model_output = model_output + x_mean
        return model_output


class AutoModelForClassification(BaseModel):
    """tfts model for classification"""

    def __init__(self, model_name):
        super(AutoModelForClassification, self).__init__()
        self.model = AutoModel(model_name)

    def __call__(
        self,
    ):
        return


class AutoModelForAnomaly(BaseModel):
    """tfts model for anomaly detection"""

    def __init__(self, model_name):
        super(AutoModelForAnomaly, self).__init__()
        self.model = AutoModel(model_name)

    def __call__(self, *args, **kwargs):
        return


class AutoModelForSegmentation(BaseModel):
    """tfts model for time series segmentation"""

    def __init__(self, model_name):
        super(AutoModelForSegmentation, self).__init__()
        self.model = AutoModel(model_name)

    def __call__(self, *args, **kwargs):
        return
