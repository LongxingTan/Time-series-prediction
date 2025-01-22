"""AutoModel to choose different models"""

import importlib
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.models.base import MODEL_MAPPING_NAMES, BaseConfig, BaseModel
from tfts.tasks.auto_task import AnomalyHead, ClassificationHead

logger = logging.getLogger(__name__)


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


class AutoModelForPrediction(BaseModel):
    """tfts model for prediction"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        model_output = self.model(x, return_dict=return_dict)

        if self.config.skip_connect_circle:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            model_output = model_output + x_mean
        if self.config.skip_connect_mean:
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

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if hasattr(self.model, "call"):
            model_output = self.model(x)
        else:
            model_output = self.model(x, output_hidden_states=output_hidden_states)
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
        model = tf.keras.models.load_model(weights_dir)
        logger.info(f"Load model from {weights_dir}")
        config_path = os.path.join(weights_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = BaseConfig.from_json(config_path)  # Load config from JSON
        return cls(model, config)


class AutoModelForSegmentation(BaseModel):
    """tfts model for time series segmentation"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        model_output = self.model(x, return_dict=return_dict)
        return model_output


class AutoModelForUncertainty(BaseModel):
    """tfts model for time series uncertainty prediction"""

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        model_output = self.model(x, return_dict=return_dict)
        return model_output
