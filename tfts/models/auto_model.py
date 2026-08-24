"""AutoModel to choose different models"""

import json
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tfts.losses.loss import MultiQuantileLoss
from tfts.models.base import BaseModel
from tfts.tasks.auto_task import AnomalyHead, ClassificationHead, apply_prediction_residual

from ..constants import TF2_WEIGHTS_NAME
from .auto_config import AutoConfig
from .registry import RegistryFieldView, get_model_class

logger = logging.getLogger(__name__)


MODEL_MAPPING_NAMES = RegistryFieldView("class_name")


class AutoModel(BaseModel):
    """tfts auto model
    input tensor: [batch_size, sequence_length, num_features]
    output tensor: [batch_size, predict_sequence_length, num_labels]
    """

    def __init__(self, model, config, predict_sequence_length: Optional[int] = None):
        predict_sequence_length = predict_sequence_length or getattr(model, "predict_sequence_length", 1)
        super().__init__(predict_sequence_length=predict_sequence_length, config=config)
        self.model = model
        # retain the wrapped model that owns generation hooks (DeepAR's ``generate``),
        # since ``build_model`` later overwrites ``self.model`` with a ``tf.keras.Model``.
        self.core_model = model
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
    def from_config(
        cls,
        config,
        predict_sequence_length: int = 1,
        task: Optional[str] = None,
        num_labels: int = 1,
        quantiles: Optional[List[float]] = None,
    ):
        """Create a model through the single task-aware factory.

        ``task=None`` preserves the historical ``AutoModel`` behavior. Task
        aliases delegate here, so model resolution and construction live in
        one place.
        """
        model_name = config.model_type
        if model_name not in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"Unrecognized model: {model_name}. Should contain one of {', '.join(MODEL_MAPPING_NAMES.keys())}"
            )

        normalized_task = task.lower().replace("-", "_") if task else None
        task_models = {
            "prediction": AutoModelForPrediction,
            "classification": AutoModelForClassification,
            "anomaly": AutoModelForAnomaly,
            "segmentation": AutoModelForSegmentation,
            "uncertainty": AutoModelForUncertainty,
            "quantile": AutoModelForQuantile,
        }
        if normalized_task not in {None, "model", *task_models}:
            raise ValueError(f"Unknown task {task!r}. Available: {sorted(task_models)}")
        if normalized_task == "classification":
            config.num_labels = num_labels
        elif normalized_task == "quantile":
            config.quantiles = list(quantiles or (0.1, 0.5, 0.9))
            config.num_labels = num_labels

        model = get_model_class(model_name)(config=config, predict_sequence_length=predict_sequence_length)
        if normalized_task is None or normalized_task == "model":
            return cls(model, config, predict_sequence_length=predict_sequence_length)

        wrapper = task_models[normalized_task]
        if issubclass(wrapper, AutoModel):
            return wrapper(model, config, predict_sequence_length=predict_sequence_length)
        return wrapper(model, config)

    @classmethod
    def from_pretrained(cls, weights_dir: Union[str, os.PathLike], predict_sequence_length: Optional[int] = None):
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
            predict_sequence_length = predict_sequence_length or getattr(config, "predict_sequence_length", 1)

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
            return model
        except Exception as e:
            raise OSError(
                f"Error loading model weights from {weights_dir}. "
                f"Ensure weights were saved using model.save_weights(...). Original error: {e}"
            )

    def get_config(self):
        return self.config.to_dict() if self.config else {}

    def generate(self, inputs, generation_config=None, **kwargs):
        """Delegate generation to the wrapped core model (opt-in feature).

        Only models that implement ``generate`` (e.g. DeepAR via ``AutoregressiveGenerationMixin``)
        support this directly. Subclasses can also add the legacy ``GenerationMixin``
        after ``AutoModel`` in their MRO.
        """
        core = getattr(self, "core_model", None) or self.model
        generate_fn = getattr(core, "generate", None)
        if generate_fn is not None:
            return generate_fn(inputs, generation_config, **kwargs)

        # Preserve the dataframe-oriented GenerationMixin API used by subclasses
        # such as ``class Model(AutoModel, GenerationMixin)``.
        inherited_generate = getattr(super(), "generate", None)
        if inherited_generate is not None:
            return inherited_generate(inputs, generation_config=generation_config, **kwargs)

        model_type = (
            self.config.get("model_type", type(core).__name__)
            if isinstance(self.config, dict)
            else getattr(self.config, "model_type", type(core).__name__)
        )
        raise TypeError(f"{model_type} does not support generation.")


class AutoModelForPrediction(AutoModel):
    """tfts model for prediction"""

    @classmethod
    def from_config(cls, config, predict_sequence_length: int = 1):
        return AutoModel.from_config(config, predict_sequence_length, task="prediction")

    def __call__(
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        model_output = self.model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)

        residual = getattr(self.config, "residual", None)
        if residual is None and getattr(self.config, "skip_connect_circle", False):
            residual = "last_window"
        elif residual is None and getattr(self.config, "skip_connect_mean", False):
            residual = "mean"
        elif residual is None and getattr(self.config, "residual_last_value", False):
            residual = "last_value"
        return apply_prediction_residual(model_output, x, residual)


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
        return AutoModel.from_config(config, task="classification", num_labels=num_labels)

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
        model = AutoModel.from_pretrained(weights_dir)
        logger.info(f"Loaded anomaly model from {weights_dir}")
        return cls(model, model.config)

    @classmethod
    def from_config(cls, config, predict_sequence_length: int = 1):
        return AutoModel.from_config(config, predict_sequence_length, task="anomaly")


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
    def from_config(cls, config, predict_sequence_length: int = 1):
        return AutoModel.from_config(config, predict_sequence_length, task="segmentation")


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
    def from_config(cls, config, predict_sequence_length: int = 1):
        return AutoModel.from_config(config, predict_sequence_length, task="uncertainty")


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
    def from_config(cls, config, quantiles: Optional[List[float]] = None):
        return AutoModel.from_config(config, task="quantile", quantiles=quantiles)

    def build_model(self, inputs):
        model_output = self.model(inputs)
        outputs = self.head(model_output)
        self.keras_model = tf.keras.Model(inputs, outputs)
        return self.keras_model

    def compile_model(self, optimizer="adam"):
        """Helper to compile with the correct loss"""
        loss_fn = MultiQuantileLoss(quantiles=self.quantiles)
        self.keras_model.compile(optimizer=optimizer, loss=loss_fn)
