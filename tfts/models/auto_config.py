"""AutoConfig to set up models custom config"""

from collections import OrderedDict
import importlib
import json
import os
from typing import Any, Dict, Union

from .base import BaseConfig

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("seq2seq", "Seq2seqConfig"),
        ("rnn", "RNNConfig"),
        ("wavenet", "WaveNetConfig"),
        ("tcn", "TCNConfig"),
        ("transformer", "TransformerConfig"),
        ("bert", "BertConfig"),
        ("informer", "InformerConfig"),
        ("autoformer", "AutoFormerConfig"),
        ("tft", "TFTransformerConfig"),
        ("unet", "UnetConfig"),
        ("nbeats", "NBeatsConfig"),
    ]
)


class AutoConfig(BaseConfig):
    """AutoConfig for tfts model"""

    def __init__(self, **kwargs: Dict[str, object]):
        super().__init__(**kwargs)

    @classmethod
    def for_model(cls, model_name: str):

        if model_name in CONFIG_MAPPING_NAMES:
            class_name = CONFIG_MAPPING_NAMES[model_name]
            module = importlib.import_module(f".{model_name}", "tfts.models")
            mapping = getattr(module, class_name)

            return mapping()
        raise ValueError(
            f"Unrecognized model: {model_name}. Should contain one of {', '.join(CONFIG_MAPPING_NAMES.keys())}"
        )

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, os.PathLike], **kwargs):
        return
