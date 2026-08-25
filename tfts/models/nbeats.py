"""
`N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
<https://arxiv.org/abs/1905.10437>`_

"""

from typing import List, Optional

import tensorflow as tf

from ..layers.nbeats_layer import GenericBlock, SeasonalityBlock, TrendBlock
from .base import BaseModel, CommonConfig
from .registry import register_model


class NBeatsConfig(CommonConfig):
    model_type: str = "nbeats"

    def __init__(
        self,
        stack_types: Optional[List[str]] = None,
        widths: Optional[List[int]] = None,
        num_blocks: Optional[List[int]] = None,
        num_block_layers: Optional[List[int]] = None,
        expansion_coefficient_lengths: Optional[List[int]] = None,
        share_weights_in_stack: bool = False,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0.0,
    ):
        super(NBeatsConfig, self).__init__()
        if stack_types is None:
            stack_types = ["trend", "seasonality"]
        if widths is None:
            widths = [32, 512]
        if num_blocks is None:
            num_blocks = [3, 3]
        if num_block_layers is None:
            num_block_layers = [3, 3]
        if expansion_coefficient_lengths is None:
            expansion_coefficient_lengths = [3, 7]

        assert (
            len(stack_types)
            == len(widths)
            == len(num_blocks)
            == len(num_block_layers)
            == len(expansion_coefficient_lengths)
        ), "all per-stack hyperparameters must have the same length as stack_types"

        self.stack_types = stack_types
        self.widths = widths
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.share_weights_in_stack = share_weights_in_stack
        self.dropout = dropout
        self.backcast_loss_ratio = backcast_loss_ratio

    # tolerate the old singular-parameter spelling
    @classmethod
    def from_dict(cls, config_dict):
        config_dict = dict(config_dict)
        legacy = {}
        if "hidden_size" in config_dict and "widths" not in config_dict:
            hs = config_dict.pop("hidden_size")
            legacy["widths"] = [hs, hs]
        if "thetas_dims" in config_dict and "expansion_coefficient_lengths" not in config_dict:
            config_dict.pop("thetas_dims")
        if "nb_blocks_per_stack" in config_dict and "num_blocks" not in config_dict:
            nb = config_dict.pop("nb_blocks_per_stack")
            legacy["num_blocks"] = [nb, nb]
        if "stack_types" in config_dict:
            config_dict["stack_types"] = [s if s != "trend_block" else "trend" for s in config_dict["stack_types"]]
            config_dict["stack_types"] = [
                s if s != "seasonality_block" else "seasonality" for s in config_dict["stack_types"]
            ]
        config_dict.update(legacy)
        config = cls()
        config.update(config_dict)
        return config


@register_model(
    "nbeats",
    config=NBeatsConfig,
    paper="https://arxiv.org/abs/1905.10437",
    tags=("mlp", "interpretable", "SOTA", "basis-expansion"),
    tier="core",
)
class NBeats(BaseModel):
    """NBeats model (interpretable trend + seasonality stacks, doubly-residual stacking)."""

    def __init__(
        self,
        predict_sequence_length: int = 20,
        config: Optional[NBeatsConfig] = None,
    ):
        super().__init__(predict_sequence_length=predict_sequence_length, config=config or NBeatsConfig())
        self.train_sequence_length = None
        self.stacks: Optional[List[List[tf.keras.layers.Layer]]] = None

        self.block_lookup = {
            "trend": TrendBlock,
            "trend_block": TrendBlock,
            "seasonality": SeasonalityBlock,
            "seasonality_block": SeasonalityBlock,
            "generic": GenericBlock,
            "generic_block": GenericBlock,
            "general": GenericBlock,
        }

    def _extract_x(self, inputs):
        """N-BEATS is univariate: keep only the lookback window ``value``."""
        if isinstance(inputs, (list, tuple)):
            return inputs[0] if len(inputs) >= 1 else inputs
        if isinstance(inputs, dict):
            if "x" in inputs:
                return inputs["x"]
            return list(inputs.values())[0]
        return inputs

    def _build_stacks(self):
        cfg = self.config
        self.stacks = []
        for stack_id, stack_type in enumerate(cfg.stack_types):
            stack = []
            units = cfg.widths[stack_id]
            num_layers = cfg.num_block_layers[stack_id]
            n_blocks = cfg.num_blocks[stack_id]
            for _ in range(n_blocks):
                if stack_type in ("trend", "trend_block"):
                    thetas = cfg.expansion_coefficient_lengths[stack_id]
                    block = TrendBlock(
                        self.train_sequence_length,
                        self.predict_sequence_length,
                        units,
                        num_layers,
                        thetas_dim=thetas,
                        dropout=cfg.dropout,
                    )
                elif stack_type in ("seasonality", "seasonality_block"):
                    block = SeasonalityBlock(
                        self.train_sequence_length,
                        self.predict_sequence_length,
                        units,
                        num_layers,
                        min_period=cfg.expansion_coefficient_lengths[stack_id],
                        dropout=cfg.dropout,
                    )
                elif stack_type in ("generic", "generic_block", "general"):
                    block = GenericBlock(
                        self.train_sequence_length, self.predict_sequence_length, units, num_layers, dropout=cfg.dropout
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")
                stack.append(block)
            self.stacks.append(stack)

    def __call__(
        self, inputs: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
        x = self._extract_x(inputs)
        # x: (batch, train_sequence_length, n_features)
        self.train_sequence_length = x.shape[1]
        self.config.train_sequence_length = int(self.train_sequence_length)
        if self.stacks is None:
            self._build_stacks()

        # squeeze the feature dim; Keras 3 forbids raw `tf.*` on symbolic tensors, so wrap in Lambda
        squeeze = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=2), output_shape=lambda s: (s[0], s[1]))
        expand = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1), output_shape=lambda s: s + (1,))

        x_sq = squeeze(x)  # (batch, train_sequence_length)
        backcast = x_sq
        forecast = backcast[:, : self.predict_sequence_length] * 0.0  # (batch, predict_sequence_length)

        for stack in self.stacks:
            for block in stack:
                b, f = block(backcast)
                backcast = backcast - b
                forecast = forecast + f

        return expand(forecast)  # (batch, predict_sequence_length, 1)
