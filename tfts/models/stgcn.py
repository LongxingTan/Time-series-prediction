"""A compact spatio-temporal graph convolutional network."""

from typing import Optional

import tensorflow as tf

from tfts.contracts import BackboneCapabilities, ModelInputSpec, OutputPort, SpatialLayout
from tfts.layers import ChebConv

from .base import BaseModel, CommonConfig
from .registry import register_model


class STGCNConfig(CommonConfig):
    model_type = "stgcn"

    def __init__(self, hidden_size=64, num_layers=2, cheb_k=3, temporal_kernel=3, dropout=0.1, **kwargs):
        super().__init__(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, **kwargs)
        self.cheb_k = int(cheb_k)
        self.temporal_kernel = int(temporal_kernel)

    def __post_init__(self):
        if self.cheb_k < 1:
            raise ValueError("cheb_k must be at least one")
        if self.temporal_kernel < 1:
            raise ValueError("temporal_kernel must be at least one")


class _STGCNBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, cheb_k, temporal_kernel, dropout, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.cheb_k = int(cheb_k)
        self.temporal_kernel = int(temporal_kernel)
        self.dropout_rate = float(dropout)
        self.temporal_filter = tf.keras.layers.Conv2D(self.hidden_size, (self.temporal_kernel, 1), padding="same")
        self.temporal_gate = tf.keras.layers.Conv2D(
            self.hidden_size, (self.temporal_kernel, 1), padding="same", activation="sigmoid"
        )
        self.graph = ChebConv(self.hidden_size, self.cheb_k, activation="relu")
        self.residual_projection = tf.keras.layers.Dense(self.hidden_size)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, values, adjacency, training=None):
        residual = self.residual_projection(values)
        temporal = tf.nn.tanh(self.temporal_filter(values)) * self.temporal_gate(values)
        shape = tf.shape(temporal)
        graph_input = tf.reshape(temporal, [-1, shape[2], shape[3]])
        if adjacency.shape.rank == 3:
            adjacency = tf.repeat(adjacency, shape[1], axis=0)
        graph_output = self.graph((graph_input, adjacency))
        graph_output = tf.reshape(graph_output, [shape[0], shape[1], shape[2], self.hidden_size])
        return self.norm(residual + self.dropout(graph_output, training=training))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "cheb_k": self.cheb_k,
                "temporal_kernel": self.temporal_kernel,
                "dropout": self.dropout_rate,
            }
        )
        return config


@register_model(
    "stgcn",
    config=STGCNConfig,
    paper="https://arxiv.org/abs/1709.04875",
    tags=("graph", "spatiotemporal", "convolutional"),
    tier="experimental",
    capabilities=BackboneCapabilities(
        output_ports=frozenset({OutputPort.NATIVE_FORECAST}),
        input_spec=ModelInputSpec(
            accepted_layouts=frozenset({SpatialLayout.NODES}),
            requires_structure=True,
        ),
    ),
)
class STGCN(BaseModel):
    """Direct graph forecast over dense shared or batch-specific topology."""

    def __init__(self, predict_sequence_length=1, config: Optional[STGCNConfig] = None):
        config = config or STGCNConfig()
        super().__init__(predict_sequence_length=predict_sequence_length, config=config)
        self.blocks = [
            _STGCNBlock(
                config.hidden_size,
                config.cheb_k,
                config.temporal_kernel,
                config.dropout,
                name=f"block_{index}",
            )
            for index in range(config.num_layers)
        ]
        self.horizon_projection = tf.keras.layers.Dense(self.predict_sequence_length)

    def adapt_batch(self, batch):
        if batch.structure.adjacency is None:
            raise ValueError("stgcn requires dense adjacency")
        return {"values": batch.past_values, "adjacency": batch.structure.adjacency}

    def call(self, inputs, training=None, return_dict=False):
        values, adjacency = inputs["values"], inputs["adjacency"]
        if adjacency.shape.rank == 4:
            raise ValueError("stgcn does not support time-varying adjacency")
        x = values
        for block in self.blocks:
            x = block(x, adjacency, training=training)
        last = x[:, -1, :, :]
        forecast = self.horizon_projection(last)
        forecast = tf.transpose(forecast, [0, 2, 1])[:, :, :, None]
        return {"predictions": forecast} if return_dict else forecast
