"""
`Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
<https://arxiv.org/abs/1912.09363>`_
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import tensorflow as tf

from .base import BaseModel, CommonConfig
from .registry import register_model


class TFTransformerConfig(CommonConfig):
    model_type: str = "tft"

    def __init__(
        self,
        encoder_input_dim: int = 1,
        decoder_input_dim: int = 1,
        static_real_dim: int = 0,
        encoder_real_dim: Optional[int] = None,
        decoder_real_dim: Optional[int] = None,
        static_categorical_cardinalities: Optional[Sequence[int]] = None,
        temporal_categorical_cardinalities: Optional[Sequence[int]] = None,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_attention_heads: int = 4,
        output_size: int = 1,
        output_transform: Optional[str] = None,
        quantiles: Optional[Sequence[float]] = None,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        ffn_intermediate_size: int = 32,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.static_real_dim = static_real_dim
        self.encoder_real_dim = encoder_real_dim or encoder_input_dim
        self.decoder_real_dim = decoder_real_dim or decoder_input_dim
        self.static_categorical_cardinalities = list(static_categorical_cardinalities or [])
        self.temporal_categorical_cardinalities = list(temporal_categorical_cardinalities or [])
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.quantiles = list(quantiles or [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        self.output_size = output_size
        self.output_transform = output_transform
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.update(kwargs)


class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.value = tf.keras.layers.Dense(units)
        self.gate = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs, training=None):
        inputs = self.dropout(inputs, training=training)
        return self.value(inputs) * self.gate(inputs)


class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size=None, dropout=0.0, use_context=False, **kwargs):
        super().__init__(**kwargs)
        output_size = output_size or hidden_size
        self.context_projection = tf.keras.layers.Dense(hidden_size, use_bias=False) if use_context else None
        self.input_projection = tf.keras.layers.Dense(hidden_size)
        self.hidden_projection = tf.keras.layers.Dense(hidden_size)
        self.glu = GatedLinearUnit(output_size, dropout)
        self.skip_projection = tf.keras.layers.Dense(output_size)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, context=None, training=None):
        hidden = self.input_projection(inputs)
        if self.context_projection is not None and context is not None:
            context = self.context_projection(context)
            if len(hidden.shape) == 3 and len(context.shape) == 2:
                context = context[:, None, :]
            hidden = hidden + context
        hidden = self.hidden_projection(tf.nn.elu(hidden))
        return self.layer_norm(self.skip_projection(inputs) + self.glu(hidden, training=training))


class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, num_variables, hidden_size, dropout, use_context, **kwargs):
        super().__init__(**kwargs)
        if num_variables < 1:
            raise ValueError("A variable selection network requires at least one variable")
        self.num_variables = num_variables
        self.weight_grn = GatedResidualNetwork(hidden_size, num_variables, dropout, use_context)
        self.variable_grns = [
            GatedResidualNetwork(hidden_size, dropout=dropout, name=f"variable_{i}_grn") for i in range(num_variables)
        ]

    def call(self, variables: List[tf.Tensor], context=None, training=None):
        if len(variables) != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} variables, got {len(variables)}")
        weights = tf.nn.softmax(self.weight_grn(tf.concat(variables, axis=-1), context, training=training), axis=-1)
        transformed = tf.stack(
            [grn(value, training=training) for grn, value in zip(self.variable_grns, variables)],
            axis=-2,
        )
        return tf.reduce_sum(transformed * weights[..., :, None], axis=-2), weights


class GateAddNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout, **kwargs):
        super().__init__(**kwargs)
        self.glu = GatedLinearUnit(hidden_size, dropout)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, residual, training=None):
        return self.norm(residual + self.glu(inputs, training=training))


class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
    """Head-specific queries/keys with TFT's value projection shared by heads."""

    def __init__(self, hidden_size, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.key = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.value = tf.keras.layers.Dense(self.head_size, use_bias=False)
        self.output_projection = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, queries, keys, causal_mask=None, training=None):
        batch, query_length, key_length = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(keys)[1]
        q = tf.transpose(
            tf.reshape(self.query(queries), [batch, query_length, self.num_heads, self.head_size]),
            [0, 2, 1, 3],
        )
        k = tf.transpose(
            tf.reshape(self.key(keys), [batch, key_length, self.num_heads, self.head_size]),
            [0, 2, 1, 3],
        )
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_size, q.dtype))
        if causal_mask is not None:
            scores = tf.where(causal_mask[None, None], scores, tf.cast(-1e9, scores.dtype))
        attention = self.dropout(tf.nn.softmax(scores, -1), training=training)
        values = tf.tile(self.value(keys)[:, None], [1, self.num_heads, 1, 1])
        attended = tf.transpose(tf.matmul(attention, values), [0, 2, 1, 3])
        attended = tf.reshape(attended, [batch, query_length, self.num_heads * self.head_size])
        return self.output_projection(attended), tf.reduce_mean(attention, axis=1)


@register_model(
    "tft",
    config=TFTransformerConfig,
    paper="https://arxiv.org/abs/1912.09363",
    tags=("interpretable", "multi-horizon", "attention", "SOTA"),
    tier="core",
)
class TFTransformer(tf.keras.Model, BaseModel):
    """Temporal Fusion Transformer accepting dedicated static feature tensors."""

    def __init__(self, predict_sequence_length=1, config: Optional[TFTransformerConfig] = None):
        config = config or TFTransformerConfig()
        tf.keras.Model.__init__(self, name="temporal_fusion_transformer")
        BaseModel.__init__(self, predict_sequence_length=predict_sequence_length, config=config)
        self.config = config
        hidden, dropout = config.hidden_size, config.hidden_dropout_prob
        self.static_cat_embeddings = [
            tf.keras.layers.Embedding(cardinality, hidden, name=f"static_cat_{i}")
            for i, cardinality in enumerate(config.static_categorical_cardinalities)
        ]
        self.temporal_cat_embeddings = [
            tf.keras.layers.Embedding(cardinality, hidden, name=f"temporal_cat_{i}")
            for i, cardinality in enumerate(config.temporal_categorical_cardinalities)
        ]
        self.static_real_projections = [
            tf.keras.layers.Dense(hidden, name=f"static_real_{i}") for i in range(config.static_real_dim)
        ]
        self.encoder_real_projections = [
            tf.keras.layers.Dense(hidden, name=f"encoder_real_{i}") for i in range(config.encoder_real_dim)
        ]
        self.decoder_real_projections = [
            tf.keras.layers.Dense(hidden, name=f"decoder_real_{i}") for i in range(config.decoder_real_dim)
        ]
        static_count = len(self.static_cat_embeddings) + len(self.static_real_projections)
        self.has_static = static_count > 0
        if self.has_static:
            self.static_selection = VariableSelectionNetwork(static_count, hidden, dropout, False)
            self.static_variable_context = GatedResidualNetwork(hidden, dropout=dropout)
            self.static_hidden_context = GatedResidualNetwork(hidden, dropout=dropout)
            self.static_cell_context = GatedResidualNetwork(hidden, dropout=dropout)
            self.static_enrichment_context = GatedResidualNetwork(hidden, dropout=dropout)
        encoder_count = len(self.temporal_cat_embeddings) + len(self.encoder_real_projections)
        decoder_count = len(self.temporal_cat_embeddings) + len(self.decoder_real_projections)
        self.encoder_selection = VariableSelectionNetwork(encoder_count, hidden, dropout, self.has_static)
        self.decoder_selection = VariableSelectionNetwork(decoder_count, hidden, dropout, self.has_static)
        self.encoder_lstm = tf.keras.layers.LSTM(hidden, return_sequences=True, return_state=True)
        self.decoder_lstm = tf.keras.layers.LSTM(hidden, return_sequences=True, return_state=True)
        self.post_lstm_gate = GateAddNorm(hidden, dropout)
        self.static_enrichment = GatedResidualNetwork(hidden, dropout=dropout, use_context=self.has_static)
        self.attention = InterpretableMultiHeadAttention(
            hidden, config.num_attention_heads, config.attention_probs_dropout_prob
        )
        self.post_attention_gate = GateAddNorm(hidden, dropout)
        self.positionwise_grn = GatedResidualNetwork(hidden, dropout=dropout)
        self.pre_output_gate = GateAddNorm(hidden, dropout)
        self.output_projection = tf.keras.layers.Dense(config.output_size)
        self.last_selection_weights = self.last_attention_weights = None

    @staticmethod
    def _embed_categoricals(values, embeddings):
        return [embedding(values[..., i]) for i, embedding in enumerate(embeddings)]

    @staticmethod
    def _project_reals(values, projections):
        return [projection(values[..., i : i + 1]) for i, projection in enumerate(projections)]

    def _legacy_inputs(self, values):
        batch = tf.shape(values)[0]
        return {
            "encoder_real": values,
            "decoder_real": tf.zeros([batch, self.predict_sequence_length, self.config.decoder_real_dim], values.dtype),
            "encoder_categorical": tf.zeros([batch, tf.shape(values)[1], 0], tf.int32),
            "decoder_categorical": tf.zeros([batch, self.predict_sequence_length, 0], tf.int32),
        }

    def call(self, x=None, output_hidden_states=None, return_dict=None, training=None, **kwargs):
        del output_hidden_states, kwargs
        inputs = x if isinstance(x, dict) else self._legacy_inputs(x)
        static_variable_context = initial_state = enrichment_context = static_weights = None
        if self.has_static:
            static_variables = self._embed_categoricals(
                inputs["static_categorical"], self.static_cat_embeddings
            ) + self._project_reals(inputs["static_real"], self.static_real_projections)
            static_context, static_weights = self.static_selection(static_variables, training=training)
            static_variable_context = self.static_variable_context(static_context, training=training)
            initial_state = [
                self.static_hidden_context(static_context, training=training),
                self.static_cell_context(static_context, training=training),
            ]
            enrichment_context = self.static_enrichment_context(static_context, training=training)
        encoder_variables = self._embed_categoricals(
            inputs["encoder_categorical"], self.temporal_cat_embeddings
        ) + self._project_reals(inputs["encoder_real"], self.encoder_real_projections)
        decoder_variables = self._embed_categoricals(
            inputs["decoder_categorical"], self.temporal_cat_embeddings
        ) + self._project_reals(inputs["decoder_real"], self.decoder_real_projections)
        selected_encoder, encoder_weights = self.encoder_selection(
            encoder_variables, context=static_variable_context, training=training
        )
        selected_decoder, decoder_weights = self.decoder_selection(
            decoder_variables, context=static_variable_context, training=training
        )
        encoded, hidden_state, cell_state = self.encoder_lstm(
            selected_encoder, initial_state=initial_state, training=training
        )
        decoded, _, _ = self.decoder_lstm(selected_decoder, initial_state=[hidden_state, cell_state], training=training)
        selected = tf.concat([selected_encoder, selected_decoder], axis=1)
        local = self.post_lstm_gate(tf.concat([encoded, decoded], 1), residual=selected, training=training)
        enriched = self.static_enrichment(local, context=enrichment_context, training=training)
        encoder_length, total_length = tf.shape(encoded)[1], tf.shape(enriched)[1]
        queries = enriched[:, encoder_length:]
        mask = tf.range(total_length)[None, :] <= tf.range(encoder_length, total_length)[:, None]
        attended, attention_weights = self.attention(queries, enriched, causal_mask=mask, training=training)
        attention_output = self.post_attention_gate(attended, residual=queries, training=training)
        positionwise = self.positionwise_grn(attention_output, training=training)
        fused = self.pre_output_gate(positionwise, residual=local[:, encoder_length:], training=training)
        output = self.output_projection(fused)
        if self.config.output_transform == "group_softplus":
            if "target_scale" not in inputs:
                raise ValueError("group_softplus output requires a target_scale input")
            target_scale = tf.cast(inputs["target_scale"], output.dtype)
            transformed = output * target_scale[:, None, 1:2] + target_scale[:, None, 0:1]
            output = tf.nn.softplus(transformed)
        self.last_selection_weights = {
            "static": static_weights,
            "encoder": encoder_weights,
            "decoder": decoder_weights,
        }
        self.last_attention_weights = attention_weights
        if return_dict:
            return {
                "prediction": output,
                "selection_weights": self.last_selection_weights,
                "attention_weights": attention_weights,
            }
        return output
