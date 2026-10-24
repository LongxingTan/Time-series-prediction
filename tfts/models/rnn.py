"""
`Long short term memory
<http://www.bioinf.jku.at/publications/older/2604.pdf>`_
"""

from typing import Literal, Optional

import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Reshape

from tfts.config import CommonConfig
from tfts.layers.rnn_layer import Encoder
from tfts.models.base import BackboneCapabilities, BaseModel, OutputPort

from .catalog import register_model


class RNNConfig(CommonConfig):
    model_type: str = "rnn"

    def __init__(
        self,
        rnn_hidden_size: int = 64,
        rnn_type: Literal["gru", "lstm"] = "gru",
        bi_direction: bool = False,
        dense_hidden_size: int = 128,
        num_stacked_layers: int = 1,
        scheduled_sampling: float = 0.0,
        use_attention: bool = False,
        return_sequences: bool = False,
        input_projection: bool = False,
    ) -> None:
        """
        Initializes the configuration for the RNN model with the specified parameters.

        Args:
            rnn_hidden_size: The number of units in the RNN hidden layer.
            rnn_type: Type of RNN ('gru' or 'lstm').
            bi_direction: Whether to use bidirectional RNN.
            dense_hidden_size: The size of the dense hidden layer following the RNN.
            num_stacked_layers: The number of stacked RNN layers.
            scheduled_sampling: Scheduled sampling ratio.
            use_attention: Whether to use attention mechanism.
            return_sequences: Predict one value per input step. The input
                length must match predict_sequence_length in this mode.
            input_projection: Learn a pointwise two-layer embedding before
                the recurrent encoder, using dense_hidden_size units.
        """
        super().__init__()

        self.rnn_hidden_size: int = rnn_hidden_size
        self.rnn_type: Literal["gru", "lstm"] = rnn_type
        self.bi_direction: bool = bi_direction
        self.dense_hidden_size: int = dense_hidden_size
        self.num_stacked_layers: int = num_stacked_layers
        self.scheduled_sampling: float = scheduled_sampling
        self.use_attention: bool = use_attention
        self.return_sequences: bool = return_sequences
        self.input_projection: bool = input_projection


@register_model(
    "rnn",
    config=RNNConfig,
    tags=("baseline", "recurrent"),
    tier="core",
    capabilities=BackboneCapabilities(
        output_ports=frozenset({OutputPort.NATIVE_FORECAST, OutputPort.TEMPORAL_SEQUENCE})
    ),
)
class RNN(BaseModel):
    """tfts RNN model"""

    def adapt_batch(self, batch):
        if any(
            getattr(batch, name) is not None
            for name in (
                "future_time_features",
                "static_real_features",
                "static_categorical_features",
                "past_categorical_features",
                "future_categorical_features",
            )
        ):
            raise ValueError("RNN accepts past_values, past_time_features, and padding_mask only")
        # Time features and padding masks are optional RNN inputs.  Without
        # them the public contract stays a plain (batch, time, features)
        # tensor so the persisted ``input_shape`` remains the flat
        # user-facing shape; fabricating a zero-width ``encoder_feature``
        # here would leak an internal input into the saved config and break
        # ``AutoModel.from_pretrained`` shape-based restoration.
        if batch.past_time_features is None and batch.padding_mask is None:
            return batch.past_values
        values = {"x": batch.past_values}
        if batch.past_time_features is not None:
            values["encoder_feature"] = batch.past_time_features
        else:
            values["encoder_feature"] = tf.zeros_like(batch.past_values[..., :0])
        if batch.padding_mask is not None:
            values["padding_mask"] = batch.padding_mask
        return values

    def __init__(self, predict_sequence_length: int = 1, config: Optional[RNNConfig] = None):
        super().__init__()
        self.config = config or RNNConfig()
        self.predict_sequence_length = predict_sequence_length
        self.input_layers = (
            [
                Dense(self.config.dense_hidden_size // 2, activation="relu"),
                Dense(self.config.dense_hidden_size, activation="relu"),
            ]
            if self.config.input_projection
            else []
        )
        self.encoder = Encoder(
            rnn_size=self.config.rnn_hidden_size,
            rnn_type=self.config.rnn_type,
            num_stacked_layers=self.config.num_stacked_layers,
            bi_direction=self.config.bi_direction,
            return_state=True,
        )

        self.dense1 = Dense(self.config.dense_hidden_size, activation="relu")
        self.dense2 = Dense(self.config.dense_hidden_size, activation="relu")
        self.project1 = Dense(
            self.config.target_dim * (1 if self.config.return_sequences else predict_sequence_length), activation=None
        )

    def call(
        self,
        inputs,
        teacher=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ):
        """RNN model call

        Parameters
        ----------
        inputs : Union[list, tuple, dict, tf.Tensor]
            Input data.
        teacher : tf.Tensor, optional
            Teacher signal for training, by default None.
        return_dict : bool, optional
            Whether to return a dictionary, by default None.

        Returns
        -------
        tf.Tensor
            Model output.
        """
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs)
        mask = inputs.get("padding_mask") if isinstance(inputs, dict) else None
        if self.config.return_sequences:
            tf.debugging.assert_equal(
                tf.shape(encoder_feature)[1],
                self.predict_sequence_length,
                message="Aligned RNN input length must equal predict_sequence_length",
            )
        for layer in self.input_layers:
            encoder_feature = layer(encoder_feature)
        encoder_outputs, encoder_state = self.encoder(encoder_feature, mask=mask, training=training)

        if output_hidden_states:
            return encoder_outputs

        encoder_out = self.dense1(encoder_outputs if self.config.return_sequences else encoder_state)
        encoder_out = self.dense2(encoder_out)
        outputs = self.project1(encoder_out)
        if self.config.return_sequences:
            return outputs
        outputs = Reshape((self.predict_sequence_length, self.config.target_dim))(outputs)
        return outputs


class RNN2(object):
    def __init__(self, predict_sequence_length=3, config=RNNConfig()) -> None:
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.rnn = GRU(
            units=config.rnn_hidden_size, activation="tanh", return_state=True, return_sequences=True, dropout=0
        )
        self.dense1 = Dense(predict_sequence_length)
        self.dense2 = Dense(1)

    def __call__(self, inputs: tf.Tensor, teacher: Optional[tf.Tensor] = None):
        """Another RNN model

        Parameters
        ----------
        inputs : _type_
            _description_
        teacher : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, _ = inputs
            encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs

        encoder_shape = tf.shape(encoder_features)
        future = tf.zeros([encoder_shape[0], self.predict_sequence_length, encoder_shape[2]])
        encoder_features = tf.concat([encoder_features, future], axis=1)
        output, state = self.rnn(encoder_features)
        output = self.dense2(output)

        return output[:, -self.predict_sequence_length :]
