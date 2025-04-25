"""Layer for :py:class:`~tfts.models.transformer` :py:class:`~tfts.models.autoformer`"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout

from tfts.layers.mask_layer import ProbMask


class Attention(tf.keras.layers.Layer):
    """Multi-head attention layer"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.0,
        position_embedding_type=None,
    ) -> None:
        """Initialize the Attention layer.

        Parameters:
        -----------
        hidden_size : int
            The number of hidden units, hidden_size = attention_dim_each_head x num_attention_heads.
        num_attention_heads : int
            The number of attention heads.
        attention_probs_dropout_prob : float, optional
            Dropout rate for the attention weights. Defaults to 0.0.
        """
        super(Attention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by the number of heads {num_attention_heads}."
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.dense_q = Dense(self.hidden_size, use_bias=False)
        self.dense_k = Dense(self.hidden_size, use_bias=False)
        self.dense_v = Dense(self.hidden_size, use_bias=False)
        self.dropout = Dropout(rate=self.attention_probs_dropout_prob)
        super(Attention, self).build(input_shape)

    def call(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        past_key_value=None,
        training: Optional[bool] = None,
        return_attention_scores: bool = False,
        use_causal_mask: bool = False,
    ):
        """use query and key generating an attention multiplier for value, multi_heads to repeat it

        Parameters
        ----------
        q : tf.Tenor
            Query with shape batch * seq_q * fea
        k : tf.Tensor
            Key with shape batch * seq_k * fea
        v : tf.Tensor
            Value with shape batch * seq_v * fea
        mask :tf.Tensor, optional
            important to avoid the leaks, by default None

        Returns
        -------
        tf.Tensor
            Tensor with shape batch * seq_q * (units * num_attention_heads)
        """
        # project the query/key/value to num_attention_heads * units
        q = self.dense_q(q)
        k = self.dense_k(k)
        v = self.dense_v(v)

        # multi-heads transfer to multi-sample
        q_ = tf.concat(tf.split(q, self.num_attention_heads, axis=2), axis=0)
        k_ = tf.concat(tf.split(k, self.num_attention_heads, axis=2), axis=0)
        v_ = tf.concat(tf.split(v, self.num_attention_heads, axis=2), axis=0)

        # => (batch * heads) * seq_q * seq_k
        score = tf.linalg.matmul(q_, k_, transpose_b=True)
        score = score / tf.cast(tf.shape(q_)[-1], score.dtype) ** 0.5

        if mask is not None:
            mask = tf.repeat(mask, repeats=self.num_attention_heads, axis=0)
            score = score * tf.cast(mask, score.dtype)

        score = tf.nn.softmax(score, axis=-1)
        score = self.dropout(score, training=training)

        # (batch * heads) * seq_q * units
        outputs = tf.linalg.matmul(score, v_)
        outputs = tf.concat(tf.split(outputs, self.num_attention_heads, axis=0), axis=2)
        outputs = tf.cast(outputs, tf.float32)

        if return_attention_scores:
            return outputs, score
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = (*input_shape[0][:-1], self.hidden_size)
        return output_shape

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.0,
        position_embedding_type=None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super(SelfAttention, self).__init__()
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            position_embedding_type=position_embedding_type,
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        super(SelfAttention, self).build(input_shape)

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: Optional[bool] = None):
        """Self attention layer

        Parameters
        ----------
        x : tf.Tensor
            3D input tensor for self-attention, (batch_size, sequence_length, feature_size)
        mask : tf.Tensor, optional
            masked, by default None

        Returns
        -------
        tf.Tensor
            3D self attention output, (batch_size, sequence_length, attention_hidden_size)
        """
        return self.attention(q=x, k=x, v=x, mask=mask, training=training)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        return base_config


class ProbAttention(tf.keras.layers.Layer):
    def __init__(
        self, hidden_size: int = 128, num_attention_heads: int = 1, attention_probs_dropout_prob: float = 0.0, **kwargs
    ):
        super().__init__()
        self.mask_flag = True
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.factor = 5
        self.scale = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.dense_q = Dense(self.hidden_size, use_bias=False)
        self.dense_k = Dense(self.hidden_size, use_bias=False)
        self.dense_v = Dense(self.hidden_size, use_bias=False)
        super().build(input_shape)

    def _prob_qk(self, q, k, sample_k, top_n):
        _, H, L, E = k.shape
        _, _, S, _ = q.shape
        B = tf.shape(k)[0]

        k_expand = tf.broadcast_to(tf.expand_dims(k, -3), (B, H, L, S, E))

        indx_q_seq = tf.random.uniform((S,), maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random.uniform((sample_k,), maxval=L, dtype=tf.int32)

        K_sample = tf.gather(k_expand, tf.range(S), axis=2)
        K_sample = tf.gather(K_sample, indx_q_seq, axis=2)
        K_sample = tf.gather(K_sample, indx_k_seq, axis=3)

        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(q, -2), tf.einsum("...ij->...ji", K_sample)), axis=3)
        M = tf.math.reduce_max(Q_K_sample, axis=-1) - tf.raw_ops.Div(x=tf.reduce_sum(Q_K_sample, axis=-1), y=L)
        m_top = tf.math.top_k(M, top_n, sorted=False)[1]

        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, H, top_n))
        head_indexes = tf.tile(tf.range(H)[tf.newaxis, :, tf.newaxis], (B, 1, top_n))

        idx = tf.stack([batch_indexes, head_indexes, m_top], axis=-1)

        q_reduce = tf.gather_nd(q, idx)
        qk = tf.matmul(q_reduce, tf.transpose(k, (0, 1, 3, 2)))
        return qk, m_top

    def _get_initial_context(self, v, L_Q):
        _, H, L_V, D = v.shape
        B = tf.shape(v)[0]
        if not self.mask_flag:
            v_sum = tf.math.reduce_sum(v, axis=-2)
            context = tf.identity(tf.boradcast_to(tf.expand_dims(v_sum, -2), [B, H, L_Q, v_sum.shape[-1]]))
        else:
            assert L_Q == L_V
            context = tf.math.cumsum(v, axis=-2)
        return context

    def _update_context(self, context_in, v, scores, index, L_Q):
        _, H, L_V, D = v.shape
        B = tf.shape(v)[0]
        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, H, tf.shape(index)[-1]))
        head_indexes = tf.tile(tf.range(H)[tf.newaxis, :, tf.newaxis], (B, 1, tf.shape(index)[-1]))
        index = tf.stack([batch_indexes, head_indexes, index], axis=-1)

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores).mask
            scores = tf.where(attn_mask, -np.inf, scores)

        attn = tf.nn.softmax(scores, axis=-1)
        context_in = tf.tensor_scatter_nd_update(context_in, index, tf.matmul(attn, v))
        return tf.convert_to_tensor(context_in)

    # @tf.function
    def call(self, q, k, v, mask: Optional[tf.Tensor] = None):
        """Prob attention"""
        q = self.dense_q(q)  # project the query/key/value to num_attention_heads * units
        k = self.dense_k(k)
        v = self.dense_v(v)

        _, L, D = q.shape
        B = tf.shape(q)[0]
        _, S, _ = k.shape

        q_ = tf.reshape(q, (-1, self.num_attention_heads, L, self.hidden_size // self.num_attention_heads))
        k_ = tf.reshape(k, (-1, self.num_attention_heads, S, self.hidden_size // self.num_attention_heads))
        v_ = tf.reshape(v, (-1, self.num_attention_heads, S, self.hidden_size // self.num_attention_heads))

        u_q = self.factor * np.ceil(np.log(L)).astype("int").item()
        u_k = self.factor * np.ceil(np.log(S)).astype("int").item()
        u_q = u_q if u_q < L else L
        u_k = u_k if u_k < S else S

        scores_top, index = self._prob_qk(q_, k_, u_k, u_q)
        scores_top = scores_top * 1.0 / np.sqrt(D // self.num_attention_heads)

        context = self._get_initial_context(v_, L)
        context = self._update_context(context, v_, scores_top, index, L)

        out = tf.reshape(context, (B, L, -1))
        return out

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseAttention(tf.keras.layers.Layer):
    """
    SparseAttention implementation
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float = 0.0, **kwargs):
        super().__init__()

    def build(self, input_shape: Tuple[Optional[int], ...]):
        super().build(input_shape)

    def call(self, x, mask: Optional[tf.Tensor] = None):
        """Sparse attention

        Parameters
        ----------
        x : tf.Tensor
            _description_
        mask : tf.Tensor, optional
            _description_, by default None
        """
        return

    def get_config(self):
        base_config = super().get_config()
        return base_config


class FastAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        super().build(input_shape)

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None):
        """Fast attention

        Parameters
        ----------
        x : tf.Tensor
            _description_
        mask : tf.Tensor, optional
            _description_, by default None
        """
        return

    def get_config(self):
        base_config = super().get_config()
        return base_config
