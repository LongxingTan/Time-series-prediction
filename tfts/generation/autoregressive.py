"""Autoregressive generation mixin (Transformers-style)."""

from typing import Any, Dict, Optional

import tensorflow as tf

from .configuration import ForecastGenerationConfig
from .outputs import ForecastGenerationOutput


class AutoregressiveGenerationMixin:
    """Mixin that gives an autoregressive model an explicit, optional ``generate(...)``.

    The model implements three hooks:

    - ``prepare_generation_inputs(inputs)``   -> ``(x, decoder_feature, static)``
    - ``initialize_generation_state(x, static)`` -> initial RNN ``state``
    - ``decode_step(previous_target, static, state, training)``
                                                -> ``(params_dict, next_state)``

    ``generate`` drives the loop, draws samples (ancestral) or follows ``loc`` (greedy),
    and aggregates the resulting trajectories into a point forecast. ``__call__`` and the
    default training path are untouched -- generation is opt-in only.
    """

    # ------------------------------------------------------------------ hooks
    def prepare_generation_inputs(self, inputs):
        if isinstance(inputs, dict):
            return inputs.get("x"), inputs.get("decoder_feature"), inputs.get("static")
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
            decoder_feature = inputs[1] if len(inputs) > 1 else None
            static = inputs[2] if len(inputs) > 2 else None
            return x, decoder_feature, static
        return inputs, None, None

    def initialize_generation_state(self, x, static):
        raise NotImplementedError("Subclasses must implement initialize_generation_state")

    def decode_step(self, previous_target, static, state, training=False):
        raise NotImplementedError("Subclasses must implement decode_step")

    def update_generation_state(self, state):
        """Caching/state-shaping hook (default: pass-through). Returns the state."""
        return state

    # ------------------------------------------------------------------ sampling
    @staticmethod
    def _aggregate(traj: tf.Tensor, aggregation: Optional[str]) -> tf.Tensor:
        """Reduce ``[batch, num_samples, horizon, dim]`` trajectories to a point forecast."""
        if aggregation is None or aggregation == "none":
            return traj[:, 0, :, :]
        if aggregation == "mean":
            return tf.reduce_mean(traj, axis=1)
        if aggregation == "median":
            sorted_ = tf.sort(traj, axis=1)
            count = tf.shape(sorted_)[1]
            lower = (count - 1) // 2
            upper = count // 2
            return (tf.gather(sorted_, lower, axis=1) + tf.gather(sorted_, upper, axis=1)) / 2.0
        raise ValueError(f"Unsupported aggregation: {aggregation!r}")

    # ------------------------------------------------------------------ driver
    def generate(
        self,
        inputs,
        generation_config: Optional[Any] = None,
        **kwargs,
    ) -> ForecastGenerationOutput:
        config = ForecastGenerationConfig.from_args(generation_config)
        if kwargs:
            values = {**config.__dict__, **kwargs}
            config = ForecastGenerationConfig.from_args(values)
        valid_modes = {"ancestral", "greedy", "teacher_forced", "sample"}
        if config.mode not in valid_modes:
            raise ValueError(f"Unsupported generation mode: {config.mode!r}")
        x, decoder_feature, static = self.prepare_generation_inputs(inputs)
        if x is None or static is None:
            raise ValueError("generate() requires inputs with keys 'x' and 'static'.")
        pred_len = int(getattr(self, "predict_sequence_length"))

        if config.mode == "teacher_forced":
            if decoder_feature is None:
                raise ValueError("teacher_forced mode requires inputs['decoder_feature'].")
            return self._generate_teacher_forced(x, static, decoder_feature, pred_len)

        return self._generate_sampled(x, static, config, pred_len)

    def _generate_sampled(
        self,
        x: tf.Tensor,
        static: tf.Tensor,
        config: ForecastGenerationConfig,
        pred_len: int,
    ) -> ForecastGenerationOutput:
        x = tf.convert_to_tensor(x)
        static = tf.convert_to_tensor(static)
        if x.shape.rank != 3 or static.shape.rank not in (1, 2):
            raise ValueError("Expected x with rank 3 and static with rank 1 or 2.")
        if static.shape.rank == 1:
            static = static[:, None]
        if x.shape[0] is None:
            raise ValueError("Generation currently requires a statically known batch size.")
        B = int(x.shape[0])
        S = config.num_samples
        if config.mode == "greedy":
            S = 1
        if S < 1:
            raise ValueError("num_samples must be >= 1.")
        N = B * S

        # Precompute deterministic noise over all batch/sample rows. Chunked execution
        # uses the same draws; underlying batched kernels may still differ by normal
        # floating-point roundoff when their batch dimensions change.
        noise_steps = None
        seed = config.seed
        if seed is not None:
            noise_steps = [
                tf.random.stateless_normal([N, 1, 1], seed=tf.cast(tf.stack([seed, t]), tf.int64))
                for t in range(pred_len)
            ]

        chunk_size = config.batch_size if config.batch_size is not None else B
        if chunk_size < 1:
            raise ValueError("batch_size must be >= 1.")
        preds, samples, locs, scales = [], [], [], []
        row0 = 0
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            c = end - start
            trajS, locS, scaleS = self._sample_chunk(
                x[start:end], static[start:end], N, row0, S, config, pred_len, noise_steps
            )
            preds.append(self._aggregate(trajS, config.aggregation))
            samples.append(trajS)
            locs.append(locS)
            scales.append(scaleS)
            row0 += c * S

        predictions = tf.concat(preds, axis=0)  # (B, pred_len, 1)
        loc_all = tf.concat(locs, axis=0)  # (B, S, pred_len, 1)
        scale_all = tf.concat(scales, axis=0)  # (B, S, pred_len, 1)
        sample_all = tf.concat(samples, axis=0) if config.return_samples or config.mode == "sample" else None

        return ForecastGenerationOutput(
            predictions=predictions,
            samples=sample_all,
            loc=loc_all,
            scale=scale_all,
        )

    def _sample_chunk(
        self,
        xc: tf.Tensor,
        sc: tf.Tensor,
        total_rows: int,
        row0: int,
        S: int,
        config: ForecastGenerationConfig,
        pred_len: int,
        noise_steps,
    ):
        c = int(xc.shape[0])
        xr = tf.repeat(xc, [S] * c, axis=0)  # (c*S, enc, feat) block-repeated per original row
        sr = tf.repeat(sc, [S] * c, axis=0)  # (c*S, 1)

        state = self.update_generation_state(self.initialize_generation_state(xr, sr))
        cur = xr[:, -1:, :]  # first decoder input = last encoder target (c*S, 1, 1)

        traj, locs, scales = [], [], []
        for t in range(pred_len):
            params, state = self.decode_step(cur, sr, state, training=False)
            loc, scale = params["loc"], params["scale"]
            if config.mode == "greedy":
                step_out = loc
            else:  # ancestral / sample
                if noise_steps is not None:
                    noise = noise_steps[t][row0 : row0 + c * S]
                else:
                    noise = tf.random.normal(tf.shape(loc))
                step_out = loc + scale * noise
            traj.append(step_out)
            locs.append(loc)
            scales.append(scale)
            cur = step_out

        target_dim = int(getattr(self.output_distribution, "target_dim", 1))
        trajS = tf.reshape(tf.concat(traj, axis=1), [c, S, pred_len, target_dim])
        locS = tf.reshape(tf.concat(locs, axis=1), [c, S, pred_len, target_dim])
        scaleS = tf.reshape(tf.concat(scales, axis=1), [c, S, pred_len, target_dim])
        return trajS, locS, scaleS

    def _generate_teacher_forced(
        self,
        x: tf.Tensor,
        static: tf.Tensor,
        decoder_feature: tf.Tensor,
        pred_len: int,
    ) -> ForecastGenerationOutput:
        state = self.update_generation_state(self.initialize_generation_state(x, static))
        cur = decoder_feature[:, 0:1, :]  # cluster first decoder input from supplied lagged target
        locs, scales = [], []
        for t in range(pred_len):
            params, state = self.decode_step(cur, static, state, training=False)
            loc, scale = params["loc"], params["scale"]
            locs.append(loc)
            scales.append(scale)
            if t + 1 < pred_len:
                cur = decoder_feature[:, t + 1 : t + 2, :]
        predictions = tf.concat(locs, axis=1)  # (B, pred_len, target_dim)
        return ForecastGenerationOutput(
            predictions=predictions,
            loc=tf.expand_dims(predictions, axis=1),
            scale=tf.expand_dims(tf.concat(scales, axis=1), axis=1),
        )
