"""Autoregressive generation mixin (Transformers-style)."""

from typing import Any, Dict, Optional

import tensorflow as tf

from .configuration import ForecastGenerationConfig
from .engine import GenerationEngine
from .outputs import ForecastGenerationOutput
from .strategy import AncestralSampling, GreedySampling, StepOutput, resolve_feedback_policy, resolve_sampling_strategy


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
        strategy_spec = config.strategy if config.strategy is not None else config.mode
        strategy = resolve_sampling_strategy(strategy_spec)
        feedback = resolve_feedback_policy(config.feedback)
        x, decoder_feature, static = self.prepare_generation_inputs(inputs)
        if x is None or static is None:
            raise ValueError("generate() requires inputs with keys 'x' and 'static'.")
        pred_len = int(config.horizon or getattr(self, "predict_sequence_length"))

        if config.mode == "teacher_forced" and config.strategy is None:
            if decoder_feature is None:
                raise ValueError("teacher_forced mode requires inputs['decoder_feature'].")
            if decoder_feature.shape[1] is not None and int(decoder_feature.shape[1]) < pred_len:
                raise ValueError("decoder_feature is shorter than the requested generation horizon.")
            return self._generate_teacher_forced(
                x,
                static,
                decoder_feature,
                pred_len,
                strategy,
                feedback,
                config.processors,
            )

        return self._generate_sampled(x, static, config, pred_len, strategy, feedback)

    def _generate_sampled(
        self,
        x: tf.Tensor,
        static: tf.Tensor,
        config: ForecastGenerationConfig,
        pred_len: int,
        strategy,
        feedback,
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
        if isinstance(strategy, GreedySampling):
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
                x[start:end], static[start:end], N, row0, S, config, pred_len, noise_steps, strategy, feedback
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
        strategy,
        feedback,
    ):
        c = int(xc.shape[0])
        xr = tf.repeat(xc, [S] * c, axis=0)  # (c*S, enc, feat) block-repeated per original row
        sr = tf.repeat(sc, [S] * c, axis=0)  # (c*S, 1)

        state = self.update_generation_state(self.initialize_generation_state(xr, sr))
        cur = xr[:, -1:, :]

        def step_fn(previous, model_state, step):
            params, next_state = self.decode_step(previous, sr, model_state, training=False)
            return StepOutput(
                prediction=self.output_distribution.mean(params),
                state=next_state,
                distribution=self.output_distribution,
                parameters=params,
            )

        def seed_for_step(step):
            if noise_steps is None:
                return None
            # DistributionOutput.sample accepts a stateless seed, not precomputed noise.
            # Fold the global row offset into the seed so chunking stays reproducible.
            return tf.cast(tf.stack([config.seed + step, row0]), tf.int64)

        # Preserve exact legacy seeded trajectories by using their precomputed draws.
        if noise_steps is not None and isinstance(strategy, AncestralSampling):

            def fixed_sample(output, *, step, seed=None, teacher=None, state=None):
                from .strategy import SamplingResult

                noise = noise_steps[step][row0 : row0 + c * S]
                return SamplingResult(output.parameters["loc"] + output.parameters["scale"] * noise, state=state)

            from .strategy import CallableSampling

            active_strategy = CallableSampling(fixed_sample)
        else:
            active_strategy = strategy

        rollout = GenerationEngine(active_strategy, feedback, config.processors).run(
            step_fn, cur, state, pred_len, context={"static": sr}, seed_for_step=seed_for_step
        )
        traj = rollout.values
        locs = tf.concat([step.parameters["loc"] for step in rollout.steps], axis=1)
        scales = tf.concat([step.parameters["scale"] for step in rollout.steps], axis=1)

        target_dim = int(getattr(self.output_distribution, "target_dim", 1))
        trajS = tf.reshape(traj, [c, S, pred_len, target_dim])
        locS = tf.reshape(locs, [c, S, pred_len, target_dim])
        scaleS = tf.reshape(scales, [c, S, pred_len, target_dim])
        return trajS, locS, scaleS

    def _generate_teacher_forced(
        self,
        x: tf.Tensor,
        static: tf.Tensor,
        decoder_feature: tf.Tensor,
        pred_len: int,
        strategy,
        feedback,
        processors,
    ) -> ForecastGenerationOutput:
        state = self.update_generation_state(self.initialize_generation_state(x, static))
        cur = decoder_feature[:, 0:1, :]

        def step_fn(previous, model_state, step):
            params, next_state = self.decode_step(previous, static, model_state, training=False)
            return StepOutput(
                prediction=self.output_distribution.mean(params),
                state=next_state,
                distribution=self.output_distribution,
                parameters=params,
            )

        # The teacher tensor contains each step's decoder input, so the feedback for
        # prediction t is decoder_feature[t + 1].
        teacher = tf.concat([decoder_feature[:, 1:, :], decoder_feature[:, -1:, :]], axis=1)
        rollout = GenerationEngine(strategy, feedback, processors).run(
            step_fn, cur, state, pred_len, teacher=teacher, context={"static": static}
        )
        predictions = rollout.values
        scales = tf.concat([step.parameters["scale"] for step in rollout.steps], axis=1)
        return ForecastGenerationOutput(
            predictions=predictions,
            loc=tf.expand_dims(predictions, axis=1),
            scale=tf.expand_dims(scales, axis=1),
        )
