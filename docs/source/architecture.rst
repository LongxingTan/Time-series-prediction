Architecture
============

TFTS separates model architecture from task and inference policy.  The public
composition is:

::

   BackboneConfig -> Backbone -> TaskHead -> TaskModel
                                      |
                                      +-> GenerationStrategy -> Sampler -> Processors

Contracts
---------

``TimeSeriesBatch`` is the only public model-input vocabulary.  It names past
and future values, real and categorical temporal features, static features,
observed masks, padding, and labels.  Architecture-specific input names stay
behind ``BackboneAdapter``.  A backbone with a specialized input structure can
implement ``adapt_batch(batch)``; adding it never requires a model-name branch
in the shared adapter.  ``TimeSeriesBatch.from_inputs`` accepts canonical named
fields or a single tensor shorthand for ``past_values``; it deliberately does
not infer positional inputs or architecture-specific field names.
Masks consistently use ``1``/``True`` for observed or valid positions.

Spatial inputs keep tensor arrangement and relational topology independent.
The rank of ``past_values`` declares its ``SpatialArrangement``: rank 3 is a
plain sequence, rank 4 is a set, and rank 5 is a grid.  An optional typed
``SpatialStructure`` sidecar carries topology such as dense adjacency.  Models
independently declare one arrangement and the
``TopologyInput`` values they consume.

Shared structure fields remain constants at the ``tf.data`` boundary.  They are
attached after dataset batching rather than copied into every window.  The
``per_node`` forecasting adapter is an explicit independent-series fallback for
spatial values; it restores the spatial axes on every forecast output.  It is
not available to graph models that require set-valued input, and there is no
flattening fallback because flattening changes the target and output contracts.

For TFT, historical targets are always included among the encoder real
variables.  Therefore ``encoder_real_dim`` is the number of target channels
plus ``past_time_features`` channels.  ``decoder_real_dim`` counts known
``future_time_features`` only.  Temporal categorical variables use
``past_categorical_features`` and ``future_categorical_features`` with one
channel per configured cardinality.

Backbones declare immutable ``BackboneCapabilities`` in the model registry.
Task factories resolve those capabilities once while constructing a head.  The
resulting adapter performs only batch-dependent arrangement and topology checks
at execution time; it does not look the model up in the registry on each call. A
backbone that exposes a temporal sequence can support reconstruction tasks; a
native forecaster does not automatically qualify as a representation backbone.

Task models return named ``ModelOutput`` subclasses.  Normal calls return the
task's primary tensor for Keras interoperability; pass ``return_dict=True`` to
receive the full structured result.

Task composition
----------------

Backbone configuration and task configuration are separate serializable
objects.  For example:

.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModelForForecasting, ForecastTaskConfig

   backbone_config = AutoConfig.for_model("bert")
   task_config = ForecastTaskConfig(
       prediction_length=24,
       target_dim=1,
       head="quantile",
       quantiles=(0.1, 0.5, 0.9),
   )
   model = AutoModelForForecasting.from_config(backbone_config, task_config)
   output = model(tf.random.normal([8, 96, 4]), return_dict=True)

Forecasting, classification, imputation, and anomaly detection each own their
head, loss semantics, and typed output.  Anomaly calibration is a fitted service
separate from the neural reconstruction head.

Generation
----------

Generation is an inference policy, not a model mixin.  ``model.generate``
selects one rollout strategy: direct, recursive-window, or a backbone's native
autoregressive decoder.  A sampler selects values from probabilistic outputs,
then processors enforce continuous-value constraints before feedback.

``ForecastGenerationConfig`` contains only serializable policy.  Custom
samplers and processors are runtime dependencies passed to ``generate`` rather
than embedded in saved configuration.

All rollout strategies resolve samplers through the same value-selection API:
``sampler="auto"`` draws from a distribution when the model exposes one and
otherwise uses its predictions. Use ``sampler="mean"`` explicitly for
deterministic feedback from a probabilistic model. Direct forecasts also honor
custom samplers, ``num_samples``, aggregation, and ``return_samples``; processors
run on selected trajectories before aggregation. Explicit distribution sampling
without a distribution raises an error.

Data window selection is independent of forecast sampling.
``tfts.data.window_sampling`` owns ``sampled_windows`` and ``final_windows``;
``WindowedTrainer`` uses these helpers to prepare each epoch's examples.
The historical imports from ``tfts.training`` and ``window_trainer`` remain
aliases. Synthetic dataset creation and padding also belong to ``data``.

Incremental decoding and training
--------------------------------

Seq2seq, WaveNet, Transformer, and DeepAR implement two hooks:
``initialize_decode(batch, horizon=..., training=...)`` returns a
``DecodeSession(context, state, previous)``; ``decode_step(previous, state,
context, offset=..., training=...)`` returns a ``StepOutput``. Context is fixed
conditioning, while state is a nest of tensors. RNNs carry recurrent states,
WaveNet carries bounded delay buffers, and Transformer carries projected
self-attention keys and values. Initialization encodes the history once.

``GenerationEngine`` owns the TensorFlow loop. A decoder can emit one timestep
or a block; the loop advances by that block's time dimension and crops the last
block to the requested horizon. Each block has shape
``[batch, block_length, ..., target_dim]``. Distribution parameters must use the
same leading batch and time dimensions. ``RolloutOutput.predictions`` and
``distribution_params`` retain raw model outputs for losses, whereas ``values``
contains sampled and processed predictions.

``FeedbackPolicy`` is separate from ``ValueSampler``. It chooses between a
teacher block and the selected model block, only after that block has been
predicted. Probability one means teacher forcing; zero means model feedback.
The Bernoulli choice is per example and block, shared across target channels.
Missing teacher elements use model feedback. Model feedback is detached by
default; advanced callers of ``decode`` can set ``detach_feedback=False``.
Seeded teacher choices and forecast samples use separate random streams.

Epoch-based teacher-probability and noise-strength schedules live in
``tfts.training.schedules``. Their historical module imports remain aliases.
``scheduled_sampling_decode`` is a compatibility adapter; new custom training
loops should call ``tfts.generation.decode`` with a ``TimeSeriesBatch`` and an
explicit ``teacher_probability``. Exposure-bias noise helpers remain training
input augmentation in ``training.exposure_bias``: they do not modify returned
forecasts and are not forecast processors.

The forecasting task routes Keras ``(x, y)`` targets to this policy during
training. The schedule is stored in the task config and evaluated using
``optimizer.iterations``:

.. code-block:: python

   config = AutoConfig.for_model("seq2seq")
   model = AutoModelForForecasting.from_config(
       config,
       prediction_length=24,
       target_dim=2,
       teacher_probability=1.0,
       teacher_final_probability=0.2,
       teacher_decay_steps=10000,
       feedback_sampler="mean",
   )
   model.compile(optimizer="adam", loss="mse")
   # x: [batch, context, 2]; y: [batch, 24, 2]
   model.fit(x, y)
   forecast = model.generate(x, prediction_length=48)

Without a decay duration the teacher probability stays constant. A value of
zero trains with predictions as feedback. ``feedback_sampler="sample"``
requires a probabilistic output head. Teacher selection is independent of
dropout: ``model.forward(batch, training=False, teacher_probability=1.0)``
explicitly evaluates teacher-conditioned predictions. Distribution likelihood
evaluation uses teachers when targets are supplied. ``generate`` always clears
targets and uses model feedback.

An optional ``decode_teacher_forced`` hook provides fast full-teacher execution
with the same weights. Transformer uses shifted targets with causal attention;
DeepAR uses its shared recurrent cells. Missing teachers and mixed feedback use
the incremental path. Transformer supports ``use_cache=False`` as an equivalent
full-prefix reference path. WaveNet's history encoder kernels are separate from
its kernel-two incremental decoder transitions.

The historical ``DecoderV1`` and ``DecoderV2`` imports are aliases for a single
decoder implementation. Legacy backbone ``scheduled_sampling`` settings are
accepted for direct backbone calls; task-model training uses the explicit task
policy above. Transformer decoder weights changed with causal cached decoding;
old decoder checkpoints require migration or retraining. New task-model
checkpoints round-trip through the normal Keras save/load API.

Direct models continue to bypass the autoregressive loop. Diffusion, flow, or
tokenized models can implement a specialized ``RolloutStrategy`` without
changing the canonical batch or forecast result. The time-axis engine does not
pretend that denoising iterations are forecast timesteps.

Extension rules
---------------

To add a backbone, register its config, implementation, and truthful output
capabilities.  To add a task, define a frozen task config, a small head, and a
typed task model.  To add inference behavior, implement ``RolloutStrategy``,
``ValueSampler``, or ``ForecastProcessor`` without changing a backbone or head.
