Architecture
============

TFTS separates backbone architecture, task semantics, and generation. A task
model combines a backbone with a head and objective. Generation uses that task
model through a decoder, processes each emitted chunk, then transforms the
assembled trajectory.

Contracts and tasks
-------------------

``TimeSeriesBatch`` is the public input vocabulary. It names historical and
future values, temporal and static features, observed masks, padding, and labels.
``TimeSeriesBatch.from_inputs`` accepts canonical named fields or a single tensor
as ``past_values``. Masks use one or true for observed and valid positions.
Architecture-specific adaptation belongs inside ``BackboneAdapter`` or the
backbone's ``adapt_batch`` method.

The rank of ``past_values`` identifies its spatial arrangement: rank three is a
sequence, rank four a set, and rank five a grid. ``SpatialStructure`` carries
topology separately. Shared topology is attached after dataset batching.
The explicit ``per_node`` forecasting strategy treats spatial locations as
independent series and restores their axes on output.

Backbone capabilities declare supported output ports and decoding behavior.
Task factories resolve these declarations when constructing the model.
Model calls return named task outputs, such as ``ForecastOutput.predictions``
and ``ClassificationOutput.logits``. Forecasting, classification, imputation, and anomaly
detection own their heads and objectives. Anomaly thresholds require calibration
on normal reference windows after loading a model.

Backbone and task configuration are separate serializable objects:

.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModelForForecasting, ForecastTaskConfig

   model = AutoModelForForecasting.from_config(
       AutoConfig.for_model("bert"),
       ForecastTaskConfig(
           output_chunk_length=24,
           target_dim=1,
           head="quantile",
           quantiles=(0.1, 0.5, 0.9),
       ),
   )
   output = model(tf.random.normal([8, 96, 4]))

Generation
----------

``output_chunk_length`` is the trained task output length. ``GenerationConfig``
uses ``horizon`` for the requested inference length. Its ``mode`` is ``auto``,
``direct``, ``recursive``, or ``native``. Direct decoding emits one model block;
recursive decoding advances a historical window with each predicted value;
native decoding uses the backbone's incremental decoder. Automatic selection
uses the available capabilities and requested horizon.

Every decoder implements ``start``, ``step``, and ``feed``. ``State`` contains
opaque decoder-owned data. ``step`` returns a ``Chunk`` containing ``offset``,
``past``, ``generated``, ``prediction``, optional ``parameters`` and ``quantiles``,
and processor fields ``value``, ``feedback``, and ``seed``.

``StepProcessorList`` runs processors in their listed order and requires exactly
one ``Selector``. ``Mean``, ``Median``, ``Quantile``, and ``Sample`` select values.
Constraints such as ``Clip`` and ``NonNegative`` operate after selection.
``Feedback`` processors can change subsequent conditioning without changing the
emitted forecast. A custom processor returns an updated immutable chunk using
``chunk.replace(...)`` and must preserve the emitted shape.

The default selector uses a distribution mean for one trajectory, distribution
sampling when ``num_samples > 1``, and a median for quantile heads. Stochastic
selectors enable independent trajectories; deterministic selectors do not
create repeated samples merely because ``num_samples`` exceeds one.

.. code-block:: python

   from tfts import GenerationConfig
   from tfts.generation import Mean, NonNegative

   model = AutoModelForForecasting.from_config(
       AutoConfig.for_model("dlinear"), output_chunk_length=8
   )
   history = tf.ones([2, 24, 1])
   forecast = model.generate(
       history,
       config=GenerationConfig(horizon=16),
       processors=[Mean(), NonNegative()],
   )
   assert forecast.predictions.shape == (2, 16, 1)

``run`` assembles chunks and crops the final block to the horizon. Stopping
criteria receive a processed chunk and return a scalar boolean request to stop
between chunks. The current early stopping implementation applies in eager
execution; graph execution runs to the configured horizon.

Generation returns a ``Trajectory`` with ``predictions``, optional ``samples``,
distribution parameters, and quantile values. Samples have shape
``[batch, num_samples, horizon, ..., target_dim]`` and predictions aggregate
them by their mean. ``quantile(q)``, ``to_quantiles(qs)``, and ``numpy()`` support
downstream use. Full-trajectory transforms run after assembly; examples include
``InverseScale``, ``MeanSamples``, and ``RepairQuantileCrossing``. By default,
inverse scaling uses a scaler supplied in batch metadata when available.
Transforms requiring fitting must be fitted before use.

Configuration stores registered component names and keyword arguments using
``ComponentSpec``. Runtime decoder, processor, trajectory-transform, and stopping
objects can be passed directly to ``generate``. Register reusable components in
``tfts.registry`` to reference them from serializable configuration.

Training and feedback
---------------------

Seq2seq, WaveNet, Transformer, and DeepAR provide incremental backbone hooks.
Their native decoder initializes historical conditioning once and carries
architecture-specific state between steps. Task-model training routes targets
to teacher forcing; generation removes future targets from conditioning.

The task configuration controls ``teacher_probability``,
``teacher_final_probability``, ``teacher_decay_steps``, and ``feedback_sampler``.
The schedule uses optimizer iterations. A teacher probability of one feeds
available targets; zero feeds predictions. Missing teacher elements use model
feedback. Parallel teacher-forced execution is a separate declared capability.

Data-window sampling belongs to ``tfts.data.window_sampling`` and remains
independent of forecast sampling. Training schedules live in
``tfts.training.schedules``. Exposure-bias input augmentation lives in
``tfts.training.exposure_bias``.

Extension rules
---------------

Register a backbone with truthful capabilities, a task with its configuration
and objective, or a generation component implementing its public contract.
Keep architecture-specific state inside the decoder and reuse the shared
chunk assembly loop for time-axis generation.
