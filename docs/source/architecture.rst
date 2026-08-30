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

Extension rules
---------------

To add a backbone, register its config, implementation, and truthful output
capabilities.  To add a task, define a frozen task config, a small head, and a
typed task model.  To add inference behavior, implement ``RolloutStrategy``,
``ValueSampler``, or ``ForecastProcessor`` without changing a backbone or head.
