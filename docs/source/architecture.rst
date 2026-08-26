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
and future values, time features, static features, observed masks, padding, and
labels.  Architecture-specific input names stay behind ``BackboneAdapter``.
Masks consistently use ``1``/``True`` for observed or valid positions.

Backbones declare immutable ``BackboneCapabilities`` in the model registry.
Task factories validate those capabilities before constructing a head.  A
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
