Architecture
============

TFTS is converging on five public concepts:

``Config -> Backbone -> Head -> Dataset -> Trainer``

The current release keeps ``BaseModel`` as a compatibility boundary while the
individual architectures are migrated. New infrastructure should follow the
contracts below and must not introduce another top-level abstraction.

Config
------

Model configs inherit from ``CommonConfig`` and serialize canonical field names
such as ``hidden_size``, ``num_layers``, ``intermediate_size``, and ``dropout``.
Legacy names including ``d_model``, ``d_ff``, ``e_layers``, and
``hidden_dropout_prob`` remain accepted through ``attribute_map``. Validation
runs after the complete model-specific constructor, so invalid configs fail
before model construction.

Models and registration
-----------------------

Each architecture is registered next to its implementation::

   @register_model(
       "example",
       config=ExampleConfig,
       tags=("attention",),
       tier="experimental",
   )
   class Example(BaseModel):
       ...

The auto mappings and benchmark model catalog are live compatibility views of
this registry. A new implementation therefore needs one module and one
decorator, without edits to a central mapping. Use ``tier="core"`` only after
the implementation satisfies the core conformance contract.

Task heads
----------

All tensor-producing heads inherit from ``BaseHead``. ``PredictionHead`` maps
the final horizon of ``(batch, time, hidden)`` states to
``(batch, horizon, labels)``. ``ClassificationHead``, ``QuantileHead``, and
``DistributionHead`` share the same hidden-state input contract. Prediction
residual behavior is centralized and supports ``last_value``, ``last_window``,
and ``mean``; old skip-connect flags remain compatibility aliases.

Until a model has been split into a true hidden-state-only backbone, task
wrappers retain their historical behavior. Do not claim arbitrary head
composition for those legacy models.

Factory
-------

Use the task-aware factory for new integrations::

   model = AutoModel.from_config(
       config,
       task="prediction",
       predict_sequence_length=24,
   )

``AutoModelForPrediction`` and the other task-specific classes delegate to this
factory and remain available as compatibility aliases.

Migration rules
---------------

* Keep ``BaseModel`` and ``build_model`` working until each architecture has a
  tested Keras-native replacement.
* Create layers in ``__init__`` or ``build``, never during a forward call.
* Raise on incompatible shapes instead of truncating channels.
* Preserve tensor, list, and named-dictionary inputs during migration.
* Add contract coverage before changing saved-model or output semantics.
