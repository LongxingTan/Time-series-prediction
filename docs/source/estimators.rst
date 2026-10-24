Estimators and model extensions
===============================

Use ``Forecaster`` for forecasting or ``Estimator(task=...)`` for other tasks.
They share one implementation for fitting, prediction, and saving. Construction
only stores parameters. ``fit`` returns the estimator; learned objects are
``model_``, ``preprocessor_``, ``history_``, and ``fit_report_``. Calling ``fit``
again creates a fresh model when constructed with a model name. A supplied task
model instance is reused. ``set_params`` invalidates fitted state.

Forecast a dataframe
--------------------

Declare columns with the existing ``TimeSeriesSchema``. No additional shape
contract is required.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import tfts
   from tfts.features import TimeSeriesSchema

   df = pd.DataFrame({
       "date": pd.date_range("2024-01-01", periods=40),
       "sales": np.arange(40, dtype=float),
   })
   forecaster = tfts.Forecaster(
       "dlinear", context=8, horizon=3,
       schema=TimeSeriesSchema("date", ("sales",)), epochs=1,
   )
   forecaster.fit(df, verbose=0)
   forecast = forecaster.predict(df.tail(8))
   assert len(forecast) == 3
   forecaster.save("sales_model")
   restored = tfts.Forecaster.load("sales_model")
   pd.testing.assert_frame_equal(restored.predict(df.tail(8)), forecast)

Predictions contain the group columns, time column, ``target`` (the target
column name), and ``prediction`` in original units. Quantile heads add columns
such as ``q0.1``. For multiple targets, each timestamp has one row per target.
``predict_output`` returns the typed task output, including distribution
parameters or quantiles. Quantile point predictions use the quantile nearest
0.5, rather than claiming to be a distribution mean.

``validation=0.2`` holds out the final 20 percent of rows in each group.
Scalers and categorical vocabularies fit on training rows only. Validation
windows may use training history, but every validation label is after the
cutoff; training labels never cross it. ``validation=None`` trains on all rows
without automatic early stopping. Window datasets gather slices from flat
arrays rather than storing a separate copy of each overlapping window.

Use ``FeatureSpec(name, FeatureRole.KNOWN_FUTURE)`` for supplied future
covariates and ``calendar=("month", "dayofweek")`` for derived calendar
features. The chosen model must support those roles. ``predict(history,
future=future)`` requires exactly the next fitted horizon of future covariates
for every group, with timestamps and no target columns. Calendar-only features
are generated from timestamps. Future values are never forward-filled.

This initial estimator requires finite, regularly spaced history. It rejects
gaps, duplicate timestamps, unsupported features, and missing future covariates.
Regularize or impute data explicitly before fitting. Scaling is global across
groups (``standard``, ``minmax``, or ``None``). Per-group scaling, automatic gap
repair, and backtesting are not part of this change. Tensor users can continue
to supply observation masks through ``TimeSeriesBatch`` or canonical dictionaries.

Prepared arrays and Keras
-------------------------

``Estimator("sequence_regression", model="rnn").fit(X, y)`` accepts arrays
with shapes ``(B,T,C)`` and ``(B,T,D)``. Forecast labels have shape ``(B,H,D)``;
classification labels have shape ``(B,)``. Output dimensions are inferred during
fit without modifying the caller's config. Supply ``validation_data=(Xv, yv)``
for arrays; the estimator does not split overlapping windows automatically.
Split raw time periods or groups before constructing those arrays.

Use ``head="quantile", quantiles=(0.1, 0.5, 0.9)`` to choose a registered
forecast head and its default pinball objective. Override with
``objective="mae"`` or a ``ComponentSpec`` when appropriate. Objectives return
``LossResult(total, weight)`` so masked losses aggregate correctly across unequal
batches, including quantile and likelihood losses. Float32 loss reductions and
optimizer loss scaling support mixed precision.

An estimator fitted from ``tf.data.Dataset`` or a Keras ``Sequence`` leaves
preprocessing to the caller. Dataset classification requires an explicit
``num_labels`` (or ``ClassificationTaskConfig``): the first batch may omit
classes, and streaming datasets cannot safely be scanned. Array classification
infers the class count from all supplied labels. Unlabeled anomaly detection
uses ``past_values`` as reconstruction targets for both arrays and datasets.

For full Keras control, construct a task model with
``AutoModel.from_config`` and call ``compile`` / ``fit`` directly. Choose its
loss with ``configure_objective``; task training uses that objective rather than
Keras' compiled ``loss`` argument. Models consume tensor dictionaries at dataset
boundaries; direct calls still accept ``TimeSeriesBatch`` and return typed
outputs. ``model.predict`` returns plain tensors.

Add a model in one file
-----------------------

Implement normal Keras layers and ``call``. Register the config and capabilities
beside the class. A temporal representation can be used for forecasting,
classification, and sequence regression without changes to those tasks or the
estimator:

.. code-block:: python

   import numpy as np
   import tensorflow as tf
   import tfts
   from tfts.contracts import BackboneCapabilities, ModelInputSpec, OutputPort
   from tfts.models.base import BaseModel, CommonConfig
   from tfts.models.registry import register_model

   class SmallConvConfig(CommonConfig):
       model_type = "example_small_conv"

   @register_model(
       "example_small_conv", config=SmallConvConfig,
       capabilities=BackboneCapabilities(
           output_ports=frozenset({OutputPort.TEMPORAL_SEQUENCE}),
           input_spec=ModelInputSpec(accepted_roles=frozenset()),
       ),
   )
   class SmallConv(BaseModel):
       def __init__(self, predict_sequence_length=1, config=None):
           super().__init__(predict_sequence_length, config or SmallConvConfig())
           self.conv = tf.keras.layers.Conv1D(8, 3, padding="causal", activation="relu")

       def call(self, inputs, training=None):
           return {"sequence_output": self.conv(inputs)}

   x = np.ones((4, 8, 1), dtype=np.float32)
   y = np.ones((4, 3, 1), dtype=np.float32)
   estimator = tfts.Forecaster("example_small_conv", epochs=1)
   estimator.fit(x, y, verbose=0)
   assert estimator.predict(x).shape == y.shape

Import the model's module before resolving its registered name or loading its
artifact. For an in-tree built-in, the registry discovers implementation modules
in ``tfts.models`` automatically. A patch or variate representation declares
``OutputPort.SEQUENCE``; only representations aligned one-to-one with input
timesteps declare ``TEMPORAL_SEQUENCE``. Alignment is checked before a per-step
head runs. Declare only covariates the backbone actually reads. A backbone with
custom inputs can implement ``adapt_batch(batch)`` locally.

There is no required new ``DataSpec``, generic component base, or separate tag
language. Existing configs, batches, output ports, and Keras layers remain the
extension surface. Head factories own their projection, typed output conversion,
and default objective, so adding a registered forecast head does not require
editing the forecasting task's dispatch logic.

Persistence
-----------

Artifacts contain ``model.keras``, ``estimator.json`` (parameters and fitted
preprocessing), and ``manifest.json`` (format/library versions and a state hash).
Unknown format versions and modified state are rejected. Custom components must
be registered before loading. Use ``Estimator`` / ``Forecaster`` for fitted
workflows and ``AutoModel`` for direct model calls and generation.
