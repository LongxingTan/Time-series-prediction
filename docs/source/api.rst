API
===

.. currentmodule:: tfts

.. autosummary::
   :toctree: api
   :template: custom-module-template.rst
   :recursive:

   data
   features
   layers
   models
   losses
   training
   tasks

Prepared-array estimators
-------------------------

Use ``tfts.Forecaster()`` for future targets, or
``tfts.Estimator("sequence_regression")`` for targets at each input timestep.
Both use an RNN by default. Array shapes are ``(samples, time, channels)``;
output length and target width are inferred during ``fit``.

.. code-block:: python

   import tfts

   pipe = tfts.Estimator("sequence_regression", bidirectional=True)
   pipe.fit(X_train, y_train, validation_data=(X_valid, y_valid))
   predictions = pipe.predict(X_test)
   pipe.save("outputs/regression")
   restored = tfts.Estimator.load("outputs/regression")

Use bidirectional processing only when the complete input sequence is available
at prediction time. Supply disjoint validation data, splitting by time or group
before constructing windows. Scaling is fitted on training data and saved with
the model; predictions use original target units. Optional ``sample_weight``
has shape ``(B,)``, ``(B,T)``, or ``(B,T,D)``. Validation weights can be supplied
as the third element of ``validation_data``. Losses use a global weighted mean.
All values must be finite, even when the corresponding weight is zero.

DLinear and PatchTST task models accept target histories only. TiDE additionally
accepts ``past_time_features`` and ``future_time_features`` with matching widths
and ordering, inferring the feature encoder width. These named fields accompany
``past_values`` in an input dictionary. Packed extra channels are rejected for
these three models. Raw backbone calls remain compatible with earlier releases.
