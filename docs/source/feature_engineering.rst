Data preparation
================

``FramePipeline`` is the dataframe preparation path used by ``Forecaster``.
It validates a ``Schema``, regularizes timestamps, creates deterministic calendar
features, splits by time, fits scalers and vocabularies on training rows, and
constructs lazy windows. The old ``DataProcessor``, ``TimeSeriesSequence`` and
``AutoPreprocessor`` entry points have been removed.

Fit once, transform with saved state
------------------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tfts.data import FramePipeline, Schema

   frame = pd.DataFrame({
       "date": pd.date_range("2024-01-01", periods=80, freq="D"),
       "value": np.arange(80, dtype=float),
   })
   pipeline = FramePipeline(Schema("date", ("value",)), context=12, horizon=4)
   prepared, train_windows, valid_windows = pipeline.fit(frame, validation=0.2)
   train = pipeline.dataset(prepared, train_windows, batch_size=16, shuffle=True)
   valid = pipeline.dataset(prepared, valid_windows, batch_size=16)

   # Applies the fitted vocabulary and scalers, without learning from these rows.
   transformed = pipeline.transform(frame)
   restored = FramePipeline.from_dict(pipeline.to_dict())

The schema declares target, observed-past, known-future and static columns.
Categorical vocabularies reserve code zero for unseen values. Missing categories
also have a false observation mask, so they remain distinguishable from unseen
but observed categories. ``prediction(history, future=...)`` requires known
future covariates for the next horizon and rejects future target columns.

Missing rows and values
-----------------------

Supply ``freq`` explicitly when a training history contains gaps. Numeric time
uses a positive numeric interval; datetime time uses a pandas frequency such as
``"h"`` or ``"D"``. Duplicates and off-grid timestamps raise. Regularization inserts
missing rows; it never fills a target from later observations. Missing targets
and covariates become zero payloads with false masks at the TensorFlow boundary.
Infinite values raise before training.

The temporal split excludes windows whose labels cross the partition boundary.
Validation contexts may use earlier training observations. Only training rows
fit scalers and categorical vocabularies. Validation extrema cannot change the
training normalization.

Stage ownership
---------------

* ``data/schema.py``: column roles and immutable feature descriptions.
* ``data/regularize.py``: frame validation, grouping and time grids.
* ``data/features/``: calendar, lag, rolling, Fourier and categorical transforms.
* ``data/split.py``: temporal partitions and label-boundary checks.
* ``data/scaling.py``: one mask-aware channel scaler family.
* ``data/windows.py``: shared window geometry and compact window indices.
* ``data/padding.py``: NumPy sequence padding and validity masks.
* ``data/tf_dataset.py``: gather tensor batches from flat arrays on demand.
* ``data/arrays.py``: fitted preprocessing for already-windowed arrays.

``WindowIndex`` stores integer starts, group ids and decoder lengths. Window
objects are created only when accessed. ``SequenceMaterializer.materialize`` uses
the same gather path as ``dataset``; it does not implement another set of window
or mask semantics.

Custom feature pipelines
------------------------

.. code-block:: python

   from tfts.data.features import FeaturePipeline, LagTransform, RollingTransform

   features = FeaturePipeline([
       LagTransform("value", lags=(1, 7)),
       RollingTransform("value", windows=(7,), functions=("mean",)),
   ])
   # Fit learned transforms on a training partition only.
   features.fit(frame.iloc[:64], Schema("date", ("value",)))
   prepared_features = features.transform(frame)

These expert transforms return a frame with a feature manifest. Lag and rolling
features must be causal at each forecast origin. Target-derived columns are
observed-past features, not known-future covariates. Fit each learned transform
inside each cross-validation training partition; never fit on the whole frame
before splitting.

``ChannelScaler`` supports standard, minmax, robust, log1p and identity transforms.
An all-missing training channel raises. Log1p requires values greater than -1;
its inverse applies to values, not automatically to distribution means.
