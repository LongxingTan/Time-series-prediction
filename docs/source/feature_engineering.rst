Feature engineering
===================

.. _feature_engineering:

Feature engineering is crucial for improving time series model performance. TFTS
provides the :class:`tfts.features.AutoFeatureEngineer` to automatically create
lag, rolling-window, second-order and datetime features from your data, plus
:class:`tfts.data.AutoPreprocessor` for missing values and normalization.


Overview
--------

There are two complementary building blocks:

- :class:`tfts.data.TimeSeriesSequence` — windows a ``pandas.DataFrame`` into
  ``(input, target)`` arrays ready for a model.
- :class:`tfts.features.AutoFeatureEngineer` — creates explanatory features
  (lags, rolling statistics, datetime and second-order terms) as new columns.

A typical pipeline is *feature engineer → sequence windowing → train*:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tfts.features import AutoFeatureEngineer
   from tfts.data import TimeSeriesSequence

   df = pd.DataFrame(
       {
           "timestamp": pd.date_range("2020-01-01", periods=400, freq="h"),
           "value": np.random.randn(400).cumsum(),
           "store": ["A"] * 200 + ["B"] * 200,
       }
   )

   # 1. Engineer features
   engineer = AutoFeatureEngineer(
       lags=[1, 2, 3, 24],          # past-target lags
       windows=[6, 24],             # rolling window sizes
       rolling_functions=["mean", "std", "median"],  # stats per window
       add_datetime=True,           # derive datetime features
       datetime_features=["hour", "dayofweek", "month"],
       group_cols=["store"],        # compute lags/stats within each group
   )
   df = engineer.fit_transform(df, time_col="timestamp", target_col="value")
   print(list(df.columns))          # value_lag_1, value_roll_6_mean, timestamp_hour, ...

   # 2. Window into inputs/targets
   loader = TimeSeriesSequence(
       df,
       target_column="value",
       time_idx="timestamp",
       train_sequence_length=24,
       predict_sequence_length=8,
       batch_size=16,
   )
   x, y = loader[0]                 # (batch, lookback, n_features), (batch, horizon, 1)


Shared features for tabular and sequence models
------------------------------------------------

For new pipelines, describe feature availability once with
``TimeSeriesSchema``. Build feature columns and window boundaries once, then
choose the physical layout only at the model boundary. ``TabularMaterializer``
defaults to one row per forecast horizon, which can be passed to LightGBM or
another sklearn-style estimator. ``SequenceMaterializer`` creates the canonical
``TimeSeriesBatch`` used by TensorFlow models.

.. code-block:: python

   from tfts.data import SequenceMaterializer, TabularMaterializer, WindowIndexer, WindowSpec
   from tfts.features import (
       DatetimeTransform,
       CategoricalEncoderTransform,
       FeatureDType,
       FeaturePipeline,
       FeatureRole,
       FeatureSelection,
       FeatureSpec,
       LagTransform,
       TimeSeriesSchema,
   )

   schema = TimeSeriesSchema(
       time_col="date",
       target_cols=("sales",),
       group_cols=("store",),
       features=(
           FeatureSpec("promotion", FeatureRole.KNOWN_FUTURE),
           FeatureSpec("store_type", FeatureRole.STATIC, FeatureDType.CATEGORICAL),
       ),
   )
   feature_pipeline = FeaturePipeline(
       [
           LagTransform("sales", [1, 7, 28]),
           DatetimeTransform(["month", "dayofweek"]),
           CategoricalEncoderTransform("store_type"),
       ]
   )
   prepared = feature_pipeline.fit_transform(train_df, schema)
   windows = WindowIndexer().build(prepared, WindowSpec(context_length=28, prediction_length=7))

   selection = FeatureSelection(
       include_tags={"target_history", "calendar", "encoded"},
       exclude_names={"sales_lag_1"},
   )
   tabular = TabularMaterializer().materialize(prepared, windows, selection=selection)
   # estimator.fit(tabular.X, tabular.y.ravel())  # for a single target

   sequence = SequenceMaterializer().materialize(prepared, windows, selection=selection)
   tf_dataset = SequenceMaterializer.as_tf_dataset(sequence, batch_size=32)

Generated target lags and rolling statistics are observed-past features and
cannot enter decoder inputs. Datetime/Fourier outputs are known-future features
and are available to both the encoder and decoder. The fitted feature manifest
records ordered feature semantics, lineage, required warm-up history, and a
stable fingerprint.


Available features
------------------

The following table maps a requested feature to the constructor argument that
creates it:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - What it creates
   * - ``lags=[1, 2, 7]``
     - ``<target>_lag_1``, ``<target>_lag_2``, ``<target>_lag_7`` (autocorrelation)
   * - ``windows=[6, 24]``
     - Rolling window sizes used by the functions in ``rolling_functions``
   * - ``rolling_functions=["mean", "std", "min", "max", "median"]``
     - ``<col>_roll_<window>_<fn>`` rolling statistics
   * - ``add_datetime=True``
     - Datetime components (``hour``, ``dayofweek``, ``month``, ...) prefixed with the time column
   * - ``datetime_features=[...]``
     - Restrict which datetime components are created
   * - ``add_fourier=True``
     - Fourier (sine/cosine) terms for seasonality
   * - ``group_cols=[...]``
     - Compute lags and rolling stats per group (e.g. per store)

The full list of datetime features supported by ``datetime_features`` includes
``year``, ``month``, ``week``, ``day``, ``dayofyear``, ``dayofweek``, ``hour``,
``minute``, ``second`` and the ``is_*`` helpers (``is_weekend``,
``is_month_start``, ...).


Normalization and missing values
--------------------------------

Use :class:`tfts.data.AutoPreprocessor` to fill missing values and standardize /
normalize columns before modeling:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tfts.data import AutoPreprocessor

   df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [1.0, 1.0, 3.0, 5.0, np.nan]})

   pre = AutoPreprocessor(
       handle_missing="interpolate",  # or "ffill", "drop"
       normalize="standard",          # or "minmax", "robust"
       columns=["a", "b"],
   )
   scaled = pre.fit_transform(df)     # fit + transform on training data
   # later, apply the same transform to validation/test data:
   test_scaled = pre.transform(test_df)
   # and map forecasts back to the original scale:
   original = pre.inverse_transform(scaled)


Registering custom features
---------------------------

Temporarily add one-off features to the feature registry so they are tracked
alongside the engineered ones:

.. code-block:: python

   import numpy as np
   from tfts.features import FeatureRegistry

   registry = FeatureRegistry()
   registry.register(["hour_sin", "is_promo"])

   # build a custom feature quickly
   def add_hour_sin(df, time_col="timestamp"):
       df["hour_sin"] = np.sin(2 * np.pi * df[time_col].dt.hour / 24)
       return df

   df = add_hour_sin(df)
   registry.get_features()          # ["hour_sin", "is_promo"]


Working with static / encoded data
----------------------------------

For a ready-made end-to-end data pipeline (windowing + normalization + batch
generation) you can use :class:`tfts.data.DataProcessor` instead of composing
the blocks above manually — see :doc:`api`. The notebook tutorials in
:doc:`tutorials` (for example ``single_step_weather_prediction``,
``tft_stallion_prediction``) show realistic feature-engineering workflows on
public datasets.


Best practices
--------------

- **Avoid leakage.** Lags and rolling statistics must use only the past (the
  engineer does this automatically — it never looks ahead relative to a window).
- **Fit only on training data.** Call ``fit_transform`` once on the training set
  and reuse the same ``AutoPreprocessor`` / ``AutoFeatureEngineer`` on
  validation and test data.
- **Handle missing values before training.** Use ``handle_missing`` in
  ``AutoPreprocessor`` or forward-fill/interpolate; rolling statistics of short
  histories naturally produce ``NaN`` which can be dropped or filled.
- **Use ``group_cols``** when series share a cross-sectional unit (store, stock,
  sensor) so lags and statistics are comparable within each group.
- **Downcast dtypes** on large datasets (e.g. ``df["hour"] = df["hour"].astype("int8")``)
  to save memory.

See Also
--------

- :doc:`tutorials` — end-to-end forecasting workflows
- :doc:`models` — model selection and configuration
- :doc:`api` — full API reference
