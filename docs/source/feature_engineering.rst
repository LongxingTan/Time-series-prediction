Feature
===================

.. _feature_engineering:

Feature engineering is crucial for improving time series model performance. TFTS provides built-in utilities to automatically generate powerful features from your time series data.


Overview
--------

TFTS offers automatic feature engineering through the ``TimeSeriesSequence`` data loader. Simply configure which features you want, and they'll be automatically generated and included in your training data.

**Benefits:**
   - Automated feature generation
   - Consistent feature computation across train/valid/test sets
   - Integration with TFTS models
   - Customizable feature sets


Available Features
------------------

Datetime Features
~~~~~~~~~~~~~~~~~

Extract temporal patterns from datetime columns.

**Supported Features:**
   - ``year``, ``quarter``, ``month``, ``week``
   - ``day``, ``dayofyear``, ``dayofweek``
   - ``hour``, ``minute``, ``second``
   - ``is_weekend``, ``is_month_start``, ``is_month_end``
   - ``is_quarter_start``, ``is_quarter_end``
   - ``is_year_start``, ``is_year_end``

**Cyclic Encoding:**

For periodic features (hour, day, month), TFTS can apply sine/cosine transformation to preserve cyclical nature:


..    \\text{sin}_x = \\sin\\left(\\frac{2\\pi x}{\\text{period}}\\right)

..    \\text{cos}_x = \\cos\\left(\\frac{2\\pi x}{\\text{period}}\\right)

**Example:**

.. code-block:: python

   from tfts.data import TimeSeriesSequence

   feature_config = {
       'datetime_features': {
           'type': 'datetime',
           'features': ['hour', 'dayofweek', 'month', 'is_weekend'],
           'time_col': 'timestamp',
           'cyclic': True  # Apply sine/cosine encoding
       }
   }

   data_loader = TimeSeriesSequence(
       data=df,
       time_idx='timestamp',
       target_column='target',
       train_sequence_length=24,
       predict_sequence_length=8,
       feature_config=feature_config
   )


Lag Features
~~~~~~~~~~~~

Create lagged versions of target or feature columns.

**Use Cases:**
   - Capture autocorrelation
   - Model dependencies on past values
   - Create autoregressive features

**Example:**

.. code-block:: python

   feature_config = {
       'lag_features': {
           'type': 'lag',
           'columns': 'target',  # or list of columns
           'lags': [1, 2, 3, 7, 14, 21],  # Lag periods
       }
   }

This creates: ``target_lag_1``, ``target_lag_2``, ..., ``target_lag_21``

**Multiple Columns:**

.. code-block:: python

   feature_config = {
       'lag_features': {
           'type': 'lag',
           'columns': ['target', 'feature1', 'feature2'],
           'lags': [1, 7],
       }
   }


Rolling Window Features
~~~~~~~~~~~~~~~~~~~~~~~

Compute rolling statistics over windows.

**Supported Functions:**
   - ``mean``: Rolling average
   - ``std``: Rolling standard deviation
   - ``min``, ``max``: Rolling extrema
   - ``median``: Rolling median
   - ``sum``: Rolling sum
   - ``var``: Rolling variance
   - ``skew``, ``kurt``: Higher moments

**Example:**

.. code-block:: python

   feature_config = {
       'rolling_features': {
           'type': 'rolling',
           'columns': 'target',
           'windows': [7, 14, 30],  # Window sizes
           'functions': ['mean', 'std', 'min', 'max'],
       }
   }

This creates features like:
   - ``target_roll_7_mean``
   - ``target_roll_7_std``
   - ``target_roll_14_mean``
   - etc.

**Advanced Rolling:**

.. code-block:: python

   feature_config = {
       'rolling_statistics': {
           'type': 'rolling',
           'columns': ['temperature', 'humidity'],
           'windows': [6, 12, 24],  # Hours
           'functions': ['mean', 'std', 'min', 'max'],
           'min_periods': 1,  # Minimum observations required
       }
   }


Transform Features
~~~~~~~~~~~~~~~~~~

Apply mathematical transformations to columns.

**Supported Transforms:**
   - ``log1p``: log(1 + x) - handles zeros
   - ``log``: Natural logarithm
   - ``sqrt``: Square root
   - ``square``: Square
   - ``cbrt``: Cube root
   - ``reciprocal``: 1/x

**Example:**

.. code-block:: python

   feature_config = {
       'transform_features': {
           'type': 'transform',
           'columns': 'target',
           'functions': ['log1p', 'sqrt'],
       }
   }

**Use Cases:**
   - Stabilize variance
   - Handle skewed distributions
   - Normalize scale


Moving Average Features
~~~~~~~~~~~~~~~~~~~~~~~

Exponential and simple moving averages.

**Types:**
   - Simple Moving Average (SMA)
   - Exponential Moving Average (EMA)

**Example:**

.. code-block:: python

   feature_config = {
       'moving_average': {
           'type': 'moving_average',
           'columns': 'target',
           'windows': [7, 14, 30],
           'ma_type': 'both',  # 'sma', 'ema', or 'both'
       }
   }

**EMA Formula:**


..    \\text{EMA}_t = \\alpha \\cdot x_t + (1 - \\alpha) \\cdot \\text{EMA}_{t-1}

..    \\alpha = \\frac{2}{\\text{window} + 1}


Second-Order Features
~~~~~~~~~~~~~~~~~~~~~

Interactions between features.

**Example:**

.. code-block:: python

   feature_config = {
       'interaction_features': {
           'type': '2order',
           'columns': ['feature1', 'feature2'],
           'operations': ['multiply', 'add', 'subtract', 'divide'],
       }
   }


Complete Example
----------------

Comprehensive Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example combining multiple feature types:

.. code-block:: python

   import pandas as pd
   from tfts.data import TimeSeriesSequence
   from tfts import AutoConfig, AutoModel, KerasTrainer

   # Sample data
   df = pd.read_csv('timeseries_data.csv')

   # Comprehensive feature configuration
   feature_config = {
       # Temporal features
       'datetime': {
           'type': 'datetime',
           'features': [
               'hour', 'dayofweek', 'month', 'quarter',
               'is_weekend', 'is_month_start', 'is_month_end'
           ],
           'time_col': 'timestamp',
           'cyclic': True  # Use sine/cosine encoding
       },

       # Lag features
       'lags': {
           'type': 'lag',
           'columns': 'target',
           'lags': [1, 2, 3, 7, 14, 21, 28],  # 1 day to 4 weeks
       },

       # Rolling statistics
       'rolling': {
           'type': 'rolling',
           'columns': 'target',
           'windows': [7, 14, 28],  # Weekly, biweekly, monthly
           'functions': ['mean', 'std', 'min', 'max'],
       },

       # Transformations
       'transforms': {
           'type': 'transform',
           'columns': 'target',
           'functions': ['log1p', 'sqrt'],
       },

       # Moving averages
       'moving_avg': {
           'type': 'moving_average',
           'columns': 'target',
           'windows': [7, 30],
           'ma_type': 'both',
       },
   }

   # Create data loader with automatic feature engineering
   data_loader = TimeSeriesSequence(
       data=df,
       time_idx='timestamp',
       target_column='target',
       group_column=['location'],  # Optional: group by location
       train_sequence_length=168,  # 1 week of hourly data
       predict_sequence_length=24,  # Predict next 24 hours
       batch_size=32,
       feature_config=feature_config,
       mode='train'
   )

   # Train model
   config = AutoConfig.for_model('transformer')
   model = AutoModel.from_config(config, predict_sequence_length=24)
   trainer = KerasTrainer(model)
   trainer.train(data_loader, epochs=50)


Custom Features
---------------

Creating Custom Features
~~~~~~~~~~~~~~~~~~~~~~~~

You can add custom features by preprocessing your dataframe before creating the data loader:

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Load data
   df = pd.read_csv('data.csv')
   df['timestamp'] = pd.to_datetime(df['timestamp'])

   # Custom features
   df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
   df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)

   # Custom domain-specific features
   df['is_business_hours'] = df['timestamp'].dt.hour.between(9, 17)
   df['is_peak_hours'] = df['timestamp'].dt.hour.isin([8, 9, 17, 18])

   # Weather impact (example)
   df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

   # Now create data loader
   data_loader = TimeSeriesSequence(
       data=df,
       time_idx='timestamp',
       target_column='target',
       train_sequence_length=24,
       predict_sequence_length=8,
   )


Using Feature Registry
~~~~~~~~~~~~~~~~~~~~~~

Register custom feature functions:

.. code-block:: python

   from tfts.features import registry

   @registry
   def add_custom_business_features(df, config):
       \"\"\"Add business-specific features.\"\"\"
       df['is_business_day'] = df['timestamp'].dt.dayofweek < 5
       df['is_holiday'] = df['timestamp'].isin(holidays)
       df['days_to_holiday'] = (df['timestamp'] - next_holiday).dt.days
       return df

   # Use in feature_config
   feature_config = {
       'custom': {
           'type': 'custom',
           'function': add_custom_business_features,
       }
   }


Best Practices
--------------

Feature Selection
~~~~~~~~~~~~~~~~~

**Start Simple:**
   Begin with basic datetime features and a few lags. Add complexity gradually based on validation performance.

**Domain Knowledge:**
   Incorporate domain-specific patterns (e.g., business hours, holidays, events).

**Avoid Leakage:**
   Never use future information. Lag features must use only past data.

**Handle Missing Values:**
   Forward-fill or interpolate missing values before feature engineering.


Feature Scaling
~~~~~~~~~~~~~~~

**Built-in Normalization:**

.. code-block:: python

   from tfts.features import Normalizer

   normalizer = Normalizer(method='standard')  # or 'minmax', 'robust'
   df_normalized = normalizer.fit_transform(df)


**Per-Group Normalization:**

.. code-block:: python

   # Normalize within each group (e.g., per store, per sensor)
   df_normalized = df.groupby('group_id').apply(
       lambda x: (x - x.mean()) / x.std()
   )


Memory Optimization
~~~~~~~~~~~~~~~~~~~

**For Large Datasets:**

1. **Generate features on-the-fly:**

.. code-block:: python

   class OnTheFlyFeatures(tf.keras.utils.Sequence):
       def __getitem__(self, idx):
           # Load batch
           batch = self.load_batch(idx)
           # Generate features
           batch = self.add_features(batch)
           return batch

2. **Use efficient data types:**

.. code-block:: python

   # Downcast numeric types
   df['hour'] = df['hour'].astype('int8')
   df['dayofweek'] = df['dayofweek'].astype('int8')

3. **Chunked processing:**

.. code-block:: python

   for chunk in pd.read_csv('large_file.csv', chunksize=10000):
       chunk = add_features(chunk)
       process(chunk)


Feature Importance Analysis
----------------------------

Analyzing Feature Impact
~~~~~~~~~~~~~~~~~~~~~~~~

Use built-in methods to understand which features matter:

**Method 1: Permutation Importance**

.. code-block:: python

   from sklearn.inspection import permutation_importance

   # Train model with all features
   model.fit(X_train, y_train)

   # Compute importance
   result = permutation_importance(
       model, X_valid, y_valid,
       n_repeats=10,
       random_state=42
   )

   # Plot importance
   import matplotlib.pyplot as plt
   sorted_idx = result.importances_mean.argsort()
   plt.barh(feature_names[sorted_idx], result.importances_mean[sorted_idx])
   plt.xlabel('Permutation Importance')


**Method 2: SHAP Values**

.. code-block:: python

   import shap

   # Create explainer
   explainer = shap.Explainer(model)
   shap_values = explainer(X_valid)

   # Plot
   shap.summary_plot(shap_values, X_valid)


Feature Engineering Workflows
------------------------------

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Baseline:** Train with minimal features
2. **Iterate:** Add feature groups one at a time
3. **Validate:** Check impact on validation metrics
4. **Prune:** Remove features that don't help
5. **Optimize:** Fine-tune feature parameters


Production Workflow
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Define feature pipeline
   from sklearn.pipeline import Pipeline
   from tfts.features import FeatureEngineer

   feature_pipeline = Pipeline([
       ('datetime', DatetimeFeatures()),
       ('lags', LagFeatures(lags=[1, 7, 14])),
       ('rolling', RollingFeatures(windows=[7, 14])),
       ('normalize', Normalizer(method='standard')),
   ])

   # 2. Fit on training data
   feature_pipeline.fit(train_data)

   # 3. Transform train/valid/test consistently
   X_train = feature_pipeline.transform(train_data)
   X_valid = feature_pipeline.transform(valid_data)
   X_test = feature_pipeline.transform(test_data)

   # 4. Save pipeline
   import joblib
   joblib.dump(feature_pipeline, 'feature_pipeline.pkl')

   # 5. Load in production
   pipeline = joblib.load('feature_pipeline.pkl')
   X_new = pipeline.transform(new_data)


Common Patterns
---------------

Seasonal Decomposition Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from statsmodels.tsa.seasonal import seasonal_decompose

   # Decompose time series
   result = seasonal_decompose(df['target'], model='additive', period=24)

   # Add components as features
   df['trend'] = result.trend
   df['seasonal'] = result.seasonal
   df['residual'] = result.resid


Fourier Features for Seasonality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   def add_fourier_features(df, period, K=5):
       \"\"\"Add Fourier terms for capturing seasonality.\"\"\"
       t = np.arange(len(df))
       for k in range(1, K + 1):
           df[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
           df[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
       return df

   # Daily seasonality (period=24 for hourly data)
   df = add_fourier_features(df, period=24, K=3)

   # Weekly seasonality (period=168 for hourly data)
   df = add_fourier_features(df, period=168, K=2)


Event/Holiday Features
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import holidays

   # Get country holidays
   us_holidays = holidays.US()

   # Add holiday indicators
   df['is_holiday'] = df['date'].isin(us_holidays).astype(int)

   # Days to/from nearest holiday
   holiday_dates = pd.Series(list(us_holidays.keys()))
   df['days_to_holiday'] = df['date'].apply(
       lambda x: (holiday_dates - x).abs().min().days
   )


Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**NaN values after feature engineering:**
   - Check for insufficient historical data for lag/rolling features
   - Use ``min_periods`` parameter
   - Forward-fill or interpolate missing values

**Memory errors:**
   - Reduce number of features
   - Use chunked processing
   - Downcaste data types

**Poor performance despite many features:**
   - May be overfitting - try feature selection
   - Some features may add noise
   - Consider feature interactions


See Also
--------

- :doc:`data_preparation` - Data loading and preprocessing
- :doc:`models` - Model selection and configuration
- :doc:`training` - Training strategies
- :doc:`tricks` - Performance optimization tips
