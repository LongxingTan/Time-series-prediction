Tutorials
=========

.. _tutorials:

.. raw:: html

    <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

The following step-by-step guides can also be opened as `notebooks on GitHub <https://github.com/longxingtan/time-series-prediction/tree/master/docs/source/tutorials>`_.

TFTS supports several time series tasks with a consistent, high-level API:

- single-value / multi-value (multivariate) forecasting
- single-step / multi-step forecasting
- probabilistic forecasting with uncertainty quantification
- classification
- anomaly detection

This tutorial walks you through the core workflow — preparing data, building a
model, training, evaluating, and deploying — and then points you to the
end-to-end notebook examples.

Notebook examples
-----------------

Runnable notebooks live in ``docs/source/tutorials/`` and are rendered here in
the documentation. The corresponding runnable plain scripts (no notebook
formatting) live in ``examples/``:

.. toctree::
   :titlesonly:
   :numbered:
   :maxdepth: 2

   tutorials/deepar_ar_demo
   tutorials/patchtst_autoformer_ar_demo
   tutorials/nbeats_ar_demo
   tutorials/tft_stallion_prediction
   tutorials/timesnet_timemixer_m4_demo
   tutorials/mitsui_recursive_lstm_demo
   tutorials/multi_steps_sales_prediction
   tutorials/single_step_weather_prediction

The complete runnable scripts (no notebook formatting) live in
``examples/`` — for example ``run_prediction_simple.py``,
``run_classification.py``, ``run_anomaly.py`` and ``run_tuner.py``.

Quick start
-----------

.. _quickstart:

The fastest way to train and forecast is the pipeline API:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import tfts

   tfts.set_seed(0)

   # 1. Your data: a DataFrame with a time column and a target column.
   df = pd.DataFrame({
       "timestamp": pd.date_range("2020-01-01", periods=500, freq="h"),
       "sales": np.random.randn(500).cumsum(),
   })

   # 2. Build a forecasting pipeline ("dlinear", "transformer", "tft", ...)
   pipe = tfts.pipeline(
       "forecasting",
       model="dlinear",
       lookback=24,
       horizon=8,
       learning_rate=1e-3,
       epochs=10,
       batch_size=16,
   )

   # 3. Train, holding out the last 20% of the series for validation
   history = pipe.fit(df, target_col="sales", time_col="timestamp", epochs=10)

   # 4. Forecast 8 steps ahead from the most recent lookback window
   pred = pipe.predict(steps=8, df=df.tail(24))
   print("predictions shape:", pred.shape)

Under the hood the pipeline composes a backbone (``AutoConfig`` → ``AutoBackbone``)
with a task head (``ForecastTaskConfig``) and a ``Trainer``. You can control
every component yourself (see :ref:`manual_api` below).

Data preparation
----------------

Synthetic / built-in datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tfts.get_data`` returns numpy arrays ready to feed a model:

.. code-block:: python

   import tfts

   # ((train_x, train_y), (valid_x, valid_y))
   train, valid = tfts.get_data("airpassengers", train_length=24, predict_sequence_length=8, test_size=0.2)
   train_x, train_y = train
   valid_x, valid_y = valid
   print(train_x.shape, train_y.shape)   # (n_samples, lookback, n_features)
   print(valid_x.shape, valid_y.shape)

Available datasets include ``"sine"`` and ``"airpassengers"``.

Your own data
~~~~~~~~~~~~~

Feed a pandas ``DataFrame`` to ``TimeSeriesSequence`` which windows the series
into ``(input, target)`` pairs and optional feature engineering:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from tfts.data import TimeSeriesSequence
   from tfts.features import AutoFeatureEngineer

   df = pd.DataFrame(
       {
           "timestamp": pd.date_range("2020-01-01", periods=400, freq="h"),
           "value": np.random.randn(400).cumsum(),
       }
   )

   # Engineers lags, rolling statistics and datetime features for you.
   fe = AutoFeatureEngineer(
       lags=[1, 2, 3, 24], windows=[6, 24],
       rolling_functions=["mean", "std"], add_datetime=True,
       datetime_features=["hour", "dayofweek", "month"],
   )
   df = fe.fit_transform(df, time_col="timestamp", target_col="value")

   data_loader = TimeSeriesSequence(
       df,
       target_column="value",
       time_idx="timestamp",
       train_sequence_length=24,
       predict_sequence_length=8,
       batch_size=16,
       mode="train",
   )
   # data_loader behaves like a tf.keras.utils.Sequence — usable with any trainer.

.. _manual_api:

Building a model manually
-------------------------

The low-level API gives full control over backbone, task head, optimizer and
loss:

.. code-block:: python

   import tensorflow as tf
   import tfts
   from tfts import AutoConfig, AutoModelForForecasting, Trainer, TrainingArguments

   # 1. Data
   train, valid = tfts.get_data("sine", 24, 8)

   # 2. Backbone config + model. The task head ("quantile") adds prediction_length outputs.
   config = AutoConfig.for_model("dlinear")
   model = AutoModelForForecasting.from_config(config, prediction_length=8)

   # 3. Loss and optimizer
   loss_fn = tf.keras.losses.MeanSquaredError()
   optimizer = tf.keras.optimizers.Adam(1e-3)

   # 4. Trainer. strategy="default" keeps the default (single-device) strategy.
   trainer = Trainer(
       model,
       args=TrainingArguments(output_dir="./output", strategy="default"),
   )

   # 5. Train
   trainer.train(train, valid, loss_fn=loss_fn, optimizer=optimizer, epochs=10, batch_size=16)

   # 6. Predict / evaluate
   pred = trainer.predict(valid[0])
   trainer.evaluate(valid)

You can swap the backbone with a single argument and keep the same workflow,
e.g. ``"transformer"``, ``"tft"``, ``"informer"``, ``"nbeats"``, ``"deep_ar"``.

.. note::

   **Multiple GPUs.** By default ``Trainer``/``pipeline`` use ``strategy="auto"``,
   which selects ``MirroredStrategy`` when more than one GPU is present
   (single GPU / CPU fall back to the default strategy). Multi-GPU training
   additionally requires a working **NCCL** installation. The examples above use
   ``strategy="default"`` to keep a single, portable device. To force one GPU
   regardless of how many are visible, set ``CUDA_VISIBLE_DEVICES=0`` before
   launching Python.

Configuring a backbone
~~~~~~~~~~~~~~~~~~~~~~

Each model has a typed, serializable ``AutoConfig``:

.. code-block:: python

   from tfts import AutoConfig

   config = AutoConfig.for_model("transformer")
   print(config)

   config.hidden_size = 256      # modify attributes directly ...
   config.update({"num_layers": 4, "dropout": 0.1})   # ... or via a dict
   config.save_pretrained("./my_config")
   reloaded = AutoConfig.from_pretrained("./my_config")

Multi-variable, multi-step input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models accept batches of shape ``(batch, lookback, n_features)`` and produce
``(batch, prediction_length, n_features)``:

.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModelForForecasting

   model = AutoModelForForecasting.from_config(
       AutoConfig.for_model("transformer"), prediction_length=7
   )
   x = tf.random.normal([4, 14, 10])      # 4 series, 14 lookback steps, 10 features
   out = model(x)                         # (4, 7, 1)

Custom defined trainer
----------------------

You can use the ``tfts`` trainers or plain Keras. ``AutoModel`` subclasses are
also ``tf.keras.Model`` objects:

.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModelForForecasting

   model = AutoModelForForecasting.from_config(
       AutoConfig.for_model("seq2seq"), prediction_length=8
   )
   model.compile(loss="mse", optimizer="rmsprop")

   # A Keras functional model wrapping the TFTS backbone
   inputs = tf.keras.Input(shape=(24, 1))
   outputs = tf.keras.layers.Dense(1, activation="sigmoid")(model(inputs))
   keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
   keras_model.compile(loss="mse", optimizer="rmsprop")

Saving, reloading and inference
-------------------------------

.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModelForForecasting, AutoModel
   from tfts import ForecastGenerationConfig

   model = AutoModelForForecasting.from_config(
       AutoConfig.for_model("dlinear"), prediction_length=8
   )
   _ = model(tf.zeros([1, 24, 1]))   # build the model (needed before saving)
   model.save_pretrained("./my_model")

   # Restore for further training (weights + architecture + task config)
   restored = AutoModel.from_pretrained("./my_model", sample_batch=tf.zeros([1, 24, 1]))

   # Autoregressive / multi-step generation at inference time
   out = restored.generate(
       tf.random.normal([4, 24, 1]),
       generation_config=ForecastGenerationConfig(prediction_length=8),
   )
   print(out.predictions.shape)

For a lightweight Keras archive you can instead call ``tfts.load_model`` (see
:doc:`models` and the ``examples`` README).

Feature engineering
-------------------

TFTS builds lag, rolling-window and datetime features automatically. See
:doc:`feature_engineering` for the full reference:

.. code-block:: python

   from tfts.features import AutoFeatureEngineer
   import pandas as pd

   df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=300, freq="D"),
                      "target": range(300)})
   engineer = AutoFeatureEngineer(
       lags=[1, 2, 7],
       windows=[7, 30],
       rolling_functions=["mean", "median", "std"],
       add_datetime=True,
       datetime_features=["dayofweek", "month"],
   )
   out = engineer.fit_transform(df, time_col="timestamp", target_col="target")
   print(list(out.columns))
   print(out.shape)

Hyper-parameter tuning
----------------------

TFTS integrates Optuna for hyper-parameter search. Define a search space over
model and training parameters, then let ``OptunaTuner`` build, train and score
each trial:

.. code-block:: python

   from tfts import get_data, OptunaTuner

   train, valid = get_data("sine", 24, 8)

   tuner = OptunaTuner(
       train_data=train,          # (x_train, y_train)
       valid_data=valid,          # (x_valid, y_valid)
       predict_sequence_length=8,
       metric="mse",
   )

   best_params, best_score = tuner.search(
       param_space={
           "model_type": ["rnn", "dlinear"],       # categorical
           "learning_rate": [1e-4, 1e-2],          # log-uniform
           "num_layers": [1, 4],                   # int uniform
       },
       n_trials=10,
       epochs=5,
   )
   print("best params:", best_params)

See ``examples/run_tuner.py`` for a full, self-contained tuning script.

Deployment
----------

Models built with Keras export to ``SavedModel`` / ``.keras`` and can be served
with TensorFlow Serving:

.. code-block:: python

   import tfts

   model = tfts.AutoModelForForecasting.from_config(tfts.AutoConfig.for_model("dlinear"), prediction_length=8)
   model.save_pretrained("./my_model")

   # Alternatively, export a single Keras archive for inference
   # tfts.load_model("my_model.keras", compile=False)

See the :doc:`models` and :doc:`faq` pages for model-specific details and
troubleshooting.
