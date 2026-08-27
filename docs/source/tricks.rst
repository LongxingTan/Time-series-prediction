Tricks
======

.. _tricks:

.. note::

    Time series forecasting is a classic "No Free Lunch" scenario. Deep learning
    models in particular require careful tuning of architecture,
    hyper-parameters, and preprocessing strategies to achieve meaningful
    results.

    There is no way to forecast the future blindly: you must first understand
    how trend, seasonality, cyclicity, and noise show up in your data.


Use TFTS flexibly in competitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want the best possible performance, import the TFTS source and modify it
directly — this is how the library is used in the top competition entries:

* The `TFTS BERT model <https://github.com/LongxingTan/KDDCup2022-Baidu>`_
  won 3rd place in `Baidu KDD Cup 2022 <https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction>`_.
* The `TFTS Seq2Seq model <https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction>`_
  won 4th place in the `Tianchi ENSO prediction <https://tianchi.aliyun.com/competition/entrance/531871/introduction>`_.


General tricks
~~~~~~~~~~~~~~

Target transformation
   The target is the most important signal. The model is usually much easier to
   train on a transformed target than on the raw series. A special and common
   transformation is a skip / residual connection (from ResNet) — TFTS exposes
   basic skip connections through the model config. For richer ones, wrap a
   backbone with your own Keras head (see :doc:`models`).

Feature engineering
   Feature engineering is an art. Lag features, rolling statistics, and
   datetime features are usually the highest-value additions; see
   :doc:`feature_engineering`.

Multiple temporal scales
   Train separate models, or predict different components, at different scales
   (hourly vs daily aggregates) and combine the forecasts.

Module usage
   Be careful with layers such as ``Dropout`` or ``BatchNorm`` on regression
   tasks — they behave differently in training and inference and can bias the
   forecast.

Multi-step prediction strategies
   There are several ways to produce a multi-step forecast:

   * train multiple single-step models (one per horizon)
   * add a hidden-size dense layer at the head that outputs the whole horizon
   * use an encoder-decoder structure
   * use an encoder-forecasting structure


Explicit multi-step rollout
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides training for the horizon directly, you can ask the model to roll out
several steps at inference time with the generation policy:

.. code-block:: python

    # forecast several steps ahead with the configured generation policy
    import tensorflow as tf
    from tfts import AutoConfig, AutoModelForForecasting, ForecastGenerationConfig

    model = AutoModelForForecasting.from_config(
        AutoConfig.for_model("dlinear"), prediction_length=8
    )
    # input window: (batch, lookback, n_features)
    window = tf.random.normal([1, 24, 1])

    out = model.generate(
        window,
        generation_config=ForecastGenerationConfig(prediction_length=8),
    )
    print(out.predictions.shape)   # (1, 8, 1)


See also
~~~~~~~~

- :doc:`models` — model selection and configuration
- :doc:`feature_engineering` — creating predictive features
- :doc:`faq` — troubleshooting and training tips
