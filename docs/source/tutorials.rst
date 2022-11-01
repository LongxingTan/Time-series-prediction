Tutorials
==========

.. _tutorials:

.. raw:: html

    <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

The following tutorials can be also found as `notebooks on GitHub <https://github.com/longxingtan/time-series-prediction/tree/master/notebooks>`_.

.. _prepare_data:

Train your own data
--------------------------

tfts supports multi-type time series prediction:

- single-value single-step prediction

- single-value multi-steps prediction

- multi-value single-step prediction

- multi-value multi-steps prediction

the input data feed into the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input 1

.. code-block:: python

	"""
	tf.data.Dataset
	"""
	batch_size = 1
	train_length = 10
	predict_length = 5
	x = tf.random.normal([batch_size, train_length, 1])
	encoder_feature


- tf.data.Dataset

- list or tuple for (x, encoder_feature, decoder_features)

- array for single variable prediction


.. _train_models:

Train the models
-----------------

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel

	loss_fn = tf.keras.losses.MeanSquaredError()
	optimizer = tf.keras.optimizers.Adam(0.003)
	lr_scheduler=None
	model = AutoModel('seq2seq', loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lf_scheduler)


Custom-defined configuration
----------------------------------------

Change the model parameters. If you want touch more parameters in model params, please raise an issue in github.

.. code-block:: python

    import tensorflow as tf
    import tfts
    from tfts import AutoModel, AutoConfig

    config = AutoConfig('rnn').get_config()
    print(config)

    custom_model_params = {
        "rnn_size": 128,
        "dense_size": 128,
    }
    model = AutoModel('rnn', custom_model_params=custom_model_params)


Multi-variables and multi-steps prediction
-------------------------------------------------

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig

	config = AutoConfig('rnn').get_config()
	print(config)

	custom_model_params = {
		"rnn_size": 128,
    	"dense_size": 128,
	}

	model = AutoModel('rnn', predict_length=7, custom_model_params=custom_model_params)

	x = tf.random.normal([1, 14, 1])
	encoder_features = tf.random.normal([1, 14, 10])
	decoder_features = tf.random.normal([1, 7, 3])
	model()


Auto-tuned configuration
----------------------------------------

.. code-block:: python

    import tensorflow as tf
    import tfts
    from tfts import AutoModel, AutoConfig, AutoTuner

    config = AutoConfig('rnn').get_config()
    tuner = AutoTuner('rnn')
    tuner.run(config)


Custom head for classification or anomaly task
-------------------------------------------------

Set up the custom-defined head layer to do the classification task or anomaly detection task

.. code-block:: python

    import tensorflow as tf
    import tfts
    from tfts import AutoModel, AutoConfig, AutoTuner

    AutoConfig('rnn').print_config()
    custom_model_head = tf.keras.Sequential(
        Dense(1)
    )
    model = AutoModel('rnn', custom_model_params=custom_model_params, custom_model_head=custom_model_head)


Custom-defined trainer
----------------------------------------

If you are already used to your own trainer, and just want to use tfts models.

.. code-block:: python

    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    import tfts
    from tfts import AutoModel, AutoConfig
    train_length = 24
    train_features = 15
    predict_length = 16

    inputs = Input([train_length, train_features])
    backbone = AutoModel("seq2seq", predict_length=predict_length)
    outputs = backbone(inputs)
    outputs = Dense(1, activation="sigmoid")(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse", optimizer="rmsprop")
    model.fit(x, y)


Deployment in tf-serving
--------------------------

save the model

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig, AutoTuner


serve the model in docker

.. code-block:: shell

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig, AutoTuner


.. toctree::
   :titlesonly:
   :maxdepth: 2
