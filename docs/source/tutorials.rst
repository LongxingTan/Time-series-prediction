Tutorials
==========

.. _tutorials:

.. raw:: html

    <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

The following tutorials can be also found as `notebooks on GitHub <https://github.com/longxingtan/time-series-prediction/tree/master/examples/notebooks>`_.

.. _prepare_data:

Train your own data
--------------------------

tfts supports multi-type time series prediction:

- single-value single-step prediction

- single-value multi-steps prediction

- multi-value single-step prediction

- multi-value multi-steps prediction

Feed the input data into the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

	# tf.data.Dataset
	batch_size = 1
	train_length = 10
	predict_sequence_length = 5
	x = tf.random.normal([batch_size, train_length, 1])
	encoder_feature


- tf.data.Dataset

- list or tuple for (x, encoder_feature, decoder_features)

- array for single variable prediction


Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- datetime features
- static features
- dynamic features


.. _train_models:

Train the models
-----------------

- Multi-GPU training with `tf.distribute <https://www.tensorflow.org/guide/keras/distributed_training>`_
- Mixed precision with `tf.keras.mixed_precision <https://www.tensorflow.org/guide/mixed_precision>`_


.. code-block:: python

    import tensorflow as tf
    import tfts
    from tfts import AutoModel, AutoConfig, kerasTrainer

    model_name = 'seq2seq
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(0.003)

    config = AutoConfig.for_model(model_name_or_path)
    model = AutoModel.from_config(config=config, predict_sequence_length=12)

    trainer = KerasTrainer(model, loss_fn=loss_fn, optimizer=optimizer)

    history = trainer.train(
        dataset_train, dataset_val, epochs=10, batch_size=32,
    )



Custom-defined configuration
----------------------------------------

Change the model parameters. If you want touch more parameters in model config, please raise an issue in github.

.. code-block:: python

    import tensorflow as tf
    import tfts
    from tfts import AutoModel, AutoConfig

    config = AutoConfig.for_model('rnn')
    print(config)

    custom_model_config = {
        "rnn_size": 128,
        "dense_size": 128,
    }
    config.update(custom_model_config)
    model = AutoModel('rnn', config=config)


Multi-variables and multi-steps prediction
-------------------------------------------------

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig

	config = AutoConfig.for_model('rnn')
	print(config)

	config.update({
	    "rnn_size": 128,
	    "dense_size": 128,
	})
	print(config)

	model = AutoModel.from_config(config, predict_sequence_length=7)

	x = tf.random.normal([1, 14, 1])
	encoder_features = tf.random.normal([1, 14, 10])
	decoder_features = tf.random.normal([1, 7, 3])
	model()


Custom head for classification or anomaly task
-------------------------------------------------

Set up the custom-defined head layer to do the classification task or anomaly detection task

.. code-block:: python

    import tensorflow as tf
    import tfts
    from tfts import AutoModel, AutoConfig, AutoTuner

    config = AutoConfig.for_model('rnn')
    custom_model_head = tf.keras.Sequential(
        Dense(1)
    )
    model = AutoModel.from_config(config, custom_model_head=custom_model_head)


Custom-defined trainer
----------------------------------------

You could use `tfts trainer`, `custom trainer` or use keras to train directly.

.. code-block:: python

    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    import tfts
    from tfts import AutoModel, AutoConfig
    train_length = 24
    train_features = 15
    predict_sequence_length = 16

    inputs = Input([train_length, train_features])
    config = AutoConfig.for_model('seq2seq')
    backbone = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
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


serve the model with tf-serving

.. code-block:: shell

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig, AutoTuner


.. toctree::
   :titlesonly:
   :maxdepth: 2
