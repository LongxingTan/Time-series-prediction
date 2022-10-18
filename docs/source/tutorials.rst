Tutorials
==========

.. _tutorials:

.. raw:: html

    <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

The following tutorials can be also found as `notebooks on GitHub <https://github.com/longxingtan/time-series-prediction/tree/master/notebooks>`_.

.. _prepare_data:

Prepare the model inputs
--------------------------

- single-value single-step prediction

- single-value multi-steps prediction

- multi-value single-step prediction

- multi-value multi-steps prediction

what's more, the input data feed into the model could be

- dict

- list or tuple


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

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig

	config = AutoConfig('rnn')
	print(config)

	custom_model_params = {
	}

	model = AutoModel('rnn', custom_model_params=custom_model_params)


Auto-tuned configuration
----------------------------------------

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig, AutoTune

	config = AutoConfig('rnn')
	model = AutoModel('rnn', custom_model_params=custom_model_params)

	tuner = AutoTune('rnn')
	tuner.run(config)


Custom head for classification or anomaly task
-------------------------------------------------

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig, AutoTune

	config = AutoConfig('rnn')
	model = AutoModel('rnn', custom_model_params=custom_model_params)

	tuner = AutoTune('rnn')
	tuner.run(config)


Deployment in tf-serving
--------------------------

.. toctree::
   :titlesonly:
   :maxdepth: 2
