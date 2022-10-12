Tutorials
==========

.. _tutorials:

.. raw:: html

    <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

The following tutorials can be also found as `notebooks on GitHub <https://github.com/longxingtan/time-series-prediction/tree/master/notebooks>`_.

Train the models
-----------------

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel

	loss_fn = tf.keras.losses.MeanSquaredError()
	optimizer = tf.keras.optimizers.Adam(0.003)
	lr_scheduler=None
	model = AutoModel('seq2seq', loss_fn=loss_fn, optimizer=optimizer, lr)


Custom-defined configuration
----------------------------------------

.. code-block:: python

	import tensorflow as tf
	import tfts
	from tfts import AutoModel, AutoConfig


.. toctree::
   :titlesonly:
   :maxdepth: 2
