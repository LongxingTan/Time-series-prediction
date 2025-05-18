.. Time-series-prediction documentation master file, created by
   sphinx-quickstart on Tue Mar  8 13:01:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TFTS Documentation
==================================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

TFTS (TensorFlow Time Series) supports state-of-the-art deep learning time series models for production, research and data competitions. Specifically, the package provides:

* Flexible and powerful modular design for time series task
* Easy-to-use advanced SOTA deep learning models
* Allow training on CPUs, single and multiple GPUs, TPU


Quick Start
-----------------

1. Requirements
~~~~~~~~~~~~~~~~~~

To get started with `tfts`, follow the steps below:

* Python 3.7 or higher
* `TensorFlow 2.x <https://www.tensorflow.org/install/pip>`_ installation instructions


2. Installation
~~~~~~~~~~~~~~~~~~
Now you are ready, proceed with

.. code-block:: shell

    $ pip install tfts

2. Learn more
~~~~~~~~~~~~~~~~~~

Visit :ref:`Quick start <quick-start>` to learn more about the package.


Tutorials
----------
The :ref:`Tutorials <tutorials>` section provides guidance on

- how to :ref:`prepare datasets<prepare_data>` for single-value, multi-value, single-step, and multi-steps prediction
- how to :ref:`use models<train_models>` and implement new ones.


Models
---------

1. Design a Custom Model with TFTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModel

   def build_model(use_model, input_shape):
      inputs = tf.keras.layers.Input(input_shape)
      config = AutoConfig.for_model(use_model)

      backbone = AutoModel.from_config(config)
      outputs = backbone(inputs)
      model = tf.keras.Model(inputs, outputs=outputs)

      optimizer = tf.keras.optimizers.Adam(0.003)
      loss_fn = tf.keras.losses.MeanSquaredError()

      model.compile(optimizer, loss_fn)
      return model

   model = build_model(use_model="bert", input_shape=(24, 3))
   model.summary()


2. More highlights
~~~~~~~~~~~~~~~~~~~~~~~~

The tfts library supports the SOTA deep learning models for time series.

- `TFTS BERT model <https://github.com/LongxingTan/KDDCup2022-Baidu>`_ — 3rd place in `Baidu KDD Cup 2022 <https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction>`_
- `TFTS Seq2Seq model <https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction>`_ — 4th place in `Alibaba Tianchi ENSO prediction <https://tianchi.aliyun.com/competition/entrance/531871/introduction>`_
- :ref:`Learn more models <models>`


Tricks
----------
Visit :ref:`Tricks <tricks>` if you want to know more tricks to improve the prediction performance.


Citation
------------
If you find tfts project useful in your research, please consider cite:

.. code-block:: text

   @misc{tfts2020,
     author = {Longxing Tan},
     title = {Time series prediction},
     year = {2020},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/longxingtan/time-series-prediction}},
   }


.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 6

   quick-start
   tutorials
   models
   tricks
   api
