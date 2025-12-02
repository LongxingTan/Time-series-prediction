.. Time-series-prediction documentation master file, created by
   sphinx-quickstart on Tue Mar  8 13:01:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TFTS Documentation
==================================================

.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

TFTS (TensorFlow Time Series) supports state-of-the-art deep learning time series models for production, research, and data competitions. Specifically, the package provides:

* **Modular Design:** A flexible and powerful architecture for time series tasks.
* **SOTA Models:** Easy-to-use access to advanced deep learning models.
* **Hardware Acceleration:** Support for training on CPUs, single/multiple GPUs, and TPUs.


Quick Start
-----------------

1. Installation
~~~~~~~~~~~~~~~~~~~~

To install `tfts <https://github.com/LongxingTan/Time-series-prediction>`_, ensure you meet the following prerequisites:

* Python 3.7+
* `TensorFlow 2.x <https://www.tensorflow.org/install/pip>`_

Once ready, install the package via pip:

.. code-block:: shell

    $ pip install tfts

Alternatively, you can run TFTS using a Docker container:

.. code-block:: shell

    $ docker build -f ./docker/Dockerfile -t "tfts" .
    $ docker run --rm -it --init --ipc=host --network=host --volume=$PWD:/app --gpus all "tfts" /bin/bash


2. Basic Usage
~~~~~~~~~~~~~~~~~~

.. currentmodule:: tfts

The general workflow for training and testing a model involves the following steps:

1.  **Prepare Data:** Build 3D training and validation datasets.
    *   Input shape: ``(samples, train_sequence_length, features)``
    *   Label shape: ``(samples, predict_sequence_length, 1)`` or ``(samples, predict_sequence_length, targets)``
2.  **Instantiate Model:** Use ``AutoConfig`` and ``AutoModel`` to load a specific architecture.
3.  **Setup Trainer:** Initialize a ``Trainer()`` or ``KerasTrainer()`` with your preferred optimizer and loss function.
4.  **Train:** Fit the model to the training dataset and monitor convergence.
5.  **Tune:** Adjust hyperparameters manually or refer to the `tuning example <https://github.com/LongxingTan/Time-series-prediction/blob/master/examples/run_tuner.py>`_.
6.  **Inference:** Load the best checkpoint and apply it to new data.

**Example Code:**

.. code-block:: python

    import tensorflow as tf
    import tfts
    from tfts import AutoConfig, AutoModel, KerasTrainer

    train_length = 36
    predict_sequence_length = 12

    # 1. Load data
    train, valid = tfts.get_data('sine', train_length, predict_sequence_length)

    # 2. Build model: 'seq2seq', 'wavenet', 'transformer', 'rnn', 'tcn', 'bert'
    model_name_or_path = 'seq2seq'
    config = AutoConfig.for_model(model_name_or_path)
    model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)

    # 3. Initialize Trainer
    opt = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    trainer = KerasTrainer(model, loss_fn=loss_fn, optimizer=opt)

    # 4. Train
    trainer.train(train, valid, epochs=30, batch_size=32)

    # 5. Predict
    trainer.predict(valid[0])


3. Advanced Training
------------------------------

3.1 Prepare the Data
~~~~~~~~~~~~~~~~~~~~~~~~
TFTS accepts time series data in a specific 3D format.

*   **Input Shape:** ``(num_examples, train_sequence_length, num_features)``
*   **Output Shape:** ``(num_examples, predict_sequence_length, num_outputs)``

Before training, ensure your raw data is preprocessed into these dimensions. Perform any necessary data cleaning, normalization, or transformation steps prior to reshaping.


3.2 Training Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~
You can customize the training process with different strategies, schedulers, and weight management.

**Multi-GPU / TPU Training**

.. code-block:: python

    from tfts import KerasTrainer

    config = AutoConfig.for_model(model_name_or_path)
    model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)

    optimizer = {
        'class_name': 'adam',
        'config': {'learning_rate': 0.0005}
    }

    strategy = tf.distribute.MirroredStrategy()
    trainer = KerasTrainer(model, strategy=strategy)
    trainer.train(train_gen, valid_gen, optimizer=optimizer, epochs=30)

**Learning Rate Scheduler**

.. code-block:: python

    opt = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
    )

    trainer = KerasTrainer(model)
    trainer.train(train_dataset, valid_dataset, optimizer=opt, loss_fn=loss_fn, lr_scheduler=lr_scheduler)

**Pretrained Weights**

.. code-block:: python

    # Save weights
    model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
    model.save_pretrained("tfts-model")

    # Load weights
    model = AutoModel.from_pretrained("tfts-model")


3.3 Model Serving
~~~~~~~~~~~~~~~~~~~~~~~
Once trained, deploy the model for inference. Ensure the model is saved in a format compatible with your serving environment (e.g., TensorFlow SavedModel or ONNX). Set up an API to handle requests, preprocess input data, and return predictions in real-time.


Tutorials
----------
The :ref:`Tutorials <tutorials>` section provides guidance on:

- :ref:`Preparing datasets <prepare_data>` for single-value, multi-value, single-step, and multi-step prediction.
- :ref:`Using existing models <train_models>` and implementing custom architectures.


Models
---------

1. Design a Custom Model with TFTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can easily integrate TFTS models as backbones within a custom Keras architecture.

.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModel

   def build_model(use_model, input_shape):
      inputs = tf.keras.layers.Input(input_shape)

      # Use TFTS model as a backbone
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


2. Highlights
~~~~~~~~~~~~~~~~~~~~~~~~

TFTS powers competitive solutions in major data science competitions:

- **TFTS BERT:** 3rd place in `Baidu KDD Cup 2022 <https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction>`_ (See `Code <https://github.com/LongxingTan/KDDCup2022-Baidu>`_).
- **TFTS Seq2Seq:** 4th place in `Alibaba Tianchi ENSO prediction <https://tianchi.aliyun.com/competition/entrance/531871/introduction>`_ (See `Code <https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction>`_).

:ref:`View all supported models <models>`


Tricks
-------------
Visit the :ref:`Tricks <tricks>` page to discover techniques for improving prediction performance.


Citation
------------
If you find TFTS useful in your research, please consider citing:

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

   tutorials
   models
   tricks
   api
