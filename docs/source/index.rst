.. Time-series-prediction documentation master file

TFTS: TensorFlow Time Series
==================================================

.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

Welcome to TFTS (TensorFlow Time Series), a Python library for state-of-the-art deep learning time series analysis. TFTS provides production-ready implementations of cutting-edge models for forecasting, classification, and anomaly detection tasks.

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://badge.fury.io/py/tfts.svg
   :target: https://pypi.python.org/pypi/tfts
   :alt: PyPI Version

.. image:: https://pepy.tech/badge/tfts/month
   :target: https://pepy.tech/project/tfts
   :alt: Downloads

Why TFTS?
---------

TFTS simplifies time series modeling by providing:

**State-of-the-Art Models**
   Access to 20+ pre-implemented deep learning architectures including Transformers, BERT, Informer, Autoformer, and more. All models are optimized for time series tasks and ready for production use.

**Unified API**
   Consistent interface across all models through ``AutoModel`` and ``AutoConfig``. Switch between architectures with a single line of code while maintaining the same workflow.

**Production Ready**
   Built on TensorFlow 2.x with native support for distributed training, mixed precision, TPUs, and TensorFlow Serving. Export models to SavedModel or ONNX formats for deployment.

**Flexible Architecture**
   Modular design allows easy customization of model components, training loops, and data pipelines. Integrate TFTS models as backbones in your custom architectures.

**Comprehensive Tasks**
   Support for forecasting (univariate/multivariate), classification, anomaly detection, and segmentation tasks with task-specific model heads.


Key Features
------------

📈 **Multiple Tasks**
   - Single/multi-step forecasting
   - Probabilistic forecasting with uncertainty quantification
   - Time series classification
   - Anomaly detection
   - Change point detection and segmentation

🚀 **20+ Models**
   - Classic: RNN, LSTM, GRU, Seq2Seq
   - CNN-based: TCN, WaveNet, UNet
   - Transformer-based: Transformer, BERT, Informer, Autoformer, PatchTST, iTransformer
   - Specialized: N-BEATS, DLinear, TFT, DeepAR, RWKV, Diffusion

⚡ **Performance**
   - Multi-GPU training with ``tf.distribute``
   - TPU support for large-scale training
   - Mixed precision training (FP16/BF16)
   - TensorFlow data pipelines for efficient I/O

🔧 **Flexible**
   - Modular layer design for custom architectures
   - Feature engineering utilities (lag features, rolling statistics, datetime features)
   - Custom training loops and callbacks
   - Integration with Keras ecosystem


Quick Start
-----------

Installation
~~~~~~~~~~~~

Install TFTS using pip:

.. code-block:: bash

   pip install tfts

Requirements:
   - Python >= 3.7
   - TensorFlow >= 2.4

For development installation:

.. code-block:: bash

   git clone https://github.com/LongxingTan/Time-series-prediction.git
   cd Time-series-prediction
   pip install -e .


Basic Usage
~~~~~~~~~~~

Here's a minimal example to get started with TFTS:

.. code-block:: python

   import tensorflow as tf
   import tfts
   from tfts import AutoConfig, AutoModel, KerasTrainer

   # 1. Load sample data
   train_length = 24
   predict_length = 8
   train, valid = tfts.get_data('sine', train_length, predict_length)

   # 2. Choose and configure a model
   config = AutoConfig.for_model('transformer')
   model = AutoModel.from_config(config, predict_sequence_length=predict_length)

   # 3. Train the model
   trainer = KerasTrainer(model)
   trainer.train(train, valid, epochs=10)

   # 4. Make predictions
   predictions = trainer.predict(valid[0])


Supported Models
----------------

TFTS provides implementations of state-of-the-art time series models:

**Transformer-Based Models**
   - ``transformer``: Standard Transformer architecture adapted for time series
   - ``bert``: BERT-style bidirectional encoder for representation learning
   - ``informer``: ProbSparse self-attention for long sequence forecasting
   - ``autoformer``: Auto-correlation mechanism for decomposition
   - ``tft``: Temporal Fusion Transformer with interpretable attention
   - ``patch_tst``: Patch-based Transformer for efficient training
   - ``itransformer``: Inverted Transformer treating variates as tokens

**RNN-Based Models**
   - ``rnn``: Configurable RNN with LSTM/GRU cells
   - ``seq2seq``: Encoder-decoder architecture with attention
   - ``deep_ar``: Probabilistic forecasting with autoregressive RNN

**CNN-Based Models**
   - ``tcn``: Temporal Convolutional Network with dilated convolutions
   - ``wavenet``: WaveNet-style architecture with causal convolutions
   - ``unet``: U-Net style encoder-decoder for sequence-to-sequence

**Specialized Models**
   - ``nbeats``: Neural Basis Expansion Analysis for interpretable forecasting
   - ``dlinear``: Simple linear model with decomposition
   - ``rwkv``: RWKV architecture with linear attention
   - ``diffusion``: Diffusion-based probabilistic forecasting
   - ``tide``: Time-series Dense Encoder
   - ``gpt``: GPT-style autoregressive model


User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   tutorials


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   models
   training


.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   feature_engineering
   tricks


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api


.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   examples
   faq


Examples
--------

Real-World Applications
~~~~~~~~~~~~~~~~~~~~~~~

TFTS has been successfully used in production and competitions:

**Competition Wins**
   - 🥉 **3rd Place** - Baidu KDD Cup 2022 (`Code <https://github.com/LongxingTan/KDDCup2022-Baidu>`_)
   - 🎯 **4th Place** - Alibaba Tianchi ENSO Prediction (`Code <https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction>`_)

**Industry Use Cases**
   - Energy demand forecasting
   - Financial time series prediction
   - IoT sensor data analysis
   - Weather and climate modeling
   - Traffic flow prediction


Advanced Examples
~~~~~~~~~~~~~~~~~

**Multi-variate Forecasting**

.. code-block:: python

   import tensorflow as tf
   from tfts import AutoConfig, AutoModel

   # Configure for multi-variate input
   config = AutoConfig.for_model('informer')
   config.num_features = 10  # 10 input features

   model = AutoModel.from_config(config, predict_sequence_length=24)

   # Input: (batch, sequence_length, num_features)
   x = tf.random.normal([32, 96, 10])
   predictions = model(x)  # Output: (32, 24, 1)


**Probabilistic Forecasting**

.. code-block:: python

   from tfts import AutoConfig, AutoModel

   # Use model with uncertainty quantification
   config = AutoConfig.for_model('deep_ar')
   model = AutoModel.from_config(config, predict_sequence_length=24)

   # Get probabilistic predictions
   predictions = model(x)  # Returns distribution parameters


**Custom Feature Engineering**

.. code-block:: python

   from tfts.data import TimeSeriesSequence
   import pandas as pd

   # Configure feature engineering
   feature_config = {
       'datetime': {
           'type': 'datetime',
           'features': ['hour', 'dayofweek', 'month'],
           'time_col': 'timestamp'
       },
       'lags': {
           'type': 'lag',
           'columns': 'target',
           'lags': [1, 2, 3, 7, 14]
       },
       'rolling': {
           'type': 'rolling',
           'columns': 'target',
           'windows': [7, 14],
           'functions': ['mean', 'std']
       }
   }

   # Create data loader with automatic feature engineering
   data_loader = TimeSeriesSequence(
       data=df,
       time_idx='timestamp',
       target_column='target',
       train_sequence_length=24,
       predict_sequence_length=8,
       feature_config=feature_config
   )


.. Performance Benchmarks
.. ----------------------

.. TFTS models have been evaluated on standard benchmarks:

.. .. list-table::
..    :header-rows: 1
..    :widths: 20 20 20 20 20

..    * - Model
..      - ETTh1 (MSE)
..      - Weather (MAE)
..      - Traffic (MSE)
..      - Training Speed
..    * - Transformer
..      - 0.495
..      - 0.245
..      - 0.612
..      - 1.0x
..    * - Informer
..      - 0.472
..      - 0.231
..      - 0.598
..      - 1.2x
..    * - Autoformer
..      - 0.449
..      - 0.217
..      - 0.573
..      - 1.1x
..    * - DLinear
..      - 0.458
..      - 0.223
..      - 0.587
..      - 3.5x

.. *Benchmarks run on single V100 GPU with batch size 32*


Community and Support
---------------------

**Getting Help**
   - 📖 Read the `documentation <https://time-series-prediction.readthedocs.io>`_
   - 💬 Ask questions in `GitHub Discussions <https://github.com/LongxingTan/Time-series-prediction/discussions>`_
   - 🐛 Report bugs in `GitHub Issues <https://github.com/LongxingTan/Time-series-prediction/issues>`_

**Contributing**
   We welcome contributions! See our `Contributing Guide <https://github.com/LongxingTan/Time-series-prediction/blob/master/CONTRIBUTING.md>`_ for details.


Citation
--------

If you use TFTS in your research, please cite:

.. code-block:: bibtex

   @misc{tfts2020,
     author = {Longxing Tan},
     title = {TFTS: TensorFlow Time Series},
     year = {2020},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/longxingtan/time-series-prediction}},
   }


License
-------

TFTS is released under the MIT License. See `LICENSE <https://github.com/LongxingTan/Time-series-prediction/blob/master/LICENSE>`_ for details.
