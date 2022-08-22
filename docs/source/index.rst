.. Time-series-prediction documentation master file, created by
   sphinx-quickstart on Tue Mar  8 13:01:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tfts's documentation!
==================================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

TFTS(TensorFlow Time Series) aims to ease state-of-the-art
timeseries forecasting with deep learning for both business cases and
data competitions. It's similar to the `pytorch-forecasting <https://github.com/jdb78/pytorch-forecasting>`_ and `gluon-ts <https://github.com/awslabs/gluon-ts/>`_, but with `TensorFlow <https://www.tensorflow.org/>`_. 
Specifically, the package provides

* A timeseries dataset class which abstracts handling variable transformations, missing values,
  randomized subsampling, multiple history lengths, etc.
* A base model class which provides basic training of timeseries models along with logging in tensorboard
  and generic visualizations such actual vs predictions and dependency plots
* Multiple neural network architectures for timeseries forecasting that have been enhanced
  for real-world deployment and come with in-built interpretation capabilities
* Multi-horizon timeseries metrics
* Ranger optimizer for faster model training


If you do not have tensorflow already installed, follow the :ref:`detailed installation instructions<install>`.

Otherwise, proceed to install the package by executing

.. code-block::

   pip install tfts

or to install via conda

.. code-block::

   conda install tfts -c conda-forge

Vist :ref:`Getting started <getting-started>` to learn more about the package and detailled installation instruction.
The :ref:`Tutorials <tutorials>` section provides guidance on how to use models and implement new ones.


.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 6

   getting-started
   tutorials
   data
   models
   metrics
   faq
   contribute
   api
   competition
   CHANGELOG
   GitHub <https://github.com/LongxingTan/Time-series-prediction>

