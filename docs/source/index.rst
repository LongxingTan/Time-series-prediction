.. Time-series-prediction documentation master file, created by
   sphinx-quickstart on Tue Mar  8 13:01:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TFTS Documentation
==================================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

TFTS (TensorFlow Time Series) supports state-of-the-art deep learning
time series models for both business cases and data competitions. The package provides

* Flexible and powerful design for time series task
* Advanced SOTA deep learning models
* TFTS documentation lives at `time-series-prediction.readthedocs.io <https://time-series-prediction.readthedocs.io>`_

Quick Start
-------------
Visit :ref:`Quick start <quick-start>` to learn more about the package and :ref:`detailed installation instructions<install>`.

.. code-block:: shell

   $ pip install tensorflow
   $ pip install tfts


Tutorials
----------
The :ref:`Tutorials <tutorials>` section provides guidance on how to use models and implement new ones.


Models
---------
The tfts library supports the SOTA deep learning models for time series.

- `The TFTS BERT model <https://github.com/LongxingTan/KDDCup2022-Baidu>`_ wins the 3rd place in `Baidu KDD Cup 2022 <https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction>`_
- `The TFTS Seq2Seq mode <https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction>`_ wins the 4th place of `Tianchi ENSO prediction <https://tianchi.aliyun.com/competition/entrance/531871/introduction>`_
- :ref:`Learn more models <models>`


Tricks
------
Visit :ref:`Tricks <tricks>` if you want to know more tricks to improve the prediction performance.



.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 6

   quick-start
   tutorials
   models
   tricks
   api
   CHANGELOG
