.. Time-series-prediction documentation master file, created by
   sphinx-quickstart on Tue Mar  8 13:01:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TFTS Documentation
==================================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/Time-series-prediction" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/Time-series-prediction on GitHub">GitHub</a>

TFTS (TensorFlow Time Series) supports state-of-the-art deep learning time series models for both business cases and data competitions. The package provides:

* Flexible and powerful design for time series task
* Advanced SOTA deep learning models
* TFTS documentation lives at `time-series-prediction.readthedocs.io <https://time-series-prediction.readthedocs.io>`_


Quick Start
-----------------
The tfts could accept any time series data of 3D data format as model input: ``(num_examples, train_sequence_length, num_features)``,
and the model supported by tfts outputs 3D data as model output: ``(num_examples, predict_sequence_length, num_outputs)``


Visit :ref:`Quick start <quick-start>` to learn more about the package.

- :ref:`detailed installation instructions<installation>`
- :ref:`how to use it<usage>`


Tutorials
----------
The :ref:`Tutorials <tutorials>` section provides guidance on

- how to :ref:`prepare datasets<prepare_data>` for single-value, multi-value, single-step, and multi-steps prediction
- how to :ref:`use models<train_models>` and implement new ones.


Models
---------
The tfts library supports the SOTA deep learning models for time series.

- `TFTS BERT model <https://github.com/LongxingTan/KDDCup2022-Baidu>`_ wins the 3rd place in `Baidu KDD Cup 2022 <https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction>`_
- `TFTS Seq2Seq model <https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction>`_ wins the 4th place in `Alibaba Tianchi ENSO prediction <https://tianchi.aliyun.com/competition/entrance/531871/introduction>`_
- :ref:`Learn more models <models>`


Tricks
-------
Visit :ref:`Tricks <tricks>` if you want to know more tricks to improve the prediction performance.


Citation
------------
If you find tfts project useful in your research, please consider cite:

.. citetool::

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
