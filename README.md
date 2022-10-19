[license-image]: https://img.shields.io/badge/License-MIT-blue.svg
[license-url]: https://opensource.org/licenses/MIT
[pypi-image]: https://badge.fury.io/py/tfts.svg
[pypi-url]: https://pypi.python.org/pypi/tfts
[build-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml?query=branch%3Amaster
[lint-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/lint.yml/badge.svg?branch=master
[lint-url]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/lint.yml?query=branch%3Amaster
[docs-image]: https://readthedocs.org/projects/time-series-prediction/badge/?version=latest
[docs-url]: https://time-series-prediction.readthedocs.io/en/latest/
[coverage-image]: https://codecov.io/gh/longxingtan/Time-series-prediction/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/longxingtan/Time-series-prediction?branch=master
[codeql-image]: https://github.com/longxingtan/Time-series-prediction/actions/workflows/codeql-analysis.yml/badge.svg
[codeql-url]: https://github.com/longxingtan/Time-series-prediction/actions/workflows/codeql-analysis.yml

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="490" align=center/>
</h1><br>

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Lint Status][lint-image]][lint-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![CodeQL Status][codeql-image]][codeql-url]

**[Documentation](https://time-series-prediction.readthedocs.io)** | **[Tutorials](https://time-series-prediction.readthedocs.io/en/latest/tutorials.html)** | **[Release Notes](https://time-series-prediction.readthedocs.io/en/latest/CHANGELOG.html)** | **[中文](https://github.com/LongxingTan/Time-series-prediction/blob/master/README_CN.md)**

**TFTS** (TensorFlow Time Series) is a python package for time series task, supporting the classical and SOTA deep learning methods in [TensorFlow](https://www.tensorflow.org/).
- Flexible and powerful design for time series task
- Advanced deep learning models
- Documentation lives at [time-series-prediction.readthedocs.io](https://time-series-prediction.readthedocs.io)

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC)

**Installation**

- python >= 3.7
- tensorflow >= 2.1

``` bash
$ pip install tfts
```

**Usage**

``` python
import matplotlib.pyplot as plt
import tfts
from tfts import AutoModel, KerasTrainer, Trainer

train_length = 24
predict_length = 8

# train is a tuple of (x_train, y_train), valid is (x_valid, y_valid)
train, valid = tfts.get_data('sine', train_length, predict_length, test_size=0.2)
model = AutoModel('seq2seq', predict_length)

trainer = KerasTrainer(model)
trainer.train(train, valid)

pred = trainer.predict(valid[0])
trainer.plot(history=valid[0], true=valid[1], pred=pred)
```

## Examples

- [TFTS-Bert model](https://github.com/LongxingTan/KDDCup2022-Baidu) wins the **3rd place** in KDD Cup 2022 Baidu-wind power forecasting
- [TFTS-Seq2seq model](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction) wins the **4th place** in Alibaba Tianchi-ENSO prediction 2021

### Performance

[Time series prediction](./examples/run_prediction.py) performance is evaluated by tfts implementation, not official

| Performance | [web traffic<sup>mape</sup>]() | [grocery sales<sup>rmse</sup>](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) | [m5 sales<sup>val</sup>]() | [ventilator<sup>val</sup>]() |
| :-- | :-: | :-: | :-: | :-: |
| [RNN]() | 672 | 47.7% |52.6% | 61.4% |
| [DeepAR]() | 672 | 47.7% |52.6% | 61.4% |
| [Seq2seq]() | 672 | 47.7% |52.6% | 61.4% |
| [TCN]() | 672 | 47.7% |52.6% | 61.4% |
| [WaveNet]() | 672 | 47.7% |52.6% | 61.4% |
| [Bert]() | 672 | 47.7% |52.6% | 61.4% |
| [Transformer]() | 672 | 47.7% |52.6% | 61.4% |
| [Temporal-fusion-transformer]() | 672 | 47.7% |52.6% | 61.4% |
| [Informer]() | 672 | 47.7% |52.6% | 61.4% |
| [AutoFormer]() | 672 | 47.7% |52.6% | 61.4% |
| [N-beats]() | 672 | 47.7% |52.6% | 61.4% |
| [U-Net]() | 672 | 47.7% |52.6% | 61.4% |

### More demos

- [Time series classification](./examples/run_classification.py)
- [Anomaly detection](./examples/run_anomaly.py)
- [Uncertainty prediction](examples/run_uncertainty.py)
- [Parameters tuning by optuna](examples/run_optuna_tune.py)
- [Serving by tf-serving](./examples)

if you prefer to use [PyTorch](https://pytorch.org/), you can use [pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting)

## Citation

If you find tfts project useful in your research, please consider cite:

```
@misc{tfts2020,
  author = {Longxing Tan},
  title = {Time series prediction},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/longxingtan/time-series-prediction}},
}
```
