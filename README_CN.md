[license-image]: https://img.shields.io/badge/License-MIT-blue.svg
[license-url]: https://opensource.org/licenses/MIT
[pypi-image]: https://badge.fury.io/py/tfts.svg
[pypi-url]: https://pypi.python.org/pypi/tfts
[build-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml?query=branch%3Amaster
[lint-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/lint.yml/badge.svg
[lint-url]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/lint.yml
[docs-image]: https://readthedocs.org/projects/time-series-prediction/badge/?version=latest
[docs-url]: https://time-series-prediction.readthedocs.io/en/latest/
[coverage-image]: https://codecov.io/gh/longxingtan/Time-series-prediction/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/longxingtan/Time-series-prediction?branch=master

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="500" align=center/>
</h1><br>

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Lint Status][lint-image]][lint-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]

**[文档](https://time-series-prediction.readthedocs.io)** | **[教程](https://time-series-prediction.readthedocs.io/en/latest/tutorials.html)** | **[发布日志](https://time-series-prediction.readthedocs.io/en/latest/CHANGELOG.html)** | **[English](https://github.com/LongxingTan/Time-series-prediction/blob/master/README.md)**

**东流TFTS** (TensorFlow Time Series) 是一个高效易用的时间序列开源工具，基于TensorFlow，支持多种深度学习模型

- 结构灵活，适配多种时间序列任务
- [多套久经考验的深度学习模型](./examples)
- [查阅文档，快速入门](https://time-series-prediction.readthedocs.io)

中文名“**东流**”，源自辛弃疾“青山遮不住，毕竟**东流**去。江晚正愁余，山深闻鹧鸪”。

## 安装

环境要求

- python >= 3.7
- tensorflow >= 2.1

``` bash
$ pip install tfts
```

## 快速使用

``` python
import tfts
from tfts import AutoModel, KerasTrainer

train_length = 24
predict_length = 8

train, valid = tfts.get_data('sine', train_length, predict_length)
model = AutoModel('seq2seq', predict_length)

trainer = KerasTrainer(model)
trainer.train(train, valid)
trainer.predict(valid[0])
```

## 示例

东流tfts专注业界领先的深度模型

- [Bert模型](https://github.com/LongxingTan/KDDCup2022-Baidu) 获得KDD CUP2022百度风机功率预测第3名
- [Seq2seq模型](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction) 获得阿里天池-AI earth人工智能气象挑战赛第4名
- [RNN]()
- [DeepAR]()
- [TCN]()
- [WaveNet]()
- [Transformer]()
- [Informer]()
- [AutoFormer]()

### 更多应用

- [时序预测](./examples/run_prediction.py)
- [时序分类](./examples/run_classification.py)
- [异常检测](./examples/run_anomaly.py)
- [Uncertainty prediction](examples/run_uncertainty.py)
- [optuna调参](./examples/run_optuna_tune.py)
- [多gpu与tpu加速训练](./examples)
- [tf-serving部署](./examples)

## 引用

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
