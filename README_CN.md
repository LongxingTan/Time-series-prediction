[license-image]: https://img.shields.io/badge/License-MIT-blue.svg
[license-url]: https://opensource.org/licenses/MIT
[pypi-image]: https://badge.fury.io/py/tfts.svg
[pypi-url]: https://pypi.python.org/pypi/tfts
[pepy-image]: https://pepy.tech/badge/tfts/month
[pepy-url]: https://pepy.tech/project/tfts
[build-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/test.yml?query=branch%3Amaster
[lint-image]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/lint.yml/badge.svg
[lint-url]: https://github.com/LongxingTan/Time-series-prediction/actions/workflows/lint.yml
[docs-image]: https://readthedocs.org/projects/time-series-prediction/badge/?version=latest
[docs-url]: https://time-series-prediction.readthedocs.io/en/latest/?version=latest
[coverage-image]: https://codecov.io/gh/longxingtan/Time-series-prediction/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/longxingtan/Time-series-prediction?branch=master
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/longxingtan/Time-series-prediction/blob/master/CONTRIBUTING.md
[codeql-image]: https://github.com/longxingtan/Time-series-prediction/actions/workflows/codeql-analysis.yml/badge.svg
[codeql-url]: https://github.com/longxingtan/Time-series-prediction/actions/workflows/codeql-analysis.yml

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="400" align=center/>
</h1><br>

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Lint Status][lint-image]][lint-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Contributing][contributing-image]][contributing-url]

**[文档](https://time-series-prediction.readthedocs.io)** | **[教程](https://time-series-prediction.readthedocs.io/en/latest/tutorials.html)** | **[发布日志](https://time-series-prediction.readthedocs.io/en/latest/CHANGELOG.html)** | **[English](https://github.com/LongxingTan/Time-series-prediction/blob/master/README.md)**

青山遮不住，毕竟东流去。江晚正愁余，山深闻鹧鸪。<br>
**东流TFTS** (TensorFlow Time Series) 是一个高效易用的时间序列框架，基于TensorFlow/ Keras。

- 为多种时间序列任务提供SOTA的深度学习模型，预测、分类、异常检测
- 提供经典与前沿的深度学习模型，用于工业、科研、竞赛
- 查阅[英文文档](https://time-series-prediction.readthedocs.io)，快速入门。欢迎移步[时序讨论区](https://github.com/LongxingTan/Time-series-prediction/discussions)


## 快速使用

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LHdbrXmQGBSQuNTsbbM5-lAk5WENWF-Q?usp=sharing)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/tanlongxing/tensorflow-time-series-starter-tfts/notebook)

**安装**

- python >= 3.7
- tensorflow >= 2.4

```shell
pip install tfts
```

**入门使用**

```python
import matplotlib.pyplot as plt
import tfts
from tfts import AutoModel, KerasTrainer, Trainer, AutoConfig

train_length = 24
predict_length = 8

# 其中，train是包含(x_train, y_train)的tuple, valid包含(x_valid, y_valid)
train, valid = tfts.get_data('sine', train_length, predict_length, test_size=0.2)
config = AutoConfig.for_model("seq2seq")
model = AutoModel.from_config(config, predict_length)

trainer = KerasTrainer(model)
trainer.train(train, valid)

pred = trainer.predict(valid[0])
trainer.plot(history=valid[0], true=valid[1], pred=pred)
```

**训练自己的数据**

为方便使用，将数据转化为三维作为tfts的输入
- 选项1 `np.ndarray`
- 选项2 `tf.data.Dataset`

编码类模型输入

```python
import numpy as np
from tfts import AutoConfig, AutoModel, KerasTrainer

train_length = 49
predict_length = 10
n_feature = 2

x_train = np.random.rand(1, train_length, n_feature)
y_train = np.random.rand(1, predict_length, 1)
x_valid = np.random.rand(1, train_length, n_feature)
y_valid = np.random.rand(1, predict_length, 1)

config = AutoConfig.for_model("rnn")
model = AutoModel.from_config(config, predict_length=predict_length)
trainer = KerasTrainer(model)
trainer.train(train_dataset=(x_train, y_train), valid_dataset=(x_valid, y_valid), epochs=1)
```

编码-解码类模型输入

```python
# option1
import numpy as np
from tfts import AutoConfig, AutoModel, KerasTrainer

train_length = 49
predict_length = 10
n_encoder_feature = 2
n_decoder_feature = 3

x_train = (
    np.random.rand(1, train_length, 1),
    np.random.rand(1, train_length, n_encoder_feature),
    np.random.rand(1, predict_length, n_decoder_feature),
)
y_train = np.random.rand(1, predict_length, 1)
x_valid = (
    np.random.rand(1, train_length, 1),
    np.random.rand(1, train_length, n_encoder_feature),
    np.random.rand(1, predict_length, n_decoder_feature),
)
y_valid = np.random.rand(1, predict_length, 1)

config = AutoConfig.for_model("seq2seq")
model = AutoModel.from_config(config, predict_length=predict_length)
trainer = KerasTrainer(model)
trainer.train((x_train, y_train), (x_valid, y_valid), epochs=1)
```

```python
# option2
import tensorflow as tf
from tfts import AutoConfig, AutoModel, KerasTrainer

class FakeReader(object):
    def __init__(self, predict_length):
        train_length = 49
        n_encoder_feature = 2
        n_decoder_feature = 3
        self.x = np.random.rand(15, train_length, 1)
        self.encoder_feature = np.random.rand(15, train_length, n_encoder_feature)
        self.decoder_feature = np.random.rand(15, predict_length, n_decoder_feature)
        self.target = np.random.rand(15, predict_length, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "encoder_feature": self.encoder_feature[idx],
            "decoder_feature": self.decoder_feature[idx],
        }, self.target[idx]

    def iter(self):
        for i in range(len(self.x)):
            yield self[i]

predict_length = 10
train_reader = FakeReader(predict_length=predict_length)
train_loader = tf.data.Dataset.from_generator(
    train_reader.iter,
    ({"x": tf.float32, "encoder_feature": tf.float32, "decoder_feature": tf.float32}, tf.float32),
)
train_loader = train_loader.batch(batch_size=1)
valid_reader = FakeReader(predict_length=predict_length)
valid_loader = tf.data.Dataset.from_generator(
    valid_reader.iter,
    ({"x": tf.float32, "encoder_feature": tf.float32, "decoder_feature": tf.float32}, tf.float32),
)
valid_loader = valid_loader.batch(batch_size=1)

config = AutoConfig.for_model("seq2seq")
model = AutoModel.from_config(config, predict_length=predict_length)
trainer = KerasTrainer(model)
trainer.train(train_dataset=train_loader, valid_dataset=valid_loader, epochs=1)
```

**修改模型配置参数**

```python
import tensorflow as tf
import tfts
from tfts import AutoModel, AutoConfig

config = AutoConfig.for_model('rnn')
print(config)
config.rnn_hidden_size = 128

model = AutoModel.from_config(config, predict_length=7, )
```

**搭建自己的模型**

<details><summary> 检查tfts AutoModel已支持的模型 </summary>

- rnn
- tcn
- bert
- nbeats
- seq2seq
- wavenet
- transformer
- informer

</details>

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tfts import AutoModel, AutoConfig

def build_model():
    train_length = 24
    train_features = 15
    predict_length = 16

    inputs = Input([train_length, train_features])
    config = AutoConfig.for_model("seq2seq")
    backbone = AutoModel.from_config(config, predict_length=predict_length)
    outputs = backbone(inputs)
    outputs = Dense(1, activation="sigmoid")(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse", optimizer="rmsprop")
    return model
```


## 示例

东流tfts专注业界领先的深度模型

- [Bert模型](https://github.com/LongxingTan/KDDCup2022-Baidu) 获得KDD CUP2022-百度风机功率预测第3名
- [Seq2seq模型](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction) 获得阿里天池-AI earth人工智能气象挑战赛第4名

<!-- - [RNN]()
- [DeepAR]()
- [TCN]()
- [WaveNet]()
- [Transformer]()
- [Informer]()
- [AutoFormer]()

### 更多应用

- [时序预测](./examples/run_prediction_simple.py)
- [时序分类](./examples/run_classification.py)
- [异常检测](./examples/run_anomaly.py)
- [Uncertainty prediction](examples/run_uncertainty.py)
- [optuna调参](./examples/run_optuna_tune.py)
- [多gpu与tpu加速训练](./examples)
- [tf-serving部署](./examples) -->


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
