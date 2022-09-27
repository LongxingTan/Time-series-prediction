<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="500" align=center/>
</h1><br>

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

**[文档](https://time-series-prediction.readthedocs.io)** | **[教程](https://time-series-prediction.readthedocs.io/en/latest/tutorials.html)** | **[发布日志](https://time-series-prediction.readthedocs.io/en/latest/CHANGELOG.html)** | **[English](https://github.com/LongxingTan/Time-series-prediction/blob/master/README.md)**


**东流TFTS** (TensorFlow Time Series) 是一个时间序列的开源工具，采用TensorFlow框架，已支持多种深度学习SOTA模型。中文名“东流”，源自辛弃疾“青山遮不住，毕竟**东流**去。江晚正愁余，山深闻鹧鸪”。
- 结构灵活，适配多种时间序列任务
- [多套久经考验的深度学习模型](./examples)
- [快速入门](https://time-series-prediction.readthedocs.io)


## 安装
``` bash
pip install tensorflow>=2.0.0
pip install tfts
```


## 快速使用
``` python
import tensorflow as tf
import tfts

data = tfts.load('passenger')
x_train, y_train
model = AutoModel('seq2seq')
# train the model
model.train(data)
# predict new data
model.predict(data)
```


## 示例
- [东流Bert模型](https://github.com/LongxingTan/KDDCup2022-Baidu) 获得KDD CUP2022百度风机功率预测第3名
- [东流Seq2seq模型](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction) 获得阿里天池-AI earth人工智能气象挑战赛第4名

更多模型
- [RNN](./examples/run_rnn.py)
- [DeepAR](./examples/run_deepar.py)
- [ESRNN](./examples/run_esrnn.py)
- [Seq2seq](./examples/run_seq2seq.py)
- [TCN](./examples/run_tcn.py)
- [Wavenet](./examples/run_wavenet.py)
- [Bert](./examples/run_bert.py)
- [Transformer](./examples/run_transformer.py)
- [Temporal-fusion-transformer](./examples/run_temporal_fusion_transformer.py)
- [Informer](./examples/run_informer.py)
- [AutoFormer](./examples/run_autoformer.py)
- [U-Net](./examples/run_unet.py)
- [Nbeats](./examples/run_nbeats.py)
- [GAN](./examples/run_gan.py)

更多应用
- [Time series classification](./examples/run_classification.py)
- [Anomaly detection](./examples/run_anomaly.py)
- [Uncertainty prediction](./examples/run_uncertrainty.py)
- [Parameters tuning with optuna](./examples/run_optuna.py)


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
