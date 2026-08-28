# 🚀 TFTS Examples

## 🛠️ Basic Usage
Get started with these basic examples:
- [Time Series Prediction](./run_prediction_simple.py): Predict future values in a time series.
- [Time Series Classification](./run_classification.py): Classify time series data into different categories.
- [Time Series Anomaly Detection](./run_anomaly.py): Detect unusual patterns or anomalies in time series data.
- [AutoML for parameters tuning](./run_tuner.py): Automatically tune model parameters using Optuna.


## 📓 Notebooks
Dive deeper with these notebooks (kept in `docs/source/tutorials/`, rendered on the docs site):

- [single step prediction](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/single_step_weather_prediction.ipynb): A guided example on predicting the next time point in a weather dataset.
- [multi steps prediction](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/multi_steps_sales_prediction.ipynb): Forecast multiple future time points in a sales dataset.
- [DeepAR forecast demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/deepar_ar_demo.ipynb): Probabilistic DeepAR on synthetic AR data, trained with scheduled sampling.
- [N-BEATS forecast demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/nbeats_ar_demo.ipynb): N-BEATS on synthetic AR data.
- [PatchTST & Autoformer forecast demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/patchtst_autoformer_ar_demo.ipynb): Compact PatchTST and Autoformer on identical synthetic AR windows.
- [TFT Stallion demand prediction](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/tft_stallion_prediction.ipynb): Temporal Fusion Transformer on the Stallion demand dataset.
- [TimesNet & TimeMixer M4 demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/timesnet_timemixer_m4_demo.ipynb): TimesNet and TimeMixer on the M4 Weekly forecasting task via the `tfts` pipeline.
- [Mitsui recursive LSTM demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/mitsui_recursive_lstm_demo.ipynb): Recursive full-feature forecasting with TFTS.

## 💾 Save, load, and fine-tune

Use the TFTS pretrained format when the restored model will be trained further:

```python
model.save_pretrained("./my_model")
restored = AutoModel.from_pretrained("./my_model")
```

Concrete model classes support the same API, for example
`TCN.from_pretrained("./my_model")`. To load a raw Keras archive for inference,
use `tfts.load_model("./my_model.keras", compile=False)`; TFTS custom layers are
discovered automatically.

## 📊 Benchmark
- [Kaggle - Forecasting Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1)


## 🏆 More examples
Check out these advanced examples and competition-winning implementations:

**Multiple steps prediction**
- [TFTS-Bert](https://github.com/LongxingTan/KDDCup2022-Baidu) wins the **3rd place** in KDD Cup 2022 wind power forecasting
- [TFTS-Seq2seq](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction) wins the **4th place** in Tianchi ENSO prediction 2021


## 🤝 Contributing
We welcome contributions! If you have an example, notebook, or improvement to share, please follow [these steps](../CONTRIBUTING.md)
