# TFTS examples

These examples are small, runnable workflows for the main TFTS time-series
tasks. Each workflow follows the same production-oriented lifecycle:

1. prepare the data and configure a task model;
2. train the model;
3. save a task-aware TFTS artifact;
4. restore the artifact in a new model instance; and
5. run a small inference pass with the restored model.

## Quick reference

| Example | Task | Command |
| --- | --- | --- |
| [Time-series prediction](./run_prediction_simple.py) | Forecast future values | `python examples/run_prediction_simple.py` |
| [Time-series classification](./run_classification.py) | Assign a label to each series | `python examples/run_classification.py` |
| [Anomaly detection](./run_anomaly.py) | Detect unusual patterns | `python examples/run_anomaly.py` |
| [Parameter tuning](./run_tuner.py) | Tune model parameters with Optuna | `python examples/run_tuner.py` |

All examples accept `--output_dir` (the tuner uses it as the parent directory
for per-trial artifacts). The default output locations are under `./outputs/`.
These paths can be changed when integrating the examples into a larger project.

## Save, load, and inference

`trainer.save_model(...)` writes a task-aware model directory containing the
architecture configuration, task configuration, and TensorFlow weights. Load
that directory with `AutoModel.from_pretrained(...)` and use the restored model
for inference:

```python
import tfts
from tfts import AutoConfig, AutoModel, AutoModelForForecasting, KerasTrainer

train_length = 24
predict_sequence_length = 8
(x_train, y_train), (x_valid, y_valid) = tfts.get_data(
    "sine",
    train_length,
    predict_sequence_length,
    test_size=0.2,
)

config = AutoConfig.for_model("rnn")
model = AutoModelForForecasting.from_config(config, prediction_length=8)
trainer = KerasTrainer(model)
trainer.train((x_train, y_train), (x_valid, y_valid), epochs=1)
trainer.save_model("./outputs/forecasting_model")

restored_model = AutoModel.from_pretrained(
    "./outputs/forecasting_model",
    sample_batch=x_valid[:1],
)
predictions = restored_model(x_valid, training=False).numpy()
print(predictions.shape)
```

Passing `sample_batch` makes the input contract explicit and is useful for
models with multiple or dictionary inputs. Keep any data preprocessing—such as
scalers and feature encoders—alongside the model artifact so inference uses the
same transformation as training.

Anomaly detection has one additional fitted stage: threshold calibration. The
loaded detector must be calibrated on normal reference windows before calling
`detect`:

```python
restored_detector = AutoModel.from_pretrained(
    "./outputs/anomaly_model",
    sample_batch=normal_windows[:1],
)
restored_detector.calibrate(normal_windows)
result = restored_detector.detect(windows_to_score)
print(result.labels.numpy())
```

For a raw Keras `.keras` archive, use `model.save("model.keras")` and
`tfts.load_model("model.keras", compile=False)`. The example scripts use the
TFTS task-aware directory format because it restores the task head and is also
usable for further fine-tuning.

## Notebooks

The longer notebooks are maintained in `docs/source/tutorials/` and rendered
on the documentation site:

- [Single-step weather prediction](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/single_step_weather_prediction.ipynb)
- [Multi-step sales prediction](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/multi_steps_sales_prediction.ipynb)
- [DeepAR forecast demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/deepar_ar_demo.ipynb)
- [N-BEATS forecast demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/nbeats_ar_demo.ipynb)
- [PatchTST and Autoformer forecast demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/patchtst_autoformer_ar_demo.ipynb)
- [TFT Stallion demand prediction](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/tft_stallion_prediction.ipynb)
- [TimesNet and TimeMixer M4 demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/timesnet_timemixer_m4_demo.ipynb)
- [Mitsui recursive LSTM demo](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/docs/source/tutorials/mitsui_recursive_lstm_demo.ipynb)

## Benchmarks and advanced projects

- [Forecasting Sticker Sales on Kaggle](https://www.kaggle.com/competitions/playground-series-s5e1)
- [TFTS-BERT](https://github.com/LongxingTan/KDDCup2022-Baidu), which placed third in the KDD Cup 2022 wind-power forecasting competition
- [TFTS-Seq2seq](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction), which placed fourth in the 2021 Tianchi ENSO prediction competition

## Contributing

Contributions are welcome. For an example, notebook, or improvement, please
follow the [contribution guidelines](../CONTRIBUTING.md).
