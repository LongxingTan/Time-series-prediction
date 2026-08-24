# TFTS Benchmark

Reproducible benchmarking for TensorFlow time-series forecasting models.

The benchmark runner provides one YAML-driven workflow for selecting datasets,
configuring model architectures, repeating experiments with deterministic seeds,
computing forecasting metrics, and exporting publication-friendly results.

## Why use it?

- **Configuration-first:** datasets, models, training settings, and evaluation behavior live in versionable YAML.
- **Reproducible runs:** every trial records its seed, configuration, sequence lengths, metrics, and history.
- **Comparable output:** all model-dataset pairs use the same result schema and export to JSON, CSV, and LaTeX.
- **Research-friendly data:** use built-in datasets, local NPZ tensors, structured inputs, or a custom adapter.
- **Experiment support:** ready-to-run configurations cover TFT, N-BEATS, and Autoformer under `exps/`.

> [!IMPORTANT]
> Consistent execution does not automatically make a comparison fair. Align
> preprocessing, forecast horizons, optimization budgets, stopping criteria,
> and metric scales when comparing models.

## Quick start

From the repository root:

```bash
pip install -e .
tfts-benchmark --config benchmark/configs/example.yaml
```

Using the repository virtual environment directly:

```bash
.venv/bin/python -m benchmark.cli \
  --config benchmark/configs/example.yaml
```

Each run writes these files to the configured `output_dir`:

```text
benchmark_results/
├── results.json  # one record per trial
├── results.csv   # mean and standard deviation by model/dataset
└── results.tex   # LaTeX table
```

## YAML configuration

A configuration has three main sections:

```yaml
benchmark:
  metrics: [mae, rmse, smape]
  runs: 3
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  seed: 42
  output_dir: benchmark_results/my_study
  verbose: 1

models:
  rnn:
    config:
      rnn_hidden_size: 64
      rnn_type: gru
    training:
      loss: mae
      early_stopping_patience: 5
  transformer:
    config:
      hidden_size: 128
      num_layers: 2
      num_attention_heads: 4

datasets:
  sine:
    train_length: 24
    predict_sequence_length: 8
    n_examples: 1000
    test_size: 0.2
```

### Benchmark options

| Option | Default | Description |
| --- | ---: | --- |
| `metrics` | `[mae, rmse, mape]` | Metrics computed on validation predictions. |
| `runs` | `1` | Trials per model-dataset pair; trial `i` uses `seed + i`. |
| `epochs` | `50` | Maximum training epochs. |
| `batch_size` | `32` | Training batch size. |
| `learning_rate` | `0.001` | Default optimizer learning rate. |
| `train_length` | dataset default | Global lookback override. |
| `predict_sequence_length` | dataset default | Global forecast-horizon override. |
| `seed` | `42` | Base random seed. |
| `output_dir` | `benchmark_results` | Result directory. |
| `verbose` | `1` | `0` silent, `1` progress, or `2` detailed training output. |

Dataset values override global training values for that dataset. Explicit CLI
options override YAML values.

### Model sections

Each selected model can contain:

- `config`: attributes applied to the TFTS model configuration.
- `training`: trainer options such as `loss`, `optimizer`, `early_stopping_patience`, and `reduce_lr_patience`.
- `prediction`: point-forecast extraction. Quantile models can specify `quantile: 0.5`; dictionary outputs can specify `output_key`.

When no overrides are needed, use the compact form:

```yaml
models: [rnn, transformer, dlinear]
datasets: [sine, air_passengers]
```

The runner evaluates the Cartesian product of the selected models and datasets.

## Reproducing repository experiments

The included configurations use the TensorFlow models and prepared data under `exps/`.

| Experiment | Configuration | Data | Geometry |
| --- | --- | --- | --- |
| Temporal Fusion Transformer | `configs/exps_tft.yaml` | `exps/phase2_tfts/*.npz` | 24 → 6 |
| N-BEATS | `configs/exps_nbeats.yaml` | `exps/nbeats_phase2_tfts/*.npz` | 60 → 20 |
| Autoformer | `configs/exps_autoformer.yaml` | M4 Weekly under `exps/dataset/m4` | 26 → 13 |

```bash
.venv/bin/python -m benchmark.cli --config benchmark/configs/exps_tft.yaml
.venv/bin/python -m benchmark.cli --config benchmark/configs/exps_nbeats.yaml
.venv/bin/python -m benchmark.cli --config benchmark/configs/exps_autoformer.yaml
```

TFT demonstrates structured inputs, multi-quantile loss, and median extraction.
N-BEATS consumes the exact Phase 2 tensors. Autoformer uses the local M4 adapter
and official M4 Weekly test horizons.

## Datasets

List registered datasets without starting a run:

```bash
tfts-benchmark --list-datasets
```

| Name/type | Intended use |
| --- | --- |
| `sine` | Fast synthetic smoke tests. |
| `air_passengers` | Small univariate experiments. |
| `grocery_sales` | Grocery sales forecasting. |
| `recruit_restaurant` | Grouped restaurant demand forecasting. |
| `m4` | Local M4 archives with seasonal-pattern selection. |
| `npz` | Generic arrays and structured TensorFlow inputs. |
| `forecasting_sticker_sales` | Migrated forecasting example. |
| `CMI_detect_sleep_states` | Migrated sleep-state example. |

### Prepared NPZ tensors

Use `type: npz` to give prepared data a study-specific name:

```yaml
datasets:
  my_preprocessed_data:
    type: npz
    train_path: artifacts/train.npz
    valid_path: artifacts/validation.npz
    input_keys: [x]
    target_key: y
    train_length: 48
    predict_sequence_length: 12
```

For multi-input models, list all arrays. They are passed to TensorFlow as a
dictionary keyed by their NPZ names:

```yaml
input_keys: [static_real, encoder_real, decoder_real]
```

### M4 data

```yaml
datasets:
  weekly:
    type: m4
    data_dir: exps/dataset/m4
    seasonal_pattern: Weekly
    train_length: 26
    predict_sequence_length: 13
    windows_per_series: 1
    window_seed: 2026
```

The adapter selects the seasonal pattern, constructs deterministic complete
training windows, and evaluates against the corresponding official test horizons.

### Adding a Python dataset

Implement the dataset contract when loading requires custom code:

```python
from benchmark import Dataset


class MyDataset(Dataset):
    name = "my_dataset"
    description = "A concise description and version of the data."
    train_length = 48
    predict_sequence_length = 12

    def prepare_data(self, **kwargs):
        x, y = load_and_transform(**kwargs)
        return x, y

    def get_train_valid_split(self, **kwargs):
        x, y = self.prepare_data(**kwargs)
        return (x_train, y_train), (x_valid, y_valid)
```

Register it before creating the runner:

```python
from benchmark import BenchmarkConfig, BenchmarkRunner, DatasetRegistry

registry = DatasetRegistry()
registry.register("my_dataset", MyDataset)
config = BenchmarkConfig(models=["rnn"], datasets=["my_dataset"])
results = BenchmarkRunner(config, dataset_registry=registry).run()
```

## Models

The benchmark uses the TFTS `AutoConfig` and `AutoModel` registries, so supported
TFTS forecasting models are automatically available:

```bash
tfts-benchmark --list-models
```

Architecture keys are model-specific. Consult the corresponding config class in
`tfts/models/` and record every non-default value in YAML.

## Metrics

| Metric | Description | Research note |
| --- | --- | --- |
| `mae` | Mean absolute error | Scale-dependent and robust to large errors. |
| `mse` | Mean squared error | Emphasizes large errors. |
| `rmse` | Root mean squared error | Expressed in target units. |
| `mape` | Mean absolute percentage error | Unstable around zero targets. |
| `mape_pct` | MAPE expressed as a percentage | Same zero-target caveat. |
| `smape` | Symmetric MAPE | Common for M4-style evaluation. |
| `r2` | Coefficient of determination | Interpret carefully for non-stationary series. |

Metrics use the model's validation point forecast. For probabilistic or quantile
models, document the extraction rule in the model's `prediction` section.

## Python API

```python
from benchmark import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig.from_yaml("benchmark/configs/example.yaml")
results = BenchmarkRunner(config).run()

results.print_table()
frame = results.to_dataframe()
results.to_csv("results.csv")
results.to_json("results.json")
results.to_latex("results.tex")
```

Core components:

- `BenchmarkConfig` loads and validates configuration.
- `BenchmarkRunner` executes trials and saves results.
- `DatasetRegistry` resolves built-in and user-defined datasets.
- `ModelRegistry` exposes TFTS forecasting models.
- `BenchmarkMetrics` computes point-forecast metrics.
- `BenchmarkResults` aggregates and exports trial records.

## Command-line overrides

Use CLI options for quick sweeps or temporary changes:

```bash
tfts-benchmark \
  --config benchmark/configs/example.yaml \
  --models rnn transformer \
  --runs 5 \
  --epochs 100 \
  --output-dir benchmark_results/ablation
```

Run `tfts-benchmark --help` for all options.

## Reproducible reporting checklist

For publication or model comparison, preserve:

1. The YAML configuration and code revision.
2. Dataset version, source, split policy, and preprocessing artifacts.
3. Target transformation and inverse transformation.
4. Lookback, forecast horizon, and evaluation scale.
5. Number of independent runs and all reported seeds.
6. Optimizer, loss, stopping rule, and maximum training budget.
7. Hardware and TensorFlow/CUDA versions when performance matters.
8. Raw JSON results, aggregated CSV results, and failed-run logs.

Report mean and standard deviation across independent runs. Avoid selecting a
model on the same test split used for its final reported score.

## Scope and limitations

- Selections form a Cartesian product. Use separate YAML files for incompatible input schemas.
- Metrics currently evaluate point forecasts. Distributional scores and full DeepAR ancestral sampling need a specialized adapter.
- The M4 adapter uses complete sampled training windows. This avoids padded-target bias but may differ from published sampling implementations.
- Dataset download, licensing, and citation requirements remain the researcher's responsibility.

## Contributing

Reusable datasets, metrics, adapters, and reproducible configurations are
welcome. Please include a focused unit test, a small smoke-test path, documented
data provenance and tensor shapes, and explicit split/preprocessing semantics.
Do not commit large generated results, checkpoints, or restricted datasets.

Before submitting changes:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
make style
```

See the repository-level `CONTRIBUTING.md` for the contribution policy and
development workflow.

## License and citation

TFTS is released under the repository's MIT license. Individual datasets may
have separate terms; review and cite the original model and dataset publications.

If this benchmark supports published work, cite the TFTS repository together
with the original methods and datasets evaluated.
