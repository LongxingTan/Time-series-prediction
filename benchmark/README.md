# TFTS Benchmark System

A flexible benchmarking framework for evaluating TFTS models across multiple datasets with multiple metrics and multiple runs. Designed for reproducibility and paper-quality results.

## Usage

### Command-Line

```bash
python -m benchmark.cli \
    --models rnn transformer dlinear \
    --datasets sine air_passengers \
    --metrics mae rmse mape \
    --runs 3 \
    --epochs 50 \
    --output-dir results/
```

### Python API

```python
from benchmark import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(
    models=["rnn", "transformer", "dlinear"],
    datasets=["sine", "air_passengers"],
    metrics=["mae", "rmse", "mape"],
    runs=3,
    epochs=50,
    output_dir="benchmark_results",
)

runner = BenchmarkRunner(config)
results = runner.run()

results.print_table()          # console
results.to_csv("results.csv")   # CSV
results.to_latex("results.tex") # LaTeX for papers
```

## Adding a New Dataset

```python
from benchmark import Dataset
import pandas as pd

class MyDataset(Dataset):
    name = "my_dataset"
    description = "Description of my dataset"
    train_length = 24
    predict_sequence_length = 8

    def prepare_data(self, **kwargs):
        # Load your data from any source (CSV, DB, API, etc.)
        x, y = ...
        return x, y

    def get_train_valid_split(self, **kwargs):
        x, y = self.prepare_data(**kwargs)
        # split into train/valid
        return (x_train, y_train), (x_valid, y_valid)
```

Then register it:

```python
from benchmark import BenchmarkRunner, DatasetRegistry

registry = DatasetRegistry()
registry.register("my_dataset", MyDataset)

config = BenchmarkConfig(datasets=["my_dataset"], ...)
runner = BenchmarkRunner(config, dataset_registry=registry)
results = runner.run()
```

## Architecture

- **BenchmarkRunner**: Orchestrates running models on datasets, collecting results.
- **Dataset**: Abstract base; each dataset subclass implements `prepare_data()` and returns standardized format.
- **DatasetRegistry**: Maintains a registry of all available datasets.
- **ModelRegistry**: Wraps existing tfts model mapping.
- **BenchmarkMetrics**: Computes standard time-series metrics (MAE, MSE, RMSE, MAPE, etc.)
- **BenchmarkResults**: Formats and exports results (CSV, JSON, LaTeX, console table).

## Metrics

Available metrics:
- `mae`: Mean Absolute Error
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `mape`: Mean Absolute Percentage Error
- `smape`: Symmetric MAPE
- `r2`: R-squared

## Migrated Example Benchmarks

The previous `examples/benchmarks` tasks are available as registered datasets:

- `forecasting_sticker_sales`
- `CMI_detect_sleep_states`

Both support `data_path` overrides through `per_dataset_config`. If the source
CSV is not available, they generate deterministic placeholder data so the
benchmark runner and CLI remain usable without Kaggle downloads.

## Output Files

After running, the following files are generated in `output_dir`:
- `results.json`: Raw results for each run.
- `results.csv`: Averaged results (mean/std per model-dataset).
- `results.tex`: LaTeX table for papers.
