# AGENTS.md - Guidelines for Agentic Coding

## Project Overview

TFTS (Time series forecasting with TensorFlow) is a deep learning library for time series prediction. The project uses TensorFlow/Keras for model building and provides auto-configuration for various time series models (RNN, Transformer, Informer, N-BEATS, etc.).

## Development Commands

### Setup
```shell
# Activate virtual environment
source ./.venv/bin/activate

# Install dependencies
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```shell
# Run all unit tests
make test

# Run a single test file
python -m unittest tests.test_trainer

# Run a specific test class
python -m unittest tests.test_trainer.SetSeedTest

# Run a specific test method
python -m unittest tests.test_trainer.SetSeedTest.test_set_seed_reproducibility
```

### Code Quality

```shell
# Run all formatters and linters (black, isort, flake8, pre-commit)
make style

# Run individual tools
black tfts examples tests
isort tfts examples tests
flake8 tfts examples tests
pre-commit run --all-files

# Type checking
mypy tfts
```

### Documentation

```shell
# Build documentation
make docs
```

---

## Code Style Guidelines

### General Principles

- **Line length**: Maximum 120 characters (configured in `pyproject.toml`)
- **Python version**: 3.8 - 3.13
- **Docstrings**: Use Google/NumPy style with `Parameters`, `Returns`, and `Raises` sections

### Imports

**Standard library first, then third-party, then local:**

```python
# Standard library
import collections
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

# Third-party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input

# Local imports (relative)
from .constants import CONFIG_NAME
from ..layers.util_layer import CreateDecoderFeature
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `BaseModel`, `FeatureRegistry`)
- **Functions/variables**: `snake_case` (e.g., `predict_sequence_length`, `build_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `TF2_WEIGHTS_NAME`)
- **Private methods**: Leading underscore (e.g., `_prepare_3d_inputs`)

### Type Annotations

- Use `typing` module: `Dict`, `List`, `Optional`, `Union`, `Tuple`, `Callable`, `Any`
- Use lowercase for type hints in Python 3.9+ or `typing` module for earlier versions
- Example: `def __init__(self, predict_sequence_length: int = 1, config: Optional["BaseConfig"] = None)`

### Error Handling

- Use specific exceptions: `FileNotFoundError`, `TypeError`, `ValueError`
- Include informative error messages with f-strings:
  ```python
  raise FileNotFoundError(f"Weights file not found at {weights_dir}")
  raise TypeError("cols must be a string or list of strings")
  ```
- Validate inputs early with explicit type checking:
  ```python
  if not isinstance(cols, (str, list)):
      raise TypeError("cols must be a string or list of strings")
  ```

### Docstrings

Follow this template for classes and methods:

```python
class FeatureRegistry:
    """A registry for managing time series features.

    This class provides functionality to register, track, and persist features used in
    time series prediction models.

    Attributes:
        columns (List[str]): List of registered feature column names.
    """

    def __init__(self) -> None:
        """Initialize an empty feature registry."""
        ...

    def register(self, cols: Union[str, List[str]]) -> None:
        """Register one or more feature columns.

        Args:
            cols: A single feature column name or a list of feature column names to register.

        Raises:
            TypeError: If cols is not a string or list of strings.
            ValueError: If any column name is empty or contains invalid characters.
        """
        ...
```

### Code Organization

- **Module structure**: Use `__init__.py` to expose public API with `__all__`
- **Base classes**: Use `abc` module (`ABC`, `abstractmethod`) for abstract interfaces
- **Logging**: Use `logging.getLogger(__name__)` for module-level loggers

### Testing Conventions

- Use Python's built-in `unittest` framework
- Organize tests in `tests/` directory mirroring source structure
- Test class naming: `<ClassName>Test`
- Test method naming: `test_<method_name>_<behavior>`
- Use `setUp` for test fixtures

Example from `tests/test_trainer.py`:
```python
class SetSeedTest(unittest.TestCase):
    """Test the set_seed utility function."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        ...
```

### Pre-commit Hooks

The project uses these pre-commit hooks (see `.pre-commit-config.yaml`):
- `trailing-whitespace`: Remove trailing whitespace
- `end-of-file-fixer`: Ensure files end with newline
- `check-yaml`: Validate YAML files
- `check-ast`: Check Python syntax
- `flake8`: Lint Python code
- `isort`: Sort imports
- `black`: Format code
- `nbqa-*`: Validate Jupyter notebooks

### TensorFlow/Keras Specific

- Use `tf.keras` API for model building
- Follow Keras conventions for model classes (inherit from `tf.keras.Model` when appropriate)
- Use symbolic tensors via `tf.keras.Input`
- Handle both functional and subclassing API patterns

---

## File Structure Reference

```
tfts/
├── __init__.py           # Package exports with __all__
├── trainer.py            # Training loop implementation
├── training_args.py      # Training configuration
├── generator.py          # Data generation
├── constants.py          # Constants and paths
├── data/                 # Data loading and preprocessing
├── features/             # Feature engineering
├── layers/               # Reusable neural network layers
├── models/               # Model implementations
└── tasks/                # High-level task pipelines
```
