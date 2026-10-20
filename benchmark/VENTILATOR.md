# Ventilator pressure prediction

The first competition dataset is [Google Brain ventilator pressure
prediction](https://www.kaggle.com/c/ventilator-pressure-prediction).
`VentilatorDataset` inherits the benchmark's `Dataset` interface.
The comparison reuses `BenchmarkConfig` and `BenchmarkResults`, with a dedicated
training entry point because this is aligned sequence regression with a masked
objective rather than ordinary future forecasting.

The reference is version 6 of theoviel's [Deep Learning Starter: Simple
LSTM](https://www.kaggle.com/code/theoviel/deep-learning-starter-simple-lstm),
retrieved through Kaggle's public kernels API. Its protocol is:

- Five unshuffled GroupKFold splits over CSV rows, grouped by `breath_id`.
- Each sample contains all 80 steps of a breath. No pressure is an input.
- Inputs: R, C, u_in, cumulative u_in within each breath, u_out.
- Training loss: inspiration MAE normalized separately within each breath,
  averaged over the batch. Evaluation: global MAE over all inspiration rows.
- Final-epoch predictions, all five folds, and a global out-of-fold score.
- Adam, batch size 128, 200 epochs, learning rate 0.001 decaying linearly
  to zero, no warmup, shuffled training batches with the final partial batch
  dropped. The TensorFlow translation uses the same architectural dimensions:
  Dense(256), Dense(512), bidirectional LSTM(512), Dense(512), Dense(1).

TensorFlow and PyTorch initialize weights and implement recurrent cells
differently. This is an architecture and protocol translation, not a claim of
bitwise reproduction of the original notebook's score. Both comparison models
use identical data, folds, loss, optimizer schedule, epochs and batch size.
The initial TFTS comparator uses the library's default final-state RNN. The
aligned experiment uses the same library class with opt-in sequence outputs,
input projection, and two bidirectional 128-unit LSTM layers. The existing
forecasting default is preserved. Depth and direction now reach the recurrent
encoder, including correct stacked widths and bidirectional states.

From `Time-series-prediction/`:

```bash
uv sync --extra benchmark
uv tool run --from kaggle kaggle competitions download \
  -c ventilator-pressure-prediction -p ../data/ventilator
unzip ../data/ventilator/ventilator-pressure-prediction.zip -d ../data/ventilator
uv run python -m benchmark.ventilator --data ../data/ventilator/train.csv \
  --threads 4 --output outputs/ventilator
uv run python -m unittest tests.test_ventilator
```

Outputs include the raw CSV checksum, protocol manifest, saved fold IDs,
per-fold weights and training losses, out-of-fold predictions, and results in
the benchmark JSON/CSV formats. Partially filled prediction files contain NaNs;
they are not complete OOF results. Only after all folds finish is a model's
global result written. A shorter `--epochs` run is a pipeline check, not a
reproduction of the notebook's training budget. Keep such outputs separate.

The competition archive has now been downloaded with authorized credentials
and passed its ZIP integrity check. The adapter loaded 75,450 complete breaths
and 2,290,968 inspiration rows from the original training CSV. Credentials are
stored outside the repository in Kaggle's standard `~/.kaggle/access_token`
file with owner-only permissions, so subsequent tasks under the same account
can reuse them. Do not commit competition data or credentials.

The initial real-data CPU runs completed with the same 10-epoch training
budget, batch size 128, seed 42, and all five reference folds. Their complete
OOF predictions were checked for shape, finiteness, identical fold IDs and
source CSV checksum. Independently recomputed global inspiration MAEs:

| Model | OOF inspiration MAE |
| --- | ---: |
| Full reference TensorFlow LSTM | 0.7606188493 |
| Default TFTS RNN | 1.5785328745 |

Their output paths are `outputs/ventilator_reference_10e` and
`outputs/ventilator_rnn_10e`. This shorter training budget is explicitly
different from the notebook's 200 epochs; these are not reproduced notebook
scores.

The next configuration is fixed in `benchmark/configs/ventilator.yaml` before
observing its fold results. It changes only the TFTS architecture, preserving
the five inputs, raw data, fold assignment, loss, optimizer schedule and
training budget. The initial comparison motivated this architecture change;
the outer-fold results are development benchmarks, not an untouched final test.
No validation checkpoint selection, fold-specific hyperparameters or pressure
inputs are used. Reuse the completed baseline and run the new TFTS comparator:

```bash
uv run python -m benchmark.ventilator --config benchmark/configs/ventilator.yaml \
  --models rnn --data ../data/ventilator/train.csv --threads 6
```

Omit `--models rnn` to reproduce both models from scratch in one command.
`return_sequences=True` requires the input length to match the declared
prediction length; it uses the recurrent output at each step, while the
default path continues to forecast from the final state.

The aligned TFTS comparator is configurable as a bidirectional LSTM through
`benchmark/configs/ventilator.yaml` (`rnn_type: lstm`, `bi_direction: true`),
independent of the library's default unidirectional final-state forecasting.

The aligned runs both completed: `outputs/ventilator_aligned_10e` (128-unit
configuration, log `/tmp/ventilator-aligned-10e.log`) and
`outputs/ventilator_aligned_final_10e` (the current `ventilator.yaml`,
512-unit). Each wrote all five fold checkpoints and a complete OOF prediction
array. I recomputed the global inspiration MAE from each saved array against
the raw CSV and each reproduces its recorded `results.json` to the last digit;
all arrays are 100% finite.

| Model | Config | OOF inspiration MAE |
| --- | --- | ---: |
| Full reference TensorFlow LSTM | baseline | 0.7606188493 |
| Default TFTS RNN | final-state forecast | 1.5785328745 |
| Aligned TFTS BiLSTM | 128-unit, 2 layers, per-step | 0.8277517977 |
| Aligned TFTS BiLSTM | 512-unit, 2 layers, per-step | **0.6821537014** |

The 512-unit bidirectional configuration beats the same-data reference
baseline (0.6822 vs 0.7606) under an identical five-fold protocol, optimizer
schedule, epochs and batch size on CPU. These 10-epoch scores are
development benchmarks on the aligned protocol, not reproductions of the
notebook's 200-epoch budget.

Resume an interrupted RNN run with its original saved protocol:

```bash
uv run python -m benchmark.resume_aligned_final --data ../data/ventilator/train.csv \
  --output outputs/ventilator_aligned_10e --threads 6
```

The resume command reads architecture, seed, learning rate, epochs and batch
size from `manifest.json`, checks the CSV checksum and saved breath IDs, and
uses the same training implementation as a fresh run. It trains missing
checkpoints and rebuilds all OOF predictions from checkpoints. Use `--folds 3 4`
only to explicitly retrain those folds. Other models' results are preserved.
Fresh training requires a new output directory; `benchmark.ventilator --resume`
also supports recovery when supplied with the original configuration.

Implementation checks: 530 project tests passed, with seven environment-gated
skips, using the existing Makefile and optional example dependencies:

```bash
make test UV_RUN='uv run --frozen --with matplotlib --with scikit-learn --with optuna'
```

RNN regressions cover configured stacking/direction, GRU/LSTM state shapes,
sequence gradients, padding invariance, declared-length rejection, and config
round trips. The original trained GRU checkpoint was also loaded after the
changes and reproduced its saved fold-0 predictions. The legacy Encoder import
remains the same class. The aligned run's saved fold arrays were checked against
both the completed reference and scikit-learn on the original row grouping.

## Reference transformer versus the TFTS BERT

```bash
uv run python -m benchmark.ventilator_bert --config benchmark/configs/ventilator_bert.yaml \
  --data ../data/ventilator/train.csv --threads 12
```

This comparison implements the objective's reference notebook, cdeotte's
["TensorFlow Transformer [0.112]"](https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-112),
in TensorFlow and evaluates it against the TFTS encoder-only `Bert` with its
per-timestep `regression_head` — the TFTS model whose architecture matches the
reference (an encoder without a decoder).

The reference uses the engineered features (`add_features` plus one-hot
R/C/R__C, RobustScaled to 64 channels) with the three-channel target stack
`[pressure, pressure_diff, pressure_integral]`, 12 transformer blocks
(embed 64, heads 8, ff 128, dropout 0) with 0.7/0.3 skip connections, masked
multi-channel MAE weighted to inspiration rows, cosine-with-restarts learning
rate, and final-epoch global OOF inspiration MAE on the pressure channel.

Both models train on the same five folds, features, loss, optimizer schedule,
epochs and batch size, on CPU only. The comparison lives in
`benchmark/ventilator_bert.py` (`run_comparison`), driven by
`benchmark/configs/ventilator_bert.yaml`. Partial runs (`--folds < 5`) write
per-fold weights and NaNs for the untouched folds; a global OOF result is only
written after all five folds complete. Use `--folds 1 --epochs 1` as a fast
pipeline check before a full budget run.

Implementation checks for this path: `tests/test_ventilator.py` now covers the
three-segment cosine-with-restarts trace, the normalized masked multi-channel
loss (expiration rows contribute nothing), and rank mismatch rejection.

### Historical results (6 epochs, 5 folds, batch 64, CPU)

These results used the earlier preprocessing implementation, which divided
zero-IQR features by `1e-9`. That bug inflated sparse feature values and has
been corrected to use unit scale. The scores below are historical artifacts,
not validated results for the corrected pipeline. Both models must be retrained
in fresh output directories before comparing their scores again.

| model    | fold0    | fold1    | fold2    | fold3    | fold4    | OOF       |
|----------|----------|----------|----------|----------|----------|-----------|
| reference| 7.447909 | 7.577437 | 7.536457 | 7.513165 | 7.437241 | **7.502448** |
| bert     | 7.344947 | 7.407012 | 7.383770 | 7.337602 | 7.275442 | **7.349761** |

For that historical run, TFTS BERT's OOF inspiration MAE was lower by 0.1527
(7.3498 vs 7.5024). This is not evidence of superiority under corrected
preprocessing. Outputs and
the fold indices/breath ids are in `outputs/ventilator_bert_6e/`
(`results.json`, `results.csv`, `*_oof.npy`, per-fold weights and histories);
both OOF arrays are 100% finite and their per-fold scores were re-derived
independently from the saved arrays as a verification pass.

Both transformer models now honor architecture overrides from the YAML
`models` mapping. `benchmark.learning_rate` controls the cosine schedule's
peak. The verifier uses the same model factory, rejects nonfinite or wrongly
shaped OOF arrays, and requires the recorded `inspiration_mae` to match.
Rechecking saved prediction arrays alone does not validate preprocessing.
