# Model review resolution and local PR split

The corrected implementation preserves the reviewed AutoFormer, DLinear, and
iTransformer accuracy in the controlled three-seed experiment. TiDE removes
unused parameters and wires actual covariates; its Stallion validation results
are essentially unchanged, with a small measured regression detailed below.
Diffusion's unrelated direct-head rewrite is removed and the prior experimental
reconstruction model is explicitly deprecated.

## Local branches

The branches form a stack rooted at `83e17f1`, the base of existing upstream
PR #93. Each branch adds the change below to the preceding branch. The first
branch contains only the value-aggregation fix and its independent regression.
The original `master` history and PR #93 are not rewritten or published.

| Branch | Commit scope |
| --- | --- |
| `review/01-autocorrelation-values` | Aggregate over v; regression checks outputs and value gradients |
| `review/02-rwkv-state` | Named tensor states, graph recurrence, T=1 shift, chunk equivalence |
| `review/03-common-config` | Common target_dim/validation, shared defaults, target/input shape helpers |
| `review/04-revin` | Stateless RevIN and iTransformer normalization; formula/inverse regression |
| `review/05-autoformer` | Progressive decomposition port with wrappers, one layer API, serialization, FFT shape fixes |
| `review/06-tide` | Dense encoder port, real covariates, RevIN, corrected scalar temporal head |
| `review/07-dlinear` | Target selection and averaging initialization with static target validation |
| `review/08-diffusion-deprecation` | Deprecate experimental reconstruction; separate design issue draft |
| `review/09-forecast-guardrails` | Registry fit/predict, tensor prediction adapter, independent transformer parity and save/config tests |
| `review/10-validation-report` | Changelog, reproducible experiments, raw measurements, this report |

Review each branch against its predecessor, or cherry-pick its commit after the
listed dependencies. For example, inspect the independently landable fix with
`git diff 83e17f1..review/01-autocorrelation-values`. The final stacked tree
matches the working implementation. Port-specific config removals travel with
the port that stops reading those keys. Normalization epsilon consolidates the
reviewed PR's 1e-5 defaults; it differs from pre-PR subclass defaults of 1e-12.

A read-only GitHub API check found the existing overlapping upstream PR
[93](https://github.com/LongxingTan/Time-series-prediction/pull/93) and no open
PRs in the fork. Its body links no issue. No duplicate PR or issue was opened.
The diffusion proposal is an explicitly unpublished [issue draft](diffusion-design.md).

## Behavior and compatibility

- Transformer parallel teacher forcing always calls `decoder.sequence`, including
  at inference. Tests forbid calls to `decoder.step` in that path and exercise an
  unknown traced horizon. GPU TF32 caused about 2.9e-4 disagreement between
  differently shaped matmuls; the parity test temporarily disables TF32, restores
  its previous setting afterward, and retains the original 2e-5 tolerance.
- AutoFormer private layers have one constructor vocabulary. DecoderLayer always
  returns `(seasonal, residual_trend)`. Encoder/Decoder wrappers retain configs
  and output-shape methods. Unknown activations raise errors. Removed private
  APIs and changed TiDE weight shapes require retraining affected checkpoints.
- AutoCorrelation accepts `factor` for log(length) delay selection. `None` keeps
  the reviewed checkpoint computation, with configurable `max_delays=8`.
  Odd-length inverse FFTs explicitly preserve length, and adjusted key/value
  shapes retain known dimensions through graph tracing.
- RevIN keeps statistics in the call result, never mutable layer state. Its
  formula and gradients preserve the previous normalization math.
- AutoFormer, TiDE, and DLinear select leading `target_dim` channels and reject
  insufficient known input channels in build. iTransformer retains its existing
  all-variate native output; it is not silently changed to univariate output.
- TiDE defaults to `feature_dim=0` for plain series. Set it to the actual shared
  width of `encoder_feature` and `decoder_feature` for covariate inputs. The
  future feature length must match the forecast horizon. Feature gradients and
  prediction sensitivity are tested. The scalar temporal output omits
  LayerNormalization, which would otherwise collapse a one-channel output.
- RWKV uses named tensor fields understood by tf.nest. The existing while-loop
  recurrence is retained to avoid storing every recurrent state as tf.scan would;
  its TensorArray uses the input dtype and shifting needs no conditional branch.
- Forecasting task `call()` returns ForecastOutput. Keras `predict()` now returns
  its prediction tensor through `predict_step`, avoiding a Python dataclass in
  the graph-to-NumPy adapter.

## Accuracy and cost evidence

Both checkouts used the locked environment, CPU float32, one thread per TensorFlow
pool, seeds 11/29/47, and identical data and optimizer settings. Baseline is
`b86a8cd`, the reviewed PR head. These are diagnostic comparisons, not guarantees
for every dataset, seed, or hardware configuration. Timing is indicative only.

Synthetic sine windows: 128 training / 32 validation examples, history 24,
horizon 8, Adam 0.001, 80 updates, batch 16, hidden size 16, one layer, two heads,
FFN width 32, hidden dropout zero. Means across three seeds:

| Model | Before MSE | After MSE | Before MAE | After MAE |
| --- | ---: | ---: | ---: | ---: |
| AutoFormer | 0.294360 | 0.294360 | 0.405367 | 0.405367 |
| iTransformer | 0.027841 | 0.027841 | 0.116548 | 0.116548 |
| DLinear | 0.079378 | 0.079378 | 0.233694 | 0.233694 |
| TiDE | 0.388738 | 0.259780 | 0.423445 | 0.371029 |

AutoFormer/iTransformer/DLinear metrics match exactly for each seed. TiDE's
synthetic mean improves, but seed 29 worsens; the raw records are retained.
TiDE's small-config parameters fall from 4,372 to 1,963; mean traced inference
latency falls from approximately 0.683 ms to 0.374 ms in this run. AutoFormer
latency remains about 5 ms. See [before](accuracy-before.json) and
[after](accuracy-after.json) for all observations.

Stallion: the local parquet used by the previous experiments contains 350 series
with 60 monthly observations. History 24, horizon 6, log1p(volume), default TiDE
config, Adam 0.001, gradient clipping 1.0, batch 64, 100 steps per epoch, at most
30 epochs, patience 6. Random windows stay within the first 54 observations;
last-six-month validation MAE selects the checkpoint. Metrics below are on that
same validation set, not an independent test set.

| Mean across seeds | Before | After | Change |
| --- | ---: | ---: | ---: |
| MAE, original units | 266.5440 | 266.5531 | +0.0034% |
| MSE, original units | 436449.74 | 436493.40 | +0.0100% |
| SMAPE, percent | 46.0638 | 46.1298 | +0.0660 percentage points |
| Parameters | 34634 | 25585 | -26.1% |

The tiny MAE/MSE regressions are much smaller than variation across seeds, but
are reported rather than rounded into a claim of improvement. The original
performance objective is supported on this protocol to that practical limit;
universal non-regression is not established. Raw [before](stallion-before.json)
and [after](stallion-after.json) results include each epoch's validation MAE.

Reproduce from the repository root after placing a baseline checkout in a
separate directory:

```sh
TF_ENABLE_ONEDNN_OPTS=0 uv run --frozen python benchmark/review_regression.py --source BASELINE_DIRECTORY --output before.json
TF_ENABLE_ONEDNN_OPTS=0 uv run --frozen python benchmark/review_regression.py --output after.json
TF_ENABLE_ONEDNN_OPTS=0 uv run --frozen python benchmark/review_stallion.py --source BASELINE_DIRECTORY --data reference/pytorch-forecasting/examples/data/stallion.parquet --output stallion-before.json
TF_ENABLE_ONEDNN_OPTS=0 uv run --frozen python benchmark/review_stallion.py --data reference/pytorch-forecasting/examples/data/stallion.parquet --output stallion-after.json
```

## Validation

- `make test`: 517 tests run, 3 skipped, no failures. Subsequent focused checks cover the
  final constructor cleanup, unknown-horizon guard, and reorganized tests.
- `make style`: formatters, Flake8, and all pre-commit hooks passed.
- `make docs`: passed with eight warnings after supplying Pandoc from a temporary
  `uv run --with pypandoc-binary` environment; the repository lockfile was unchanged.
- The standalone value-aggregation regression fails against the pre-PR layer
  and passes after the fix.
- The registry sweep checks all registered sequence models with `(B,T,C)=(2,24,1)`,
  horizon 3 and target_dim 1, including graph-mode fit and predict. Spatial
  models must reject that layout; their node-aware tests remain in the existing
  spatial suite. Additional changed-model tests cover multiple target channels,
  save/load, config round trips, normalization, covariates, and recurrent state.
