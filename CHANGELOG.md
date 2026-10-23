# Release notes

## v0.0.24

- Add `Estimator` and `Forecaster` with lazy model construction, fitted state,
  train-only scaling, lazy dataframe windows, explicit future covariates, and
  versioned model/preprocessing artifacts. Keep existing simple batch/config contracts.
- Deprecate `TaskPipeline`, `ForecastingPipeline`, and `pipeline()` in favor of
  the estimator. They delegate to one implementation and remain available for
  at least one minor release. Legacy pipeline artifacts remain loadable.
- Register forecast heads independently of task dispatch. Learned heads emit
  any horizon, quantiles are ordered, and defaults select the matching objective.
  Versioned task configs preserve the original head shapes for older artifacts.
- Allow sequence regression with any backbone exposing temporal representations.
- Track exact weighted losses for point, quantile, and distribution objectives;
  mask target operands before loss arithmetic and support optimizer loss scaling.
- **Breaking:** task calls now reject unsupported covariate fields rather than
  silently ignoring them. DataProcessor rejects gapped/nonfinite history and
  `fill_missing_dates=True`; regularize data explicitly. Custom training objectives
  implement `reduce(batch, output) -> LossResult(total, weight)` and should be
  registered for serialization.

## v0.0.23

- Fix AutoCorrelation aggregation over value projections and odd-length FFT reconstruction.
- Preserve independent parallel transformer teacher forcing; test equivalence in full float32.
- Centralize target dimensions and shared config defaults; add reusable RevIN.
- Use named RWKV tensor states with graph-compatible recurrence.
- Restore AutoFormer encoder/decoder wrappers and serializable layer configs.
- Wire TiDE to supplied history/future covariates and remove zero-feature compute.
  Correct its temporal output head; covariate-free inputs use `feature_dim=0`.
- Deprecate the experimental Diffusion reconstruction model pending a conditional
  diffusion design; remove the unrelated direct forecast-head rewrite.
- Add registry-wide forecast shape/fit/predict checks and Keras forecast tensor output.
- AutoFormer private layer constructors and TiDE weight shapes change; retrain
  affected checkpoints.


## v0.0.22

- Add typed spatial arrangements and topology inputs to ``TimeSeriesBatch``.
- Add experimental STGCN layers and forecasting support for shared or batched
  dense graphs.
- Preserve shared graph metadata across Keras/``tf.data`` boundaries without
  copying static topology into every training window.
- Add documented graph builders and a runnable spatial forecasting example.
- Introduce a shared time-axis generation pipeline, typed `GenStep` hooks,
  parameter and value processors, stopping criteria, and sample aggregation.
- Resolve decoder capabilities when task models are built. RNN forecasts now
  honor declared padding masks for variable-length histories.
- **Checkpoint break:** Transformer causal cached decoding uses decoder format 2.
  Legacy decoder checkpoints require migration or retraining; config-based
  loaders reject missing or incompatible format markers before loading weights.
  This change must ship as a minor release (at least 0.1.0).
- Custom rollout strategies are runtime instances; serialized configs support
  only the implemented built-in strategies. Remove unused generation quantiles.
- Use `PointSampler` (`sampler="point"`), `TeacherForcingPolicy`, and
  `TimeAxisEngine`.


## v.0.0.15
- fix classification/anomaly detection
- fix from_pretrained


## v0.0.13
- support training style of transformers
- solve pandas version conflict
- support cnn/rnn model for keras3


## v0.0.4 Add models support (21/11/2022)

### Added
- input support
    - tf.Data or array
    - single item or three items

- function support
    - serving

- model support
    - tft
    - nbeats
    - unet
    - informer
    - deepar

## v0.0.3 Add models support (21/10/2022)

### Added
- function support
    - serving

- model support
    - tft
    - nbeats
    - unet

## v0.0.2 Add function support (1/10/2022)

### Added
- function support
    - classification

## v0.0.1 Initial release (15/03/2022)

### Added
- model support
    - rnn
    - tcn
    - bert
    - seq2seq
    - wavenet
    - transformer
- example support
    - sine data
    - air passenger data
- train
    - trainer
    - keras_trainer

### Contributor
- LongxingTan
