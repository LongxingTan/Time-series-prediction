# Fork architecture: revised design

Status: partially implemented in the working tree. This document refines the supplied review for
the fork at `2e13a13`. The current architecture documentation describes the
running implementation; this document defines its replacement. The restructure
lands atomically, deletes obsolete modules, and provides no import shims or
old-artifact compatibility.

Keep the review's five decisions: a time-series-only core, one forecasting
interface, types owned by their producers, data organized by stage, and CI
enforcement. Tighten the contracts before moving files. Moving today's coupling
into a better-looking tree would preserve the underlying problems.

## Implementation status

The data and ownership changes are implemented: `contracts/` is removed, task
configs and outputs live with their owners, task builders live in `tasks/`, and
backbones use the shared registry. `FramePipeline` owns frame preparation through
regularization, fitted features, training-only scaling, temporal splits and a
compact NumPy window index. TensorFlow materialization is a separate boundary.
Raw loaders and tutorial preparation have separate homes. Estimator input sources
share one preparation result, and CI checks the boundaries established so far.

The complete target architecture below is still pending. In particular, the batch
still supports spatial inputs, configs are not yet frozen, forecasting does not
yet use the proposed universal predictor interface, and the existing training and
generation implementations remain. These are not compatibility shims for the
removed data or contracts modules; they are subsequent implementation work.

## 1. Decisions that change the proposal

| Proposed rule | Revised decision | Reason |
| --- | --- | --- |
| Every field is rank 3 | Temporal values are `(B,T,C)`; static values, labels, samples and quantiles have explicitly different ranks | The original dataclasses already contain legitimate exceptions |
| Target masks also cover padding | Keep per-field observation masks and a separate time-validity mask | A missing measurement is a real time step; left padding is not |
| Forecast output always has `mean` | Require `point` with a declared statistic; expose `mean` only when available | A median or a point trained with MAE is not an expectation |
| Forecast output belongs to a Keras task | Put the shared forecast result in `forecast.py` | Baselines and runtime adapters also produce it; they should not import tasks |
| `PredictorSpec` is a class variable | Expose an immutable instance property | Checkpoint, head, decoder and fitted shape determine actual capabilities |
| All predictors use `tf.data.Dataset` | Sources expose lazy named array batches; the Keras adapter adds `tf.data` | NumPy baselines and Torch adapters do not need a TensorFlow execution pipeline |
| Spatial predictors implement the exact rank-3 request protocol | Use a generic request/result protocol, specialized by time-series and spatial APIs | A rank-4 request is not substitutable for a rank-3 request |
| Load every plugin on first lookup | Discover metadata lazily; activate a provider explicitly or when its qualified entry is selected | An installed extension must not import Torch or modify ordinary lookups |
| Registry validates ports by importing `BackboneOutput` | The backbone kind registers its own tag validator | Core registry must not depend on a higher layer |
| Symmetric averaging is a postprocessor | It is a predictor wrapper making two forecasts | It needs a second network invocation and input transformation |
| Stage 5 is the only thing that learns | All fitting is restricted by the split, including feature selection, clipping and encoding | A transform having `fit()` before the split is a leakage route |
| Every predictor must pass gaps, arbitrary horizon and quantile tests | Generate positive and rejection cases from effective capabilities | Point baselines and fixed-horizon tasks cannot promise all outputs |

The additional neutral `forecast.py` is justified by the same ownership rule as
`batch.py`: tasks and independent forecasting implementations both need it.
There is no numerical limit on shared types; each needs a demonstrated dependency
reason. Avoid re-creating an unbounded `contracts/` package under another name.

## 2. Public boundaries and dependency graph

Retain TensorFlow/Keras as the training backend. Do not build a general tensor
backend abstraction. At the external predictor boundary, use NumPy arrays with
explicit dtypes; `TaskPredictor` converts once into TensorFlow and once back.
Expert Keras callers use tensor dictionaries and typed internal outputs directly.
Foundation adapters convert NumPy to their native runtime. This simple boundary
may copy host data; device-native interchange is a separately measured future
optimization, not a hidden promise of zero-copy interoperability.

`Batch[ArrayT]` is a generic immutable container with named fields. Core structural
validation uses shapes and dtypes without importing TensorFlow. The NumPy boundary
normalizes raw arrays; the Keras boundary performs graph-safe dynamic assertions
and mask construction. No NumPy conversion occurs inside a traced Keras call.
TensorFlow remains an installation dependency in this restructure; avoiding its
eager import is distinct from offering a TensorFlow-free distribution.

Allowed direct internal dependencies are explicit rather than inferred from
an ambiguous diagram. Each row also permits Python and its approved third-party
dependencies.

| Package | Allowed tfts dependencies |
| --- | --- |
| `batch.py`, `forecast.py`, `config.py`, `registry.py`, `constants.py` | No higher package; keep registry independent of the other type definitions |
| `layers`, `distributions` | Core files; no imports between these sibling packages |
| `models` | Core, layers, distributions |
| `data` | Core; no models, predictors or tasks |
| `datasets` | Core; raw loaders only |
| `tasks` | Core, models, layers, distributions |
| `predictors` | Core, tasks; task imports isolated to `predictors/task.py` |
| `training` | Core, tasks |
| `evaluation` | Core, predictors, `data.split` |
| `estimator` | Core, data, predictors, training, evaluation |
| `generation` | Core, models, tasks, predictors |
| `foundation` | Core, predictors; reference runtime libraries |
| `tuning` | Core, estimator, evaluation |
| `export` | Core, estimator, predictors, tasks |
| `cli` | Core, estimator, datasets |

Optional modules have different dependencies: tuning necessarily uses the
estimator and export needs fitted transforms. Do not claim they all depend only
on tasks and predictors. Nothing in the non-optional packages imports them,
including local imports and imports hidden behind `TYPE_CHECKING`.

Only data, datasets, evaluation and estimator import pandas. Only foundation
directly imports Torch, ONNX Runtime, safetensors or huggingface_hub. Core never
imports an extension distribution. The public `__init__` lazily exposes
`Forecaster`, `Estimator`, `Schema`, `TemporalSplit`, `load`, `build`, `available`
and `describe`, without importing optional providers. Internal modules use the
owning modules, not these convenience exports.

## 3. Batch, masks and inference isolation

Keep the proposed target and covariate names and lengths. Every temporal numeric
or categorical role has its own observation mask with the same shape. In
particular, `past_mask` cannot describe missing `past_covariates` of another width.

| Values | Shape | Observation mask |
| --- | --- | --- |
| `past_target` | `(B,T,D)` | `past_mask` |
| `future_target` | `(B,H,D)` | `future_mask` |
| `past_covariates`, `past_cat` | `(B,T,C)` for each role's width | `past_covariates_mask`, `past_cat_mask` |
| `future_covariates`, `future_cat` | `(B,T+H,C)` for each role's width | `future_covariates_mask`, `future_cat_mask` |
| `static_real`, `static_cat` | `(B,C)` | Corresponding role mask |

Add `past_valid: (B,T)` for real time steps versus padding. All missing values
inside a valid time step retain their temporal position. Attention masks can
exclude padding without removing a real gap. An observation implies validity.
Absent roles have neither values nor masks; present roles always have masks after
normalization. Do not materialize fictitious future labels just to create a mask.

At raw input boundaries, combine a supplied mask with non-NaN presence, replace
missing numeric payloads by zero, and reject infinities. Never infer missingness
from numeric zero. Category code 0 means unseen; a false categorical mask means
missing. Preserve integer codes without lossy float32 conversion. Vocabularies
are per column, with cardinality including the unseen slot.

Structural checks belong to the container; temporal alignment with a requested
horizon belongs to request validation; target and weight shapes belong to the
task. A classification batch has `(B,)` labels and class-compatible weights,
whereas sequence regression has per-step labels and an explicit `labels_mask`.
Imputation must distinguish observed inputs from deliberately held-out values
used for its loss. Do not score imputations against the mask that hides them.

`Batch.from_inputs` continues rejecting positional tuples. Keras's outer
`(inputs, targets, sample_weight)` envelope remains valid and is unpacked by the
task boundary; it is not interpreted as three positional covariate roles.

`ForecastRequest` contains an inference batch, positive horizon, optional sorted
unique quantile levels in `(0,1)`, optional positive sample count, and an optional
random seed. Its validator rejects `future_target`, `labels`, label masks and
training weights. Evaluation removes supervision before constructing a request.
This makes future-target leakage fail at the public boundary instead of depending
on each backbone remembering to ignore labels.

## 4. DataSpec and immutable configuration

Separate user choices from inferred facts:

- Architecture config: hidden width, layers, dropout and architecture choices.
- Task config: output block length, objective, head and target semantics.
- Window specification: desired context, stride, horizon and sampling policy.
- `DataSpec`: validated role widths, ordered columns, categorical cardinalities,
  dtypes, training context/block shapes and input signatures.
- `PredictorSpec`: effective inference limits and supported operations.

Horizon and context are often supplied by the user, not learned. `DataSpec`
records their resolved training values but does not prohibit a decoder from
serving another inference horizon. Store separate training and inference tensor
signatures; inference must not require training labels. Stable dictionary keys
reduce retracing, but dynamic batch dimensions and declared supported lengths
also need explicit signatures and tests.

Frozen dataclasses alone are shallow. Copy and freeze nested mappings and
sequences, and serialize them to plain JSON through one canonical encoder.
Unknown config keys fail. `dataclasses.replace` produces a new configuration.
`set_params` changes constructor choices and invalidates fitted state; `fit`
creates fitted clones and never mutates the caller's config or predictor.

## 5. Forecast results and predictor lifecycle

`forecast.py` owns `ForecastOutput[ArrayT]`. Required fields are `point: (B,H,D)`
and `point_statistic` (for example `mean`, `median`, or `unspecified`). Optional
fields are an actual `mean`, quantiles `(B,H,D,Q)` with their levels, samples
`(B,S,H,D)`, and typed distribution parameters with a distribution identifier.
`prediction()` returns `point`. There is no automatic median-to-mean conversion.
Finite quantile grids do not determine a mean without tail assumptions.

Forecast outputs enforce matching batch, horizon and target dimensions. Requested
quantiles are returned in requested order after validation; linear interpolation
is allowed only inside the native range unless an explicit tail policy exists.
Unsupported samples, distributions or quantile levels fail before inference.
Marginal quantiles cannot manufacture coherent sample trajectories.

`Predictor[RequestT, OutputT]` defines `predict(request)`. Its time-series
specialization uses `ForecastRequest` and `ForecastOutput[np.ndarray]`. The
effective `spec` is a read-only instance property. Static registration metadata
describes the family; loading a checkpoint and composing its head and decoder
produce the resolved instance spec. Store capabilities once and derive views,
rather than maintaining contradictory booleans in registry, task and estimator.

Keep backbone ports distinct from predictor outputs. A backbone advertises
`tokens`, `temporal`, `forecast` or `dist_params`. A predictor advertises point,
quantiles, distribution or samples. A head can change output capabilities, and
a decoder can change supported horizons. No single flat tag table should erase
these distinctions; tag schemas are scoped by registration kind.

Replace `trainable: bool` as the lifecycle switch with `fit_mode`:

| Fit mode | What `Forecaster.fit` does |
| --- | --- |
| `none` | Validates input and fits permitted preprocessing; does not call network training |
| `statistics` | Fits baseline state as well as preprocessing |
| `gradient` | Delegates training to the Keras task |

Fine-tuning is a separately declared operation, never inferred from whether
weights exist. `TaskPredictor` exposes training through its adapter; baselines
implement their own statistical fit; pretrained implementations need not define
a misleading no-op `fit` method. The estimator uses lifecycle capabilities and
public factories, not task-name checks or imports of concrete task classes.

All forecast requests supply history. For dataframe convenience,
`Forecaster.predict()` may use a stored trailing context, but that context must
be explicitly retained and saved if prediction without history is supported
after reload. Otherwise require history consistently before and after saving.
Use the latter, simpler rule for the first implementation.

## 6. Data stages and leakage guarantees

Keep the proposed stage files, one scaler family and one window implementation.
Change the execution order to make the causal guarantee real:

1. Validate schema, group keys, timestamps and declared frequency.
2. Regularize each group independently. Reject duplicate and off-grid times;
   represent inserted rows through masks. Do not infer frequency from held-out
   rows. Timezone and DST policy are explicit schema state.
3. Assign raw timestamps to train/validation/test by forecast origin and label
   interval. Define any gap in time steps, not vaguely as a lag count.
4. Fit the transform graph only on eligible training rows. This includes clipping,
   vocabularies, scaling and any learned feature selection.
5. Transform chronologically with frozen state and causal historical context.
6. Build a NumPy `WindowIndex`, then lazily gather named array batches.
7. Wrap those batches in `tf.data` only for the Keras adapter.

These are execution phases, not a requirement to add seven stateful abstractions.
`FramePipeline` owns their sequencing; feature transforms declare whether they
learn state, their required history, and whether their inputs are known at the
forecast origin. No transform may read future targets. Target lags over future
steps are not known-future covariates: they need forecast feedback or must be
rejected for that horizon. Centered rolling windows are not causal features.

Train windows have all labels in the train interval. Validation windows may read
earlier observed history, including training history, but every scored label is
in validation. A rolling-origin evaluation states whether observations revealed
between origins become available; fixed-origin evaluation does not reveal them.
Each cross-validation fold fits new transforms and new trainable model state.
An embargo does not repair a non-causal transform.

`WindowIndex` stores compact integer group/start arrays and offsets into flat role
arrays, never materialized overlapping windows. Array and dataframe sources use
the same window geometry. Already-windowed dataset sources validate their
contract and do not window a second time. Unknown or infinite dataset cardinality
requires explicit steps or a finite preparation policy; do not exhaust a stream
to infer its shape.

`PreparedData` contains a reusable batch source, optional validation source,
`DataSpec`, fitted transforms, and an origin/group/time index for aligning results.
It does not require `tf.data.Dataset`. Sources receive explicit preparation
options, not the estimator object. This keeps preprocessing independently usable
and prevents sources from reaching into estimator internals.

The estimator resolves normalization policy and passes concrete transform choices
to data. Data never imports `PredictorSpec`. Internal normalization defaults to
identity target scaling, with an explicit override required for double scaling.
Global/per-group scaler choices define behavior for unseen groups and all-missing
training columns; default to a clear error unless a fitted global fallback was
requested. Never fit that fallback on prediction history implicitly.

Dataset-specific preparation moves to examples. Raw reusable loaders stay in
`datasets` and benchmarks reuse them. The rule concerns dataset-specific feature
engineering, not the unavoidable dataset names in loader modules.

## 7. Tasks, losses and decoding

Keep one task file per task, heads by family, and one losses package. Heads declare
ports and produce their own output types. Native forecasting is an identity head;
distribution heads declare the distribution family and parameter constraints.
Compatibility checking reports all mismatches before building or fitting.
Explicitly select a compatible registered default head rather than dispatching
on head-name strings in the forward method.

Preserve the actual reduction contract already present at `2e13a13`:

```text
LossResult(total, weight)
value = divide_no_nan(total, weight)
epoch_loss = sum(batch.total) / sum(batch.weight)
```

Observation masks and nonnegative sample weights define effective weights with
explicit broadcasting. Average the quantile axis before applying per-target
weights for pinball loss, so increasing Q does not silently change its scale.
Mask operands before arithmetic to prevent NaN gradients. An all-zero-weight
batch contributes neither numerator nor denominator and skips optimizer updates;
an epoch without any supervised weight raises. Report regularization separately
from weighted data loss, while including it in the training objective.

Distributed training must differentiate the global weighted objective, not the
mean of replica-local means. Verify unequal replica counts, gradient accumulation
over unequal microbatches, mixed precision and regularization scaling against one
equivalent concatenated batch. `Task.train_step` owns the sole optimizer gradient
loop; `run_eagerly=True` is the debugging mode. Evaluation and callbacks do not
compute independent training gradients.

Separate fixed-block native computation from extending a forecast across blocks.
Backbones own their fixed-block forward method and native incremental state;
shared native hook types live with their producer in `models/`, not in optional
`generation/`. Tasks can train a native fixed block without importing generation.
Optional generation owns inter-block assembly and feedback. A sequence loop
inside a recurrent backbone is not a second forecast-assembly loop.

The optional decoder is composed into a predictor explicitly, never looked up
from `Task.forward`. A registered provider can construct that composition when
selected. Missing optional support produces an actionable error rather than an
import triggered accidentally by ordinary training.

After each generated block, target feedback uses the declared statistic or sampled
trajectory, known covariates use their true supplied future values, and past-only
covariates become unavailable. Track generated versus observed target provenance
in decoder state; a generated value may be usable for conditioning without being
an observed label. Trim context and its masks together. Crop final forecasts to
the requested horizon. Seeded sampling is reproducible and distinct samples
follow distinct feedback paths.

## 8. Timestral: wrap before extracting abstractions

The reference check is pinned to Timestral commit
`a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`, not a moving branch. Its public forecaster
already owns block decoding and preprocessing. The first adapter delegates to
that public forecaster, transposes boundary arrays, restores missing covariates
as NaNs from their masks, and maps outputs back. It does not add a second tfts
decoder around the reference decoder.

The reference returns quantiles and a median, so the adapter's point statistic
is `median`. Its categorical target encoding applies to univariate targets;
multivariate targets receive codes directly. Capability resolution is checkpoint
specific. Symmetric averaging runs a reflected second forecast, not just a
transformation of the first output. See the pinned
[forecaster implementation](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py).

Preserve reference preprocessing, clipping and block refinements initially.
Moving any of them into common components requires full-forecast parity,
including horizons crossing a block boundary, partial final blocks, missing
targets/covariates, multiple targets, categorical inputs and context truncation.
Specify dtype, device, tolerances, flags, checkpoint revision and input fixtures.
Mocks test adapter plumbing; only a real checkpoint/reference comparison proves
numerical parity. Record that evidence before claiming readiness.

Weights remain externally downloaded on explicit model loading. Record the
resolved immutable revision, adapter/reference versions, file hashes and license
metadata. The reference documents separate code and weight licenses in its
[pinned README](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/README.md).

## 9. Postprocessing must preserve statistical meaning

Postprocessors consume an already computed output. Predictor wrappers may invoke
the inner predictor again. Symmetric averaging belongs to the second category.
Fitted calibration state is separate from both model weights and training
preprocessing and uses a declared held-out calibration interval.

Inverse affine scaling can transform means, quantiles and supported distribution
parameters analytically. Under nonlinear transforms such as log1p, inverse(mean)
is generally not the original-scale mean. Transform samples then aggregate, use
a distribution-specific formula, or report only statistics justified by the
available output. Monotone inverse transforms preserve quantiles; arbitrary
clipping does not preserve a distribution family. Never retain stale parameters
after changing the represented forecast.

Record the ordered processing configuration in artifacts. Quantile crossing
repair is explicit and tested, not a silent way to hide an untruthful head.
Evaluation records which point statistic and quantile transformations were used.

## 10. Registry and spatial extensions

One registry owns `(kind, qualified_name)` entries. Each record contains a factory,
immutable metadata, a config schema and provider/version information. Each kind
owns its metadata validator. Port validation is installed by models and derives
port names from `BackboneOutput`; the registry never imports models to do so.

Built-in discovery uses an explicit lazy catalog. `available` reads catalog
metadata without importing every implementation. Plugins use `tfts.plugins` for
discovery; provider activation runs once in deterministic order and commits
registrations transactionally. Duplicate names, unknown tags and incompatible
plugin API versions fail; there is no silent override. Support disabling plugin
discovery for reproducible isolated runs. Registration functions register
factories without loading weights or optional runtimes.

Spatial types, source logic, models and TensorFlow layers move into a separate
distribution. `PerNodePredictor` accepts the spatial package's request, folds it
into an ordinary core forecast request, then reconstructs a spatial result. It
implements the generic predictor protocol with different type arguments, not
the time-series specialization. Spatial-native predictors do the same at their
own boundary. The extension contract tests verify node permutation, masks,
static-feature broadcasting, grouping and restored output axes.

Do not promise the dataframe `Forecaster` or time-series backtester accepts any
spatial input unchanged. Shared orchestration and artifact machinery can be
reused through registered source/serializer strategies; the extension provides
its typed facade and spatial metric reduction. Core contains no spatial type
names or rank branches. A monorepo with two distributions is acceptable.

## 11. Artifacts and estimator state

Use a common manifest with a versioned predictor payload:

```text
artifact/
  manifest.json         # format, provider, resolved capabilities, file hashes
  estimator.json        # constructor configuration and fitted DataSpec
  schema.json           # ordered roles, dtypes, frequency, timezone policy
  transforms.json       # preprocessing state and transform versions
  predictor/            # payload selected by its registered serializer
    model.keras         # Keras predictor
    OR state.json       # statistical baseline
    OR pretrained.json  # model identity and resolved immutable revision
  calibration.json      # only when fitted calibration is used
```

A serializer belongs to the predictor provider; estimator code does not branch
on model names. The manifest hashes all payload and transform files, not only
model weights. Save through a temporary sibling directory and atomic rename.
Loading validates format and hashes before restoring objects; never install or
download a provider implicitly. A pretrained artifact supports offline cache
resolution or an explicit download request, with a clear cache-miss error.

Validate incoming frames against the stored schema and reorder by stored column
names. Hash the canonical schema definition, not raw dataframe column order or
the dtype of a newly inferred schema. Define permitted lossless dtype conversions;
renamed roles and incompatible dtypes fail. Hash equality alone is not semantic
validation. Persist category vocabularies, group fallback policy and calibration
state so reload reproduces forecasts.

Refitting replaces all fitted state together after successful preparation and
training. Failed fits must not leave a mixture of new transforms and old weights.
No learned history is smuggled into batch metadata.

## 12. Enforceable checks and implementation gates

Use import rules matching the dependency table, with dedicated tests for lazy
imports and runtime behavior. AST checks cover absolute and relative imports,
cross-package private module imports, and private attribute calls where statically
resolvable. Do not claim a simple import scan detects every dynamic call.

Compare registrations against an explicit catalog of architecture modules, not
all files in `models/`: helpers such as `_autoregressive.py` are not backbones.
Run contract tests on registrations, with declared support and rejection cases:

- Structural validation, field masks, static/dynamic shapes and inference-label
  rejection; padding distinct from missing observations.
- Head/port compatibility, capability composition and rejection of unsupported
  covariates, missingness, horizons, samples and quantiles before network work.
- Keras train/predict and config/artifact round trips for trainable tasks; NumPy
  baseline round trips; pinned reference parity for foundation integrations.
- Frozen nested configs and unchanged constructor parameters after fitting.
- Future-target perturbation invariance, no cross-group windows, split-boundary
  checks, training-only fitted state and fold-local preprocessing.
- Weighted losses and gradients, including unequal replica/microbatch weights,
  all-missing data and all-zero weights.
- Stable trace counts for supported signatures, not an impossible global promise
  of no retracing across every shape, execution mode and runtime.
- Quantile levels, statistic labels, nonlinear inverse scaling and seeded sample
  trajectories; backtest metrics use the proper training-only MASE denominator.
- Lazy import in fresh subprocesses; core installed without Torch or spatial;
  selected plugin activation, collisions and provider failures.

Core examples run offline with `--smoke`. Optional examples run in separate jobs
with their declared dependencies. A fixture smoke run is distinct from a real
checkpoint integration test. Network downloads are never required for ordinary
core CI. Use unittest, matching this repository's test suite.

Build in vertical gates while delivering one final breaking change:

| Gate | Deliverable | Required evidence |
| --- | --- | --- |
| 1 | Characterization tests, dependency rules and revised type definitions | Current direct, native and probabilistic behavior recorded; known structural violations explicitly enumerated |
| 2 | Shared types and registry; one complete DLinear/point/array path | Fit, predict, masks, weighted loss and artifact round trip |
| 3 | Task/head migration and spatial extraction | No contracts imports or models-to-tasks cycle; standalone spatial distribution tested |
| 4 | Data stages, sources and frozen fitted state | Causality, gaps, category handling, dataframe/array equivalence, bounded lazy gathering |
| 5 | Predictor lifecycle, baselines, estimator and serializers | One estimator orchestration path; all three fit modes and baseline persistence |
| 6 | Native generation and reference Timestral adapter | No optional imports from tasks; multiblock behavior and pinned reference parity |
| 7 | Remaining backbones, examples, CLI and cleanup | No dead paths/shims, all dependency checks green, core and optional CI split |

Early gates may use a checked-in, shrinking list of existing architecture
violations. New violations fail immediately, and the final gate removes the list.
Do not merge a permanently failing CI suite or describe temporarily failing
structural tests as a passing milestone. Representative behavior is preserved
while old import paths and artifact formats are intentionally broken.

The proposed package tree otherwise stands, with these adjustments: add
`forecast.py`; retain model-owned native step types; isolate Keras conversion in
`data/tf_dataset.py` and `predictors/task.py`; add predictor wrappers separately
from output transforms; and keep source/serializer extension contracts explicit.
Prefer these proven boundaries over adding a public class for every pipeline box.

## Sources and scope

- [PR #95 at the reviewed commit](https://github.com/LongxingTan/Time-series-prediction/tree/2e13a13)
  and the local files at that revision establish the implementation baseline.
- [TSKit design comment](https://github.com/LongxingTan/Time-series-prediction/pull/95#issuecomment-6356598610)
  supplies the original composition and estimator proposal.
- The pinned Timestral sources above establish adapter behavior. They do not
  establish numerical parity for an adapter that has not yet been implemented.

This revision is an architectural decision document. Implementation, migration
tests and reference parity remain work to execute at the gates above.
