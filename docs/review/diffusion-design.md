# Conditional diffusion forecasting: design issue draft

Status: local draft; no issue has been published. The experimental `Diffusion`
class is deprecated. This review removes the direct forecast-head rewrite and
retains the pre-PR reconstruction implementation for existing callers during
its deprecation period. It must not be presented as a validated conditional
forecaster. Accuracy claims for the rejected forecast head do not transfer to
this implementation.

The existing implementation reconstructs a noisy history tail. A proper
conditional diffusion forecaster must diffuse the future target window while
conditioning on clean history and known future covariates. Training must sample
a timestep and minimize error between predicted and injected noise. Forecasting
must initialize future targets with noise and run an actual reverse sampler.
`NoiseScheduler.remove_noise` currently estimates the clean sample; it is not
by itself a DDPM reverse transition.

A replacement proposal must specify:

- The batch/objective contract for future targets, masks, and covariates.
- A prediction-of-noise objective, timestep embeddings, and conditional denoiser.
- Reverse-transition coefficients, variance, final-step behavior, and random seeds.
- Whether DDPM sampling is the baseline and whether a separately configured
  accelerated sampler is supported; sampling steps cannot silently become one.
- Integration with `generate`, sample axes, and distribution evaluation.
- Tests for the analytic noising identity, reverse transitions, seed
  reproducibility, finite gradients, dynamic horizons, and serialization.
- Accuracy and sampling-cost comparisons on fixed dataset splits with repeated
  seeds. A deterministic direct transformer belongs under its own name if it is
  proposed again.

Implementation of that replacement is intentionally outside the model-refactor
stack. The deprecation warning and this issue draft make the current limitation
explicit without pretending a new diffusion algorithm has been validated.
