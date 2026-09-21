# Roadmap

GLLVModels.jl is being built as a fast Julia **companion to
[`gllvmTMB`](https://itchyshin.github.io/gllvmTMB/)**: the same core estimands
where paired, usable directly in Julia and through a **narrow** R bridge
(`engine = "julia"`). The bridge does not yet cover every R workflow, and an
inventory of matching functions is not true parity; see
[Capability parity](gllvmtmb-parity.md) for the currently documented scope. A
full 0.7.1 surface port (column_coef, slopes, formula grid) is not part of the
current release scope.

Current sequencing is R-first. Native `gllvmTMB` functionality and the R user
workflow define the reference behavior; `GLLVModels.jl` adds a Julia route only
after its point estimates, log-likelihood, confidence intervals (or their
availability status), documentation, and tests agree at the stated scope. REML
is Gaussian-only; AI-REML is future work for exact Gaussian cells, not
non-Gaussian Laplace.

## Phase → release map

| Release | Theme | Highlights |
|---------|-------|-----------|
| **v0.2.0** | Gaussian complete | closed-form marginal, O(p) phylogenetic fitter, post-fit tools, this docs site |
| **v0.3.0** | Non-Gaussian catch-up | one-part Laplace families, first two-part fitters, analytic-gradient hardening |
| **v0.4.0** | Interface and bridge catch-up | `@formula` front-end, wide/long parity, gllvmTMB-mirroring tutorials, and a live `gllvmTMB` bridge |
| **v1.0** | True-parity milestone (aspirational) | Targets documented agreement with frozen gllvmTMB 0.7.0 beyond point estimates across realistic-size data, real-data workflows, and grouping-level pairing; it does **not** promise a complete Julia bridge for every R workflow, and an inventory of matching functions or a "complete bridge" label alone is not enough |

## What works today

- Gaussian + phylogenetic GLLVM fitting (closed-form Gaussian marginal).
- An **O(p)** phylogenetic gradient — exact, linear-in-species scaling.
- Wald / profile-likelihood / parametric-bootstrap confidence intervals,
  including derived quantities (Sigma_y, communality, phylogenetic signal), where
  the relevant family and structure have their own documented checks.
- One-part Laplace families through `fit_gllvm`: Binomial, Poisson,
  NegativeBinomial, Beta, Ordinal, and Gamma.
- Wald/profile/bootstrap confidence-interval routes for scalar-dispersion
  one-part Laplace families and grouped NB2/NB1/Beta/Gamma; grouped Tweedie and
  per-trait ordinal endpoints remain explicit unavailable status rows while
  parity and bridge exposure are audited.
- Native ordinary `X_lv` trait-effect inference for Gaussian, Poisson, NB2,
  Binomial, Beta, Gamma, and shared-cutpoint Ordinal fits, including
  selected-entry profile-likelihood canaries for `B_lv`.
- Dedicated two-part fitters for Delta-lognormal, Hurdle-Poisson, and
  Hurdle-NB.
- Minimal Julia-side `bridge_fit` for no-covariate one-part families, selected
  fixed-effect-X / missing-response rows, selected ordinary `X_lv` point rows
  tested by paired `gllvmTMB` branches, and guarded mixed-family point/post-fit
  rows.

## What's planned

- **Non-Gaussian inference hardening** — R-parity evidence, derived covariance
  summaries, and bridge exposure beyond the current one-part CI routes.
- **Structured non-Gaussian dependence** — phylogenetic, animal, and spatial
  covariance in the Laplace path.
- **Zero-inflated and additional two-part families** — ZIP/ZINB and Delta-Gamma
  after the current two-part substrate hardens.
- **Same-as-R model syntax**: `gllvm(@formula(traits(...) ~ ... + phylo(...)), data; family = ...)`.
- **The R bridge** — keep the live `gllvmTMB` JuliaCall route as the admission
  oracle, then widen deliberately from the current partial rows: fixed-effect
  `X` for selected one-part families, response masks for selected no-X
  non-Gaussian families, complete balanced mixed-family no-X/no-mask/no-CI point
  fits, and in-sample post-fit methods. Remaining bridge work is X+mask,
  mixed-family X/masks/CIs, broader newdata contracts beyond the existing
  fit-specific prediction routes, richer diagnostics, and parity evidence for
  every promoted row.

This roadmap evolves as new model routes earn their evidence.
