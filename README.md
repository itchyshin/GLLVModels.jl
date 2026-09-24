# GLLVModels.jl

Development grouping route: `fit_gllvm(Y; grouping=[GroupingTerm(:unit;
mode=:indep)], unit=labels)` jointly fits shared effects for Gaussian and the
four named non-Gaussian families. Formula
routing, interval diagnostics, full fixed-effect `vcov` and structured summaries are described in
[Joint named grouping models](docs/src/grouped-models.md). Full Destination B
qualification remains in progress. The explicit Gaussian `phylo=PrecisionPhy`
route supports precision-only fits and a bounded joint independent-grouping
model; source-specific covariance and full-marginal interval diagnostics remain
separate from frozen-R admission and recovery evidence.
GLLVModels.jl remains an experimental partial R-to-Julia bridge, not 0.7 parity.
Precision-only fitting offers `residual_mode=:shared` alongside the unchanged
trait-specific default; the joint grouping route remains trait-specific.
Eligible independent Gaussian grouping models also have an explicit
`grouped_gaussian_variance_profile(Y, fit; term=:unit, trait=1)` route;
inspect its status and limitations in the grouping guide before using endpoints.
The explicit Julia multivariate precision route is documented in the
[development bridge guide](docs/src/precision-bridge-development.md);
public R `phylo_rr` admission is still closed.

[![Build Status](https://github.com/itchyshin/GLLVModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/itchyshin/GLLVModels.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/itchyshin/GLLVModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/itchyshin/GLLVModels.jl)

Fast Generalised Linear Latent Variable Models (GLLVMs) in Julia, with a broad,
status-tracked GLM response-family surface.

> API may change before v1.0.

`fit_gaussian_sources(...; sigma_eps_fixed=s)` also supports a specified positive
residual SD; omitted means it is estimated. Fixed coordinates are excluded from
`dof`, gradients and Hessians. This does not establish full source-model parity.

Local development candidate: `SourceCovariance` and `fit_gaussian_sources`
represent fixed covariance among source nodes with explicit observation-to-node
projections. Targeted numerical checks and six retained comparisons with R
0.7.0 pass. This evidence is limited to those fixed Gaussian source models;
explicit-source Gaussian formulas support fixed predictors with a complete mean
design. R bridge support, calibrated uncertainty and performance improvements
remain unverified. See the structured-dependence guide for covariance axes
and the nearly singular unique-variance case.

## Why

GLLVMs decompose a multivariate response into a low-rank latent factor
structure plus optional fixed effects, observation-level random effects, and
phylogenetic / spatial random effects. The Gaussian case has a closed-form
marginal:

```
y[t, s] = X[t, s, :]'β + Λ_B η_B[s][t] + ε[t, s]
```

with η_B i.i.d. standard Gaussian, ε i.i.d. Gaussian with variance σ_eps².

`GLLVModels.jl` exploits the closed-form Gaussian marginal:

```
y_s ~ N(X_s β, Λ_B Λ_B' + diag(d_total))
```

— solving directly via SVD (PPCA closed form) when possible, otherwise
warm-starting LBFGS with the PPCA initialisation. Per-iteration cost is
O(p K² + K³) via the Woodbury identity (instead of O(p³) for generic
Cholesky). On our published Gaussian closed-form profile grid that path runs
**median 265.1× faster (range 161–698×)** than the R `gllvmTMB` engine on the
same problem, with answers agreeing to at
least six significant digits (worst case across our benchmark grid:
`|Δ logLik| = 2.3e-07`, `Σ_y` relative Frobenius `4.4e-05`; see
[Benchmarks](https://itchyshin.github.io/GLLVModels.jl/dev/benchmarks)).

That range describes the **Gaussian closed-form path only**, where GLLVModels.jl
uses a closed-form marginal and R uses a TMB Laplace approximation — an
algorithmic difference, not a language one. Non-Gaussian families use a dense
Laplace on both sides and the measured factors are far smaller (Gamma ≈ 1.6×,
zero-truncated Poisson ≈ 2.2×). Do not read the headline as a general claim
about the package; the Benchmarks page opens with the same warning.

Corrected 2026-08-25: this previously read "matched to 1e-7 in log-likelihood
and 1e-5 in Σ_y". Both bounds are exceeded by the package's own published
table — worst case 2.343e-07 and 4.424e-05 respectively. The benchmarks page
was already accurate; the summary here was not.

Corrected 2026-08-26: the speedup claim in the same sentence read "often
10-100× faster". Every cell in the package's own wall-clock table is between
161× and 698×, so no measured cell fell inside the advertised range — it
understated the repo's own data. The 2026-08-25 pass fixed the agreement
bounds and left the neighbouring clause untouched.

## Quick start

```julia
using Pkg
Pkg.add(url = "https://github.com/itchyshin/GLLVModels.jl")
using GLLVModels

# `using GLLVM` cannot resolve after the package rename. After loading
# GLLVModels, `GLLVModels.GLLVM` is a temporary deprecated source-level alias.

# Simulate the per-response-residual Gaussian model used for the R comparison
using Random
Random.seed!(0)
p, K, n = 20, 2, 200
Λ_true = randn(p, K); for i in 1:K, k in 1:K; if i < k; Λ_true[i, k] = 0; end; end
for k in 1:K; Λ_true[k, k] = abs(Λ_true[k, k]) + 0.5; end
ψ_true = 0.15 .+ 0.10 .* rand(p)           # one residual variance per response
y = Λ_true * randn(K, n) .+ sqrt.(ψ_true) .* randn(p, n)  # responses × sites

# Fit the matching diagonal-residual model
fit = fit_gaussian_pervar_gllvm(y; K = K)

# Inspect
fit.pars.Λ                            # estimated loadings
fit.pars.σ_eps                        # observation SD
fit.logLik                            # log-likelihood
fit.cputime                           # wall-clock seconds
```

This is the matrix-first companion to the ordinary R
[`gllvmTMB`](https://itchyshin.github.io/gllvmTMB/) teaching route:
`traits(...) + latent(...)` also implies
`Sigma = Lambda * Lambda' + Psi`, with one diagonal residual variance per
response. Julia stores responses in rows (`p x n`), whereas the R wide table
stores units in rows. The simpler `fit_gaussian_gllvm` route uses one shared
residual SD, so it is a restricted model and is **not** identical to this
per-response-residual teaching fit. The two packages have partial parity, not
a drop-in equivalence; use R for the richer formula-first documentation and
read its current limits before making a same-model or inference claim.

The published Gaussian speed benchmark is deliberately separate: it compares
the shared-residual closed-form special case against a matched R configuration.
Its speed numbers do not establish speed for this per-response route or for
non-Gaussian models.

## Confidence intervals

Three methods, matching the surface of R's `confint()` from the
`gllvmTMB` package:

```julia
GLLVModels.confint(fit)                                    # Wald (default)
GLLVModels.profile_ci(fit, "sigma_eps")                    # profile likelihood
GLLVModels.bootstrap_ci(fit; n_boot = 1000, seed = 42)     # parametric bootstrap
```

## Comparison to MixedModels.jl

> **Loading both in one session breaks six verbs.** GLLVModels.jl and MixedModels.jl each
> export `confint`, `aic`, `bic`, `predict`, `fitted` and `residuals` as unrelated generics,
> so the bare names become ambiguous. Qualify the call (`GLLVModels.confint(...)`) — see
> [Common pitfalls](https://itchyshin.github.io/GLLVModels.jl/dev/pitfalls).

`MixedModels.jl` is the canonical Julia engine for linear mixed models
with sparse random-effect design matrices. `GLLVModels.jl` solves a
*different* model class — reduced-rank latent factors. Use:

| Model | Engine |
|-------|--------|
| `(1 | site)` random intercept, no latent factors | MixedModels.jl |
| GLLVM with K ≥ 1 latent factors | GLLVModels.jl |

## Features

- Closed-form Gaussian marginal log-likelihood (no Laplace approximation)
- One-part GLM response families via a Laplace marginal: Poisson, negative binomial
  (NB2 and NB1, linear variance), Binomial / Bernoulli, beta-binomial
  (overdispersed binomial), Beta, Gamma, Exponential, Ordinal (logit or probit only),
  Tweedie, Student-t (heavy-tailed continuous, fixed finite positive `ν` or estimated
  degrees of freedom with `StudentTFamily()`; outlier-robust
  alternative to Gaussian, `family = StudentTFamily(ν)`), Conway–Maxwell–Poisson
  (under- or over-dispersed counts, `family = COMPoisson()`; marker `ν` is a
  tag payload, always estimated)
- Grouped Tweedie has explicit fixed-common, shared-estimated and per-species
  estimated power controls; these are distinct models, with separate parameter
  counts. Full R0.7.0 Core + AGHQ parity remains under validation.
- Heteroscedastic Gaussian with per-species variance (`fit_gaussian_pervar_gllvm`),
  including explicit full-rank fixed-effect designs profiled by GLS. The development
  `fixed_residual_sd` option separates a supplied residual scale from estimated
  unique variances. `gllvm(...; family=Normal(), pervar=true)` exposes the complete
  formula mean design. AGHQ requests retain exact Gaussian/Laplace with explicit
  fallback provenance; bridge and interval parity remain unverified.
- Per-species / grouped dispersion (`disp.group`) for NB2, NB1, Beta, beta-binomial,
  Gamma, and Tweedie via the `_grouped` drivers — per-species is the `fit_gllvm`
  default for NB2, NB1, Beta, and beta-binomial (`family = NB1()`,
  `family = BetaBinom()`), matching gllvmTMB. Delta-lognormal and Delta-Gamma
  likewise default to per-species dispersion (`disp_group = :species`); shared
  dispersion remains available via `disp_group = :shared`
- Two-part / mixture families: Delta-lognormal, Delta-Gamma, Hurdle-Poisson,
  Hurdle-NB, Beta-hurdle, and ordered-beta via `family = DeltaLogNormal()` /
  `DeltaGamma()` / `HurdlePoisson()` / `HurdleNB()` / `BetaHurdle()` /
  `OrderedBeta()` on `fit_gllvm` (no-X; Delta marker `σ`/`α`, Hurdle-NB `r`,
  Beta-hurdle `φ`, and Ordered-beta `c0`/`c1`/`φ` are tag payloads;
  `HurdlePoisson()` is empty), plus named drivers; ZIP, ZINB, ZIB
  (zero-inflated binomial)
- Variational (VA / ELBO) estimator alongside Laplace, with VA-based SEs
- Ordination trio: unconstrained, concurrent (`num.lv.c`), constrained / RRR (`num.RR`)
- Fixed effects (X β), including fixed-zero coefficient masks for shared
  Gaussian and non-Gaussian covariates; species-specific covariates, fourth-corner
  trait–environment interactions, fixed and random community row effects,
  quadratic response
- Phylogenetic random effects (with user-supplied Σ_phy) — and a phylogenetic
  GLM fit (`fit_phylo_glm`) for non-Gaussian families via an augmented-state
  joint Laplace
- SPDE / Matérn spatial latent field, with kriging prediction
- Offsets, response-missing masks for GLM Laplace rows, Dunn–Smyth residuals, AIC / BIC,
  `predict` / `getLV` / `ordination`, and an `@formula` front-end
  (wide formula tables require one row per site, including intercept-only fits;
  an empty table is allowed when there are no covariates)
- Wald / profile / bootstrap CI routes across scalar-dispersion GLM, grouped
  NB2/NB1/Beta/Gamma, and two-part families; grouped Tweedie, per-trait
  ordinal, and bridge-only edge rows remain status-gated before promotion
- Predictor-informed latent-score effect CIs target `B_lv = Lambda * alpha_lv'`
  for admitted ordinary Gaussian, Poisson, NB2, Binomial, Beta, Gamma, and
  shared-cutpoint Ordinal fits; profile-likelihood calls can be limited to
  selected entries of `vec(B_lv)` with `profile_indices`
- PPCA closed-form initialisation
- Structure-aware Cholesky (Woodbury for Λ Λ' + diag)
- EM-FA solver as an alternative to LBFGS

Poisson, NB2, Binomial, Beta, and Gamma use analytic Laplace outer gradients by
default on plain no-mask/no-offset fits, with finite-difference fallback. The
remaining sparse-Cholesky / CHOLMOD paths stay conservative until their analytic
gradients clear the same runtime accuracy gate; the VA estimator adds analytic
inner and envelope-theorem outer gradients.

Truncated NB2 accepts explicit `disp_group=:species` through `fit_gllvm` and
intercept-only wide/long formulas. Its default remains shared dispersion;
partial grouping and site covariates remain unsupported.

The truncated-Poisson R→Julia bridge rejects fractional, non-finite, or
inexactly representable counts before fitting; it never rounds the response.

## Citation

If you use `GLLVModels.jl` in published work, please cite:

> Nakagawa, S. (2026). GLLVModels.jl: Generalised Linear Latent Variable Models in
> Julia. <https://github.com/itchyshin/GLLVModels.jl>

## License

MIT

### Poisson AGHQ candidate

`fit_poisson_gllvm(Y; K=2, aghq=5)` opts into unpenalized adaptive quadrature
for an ordinary log-link Poisson latent block. Default fits remain Laplace.
Inspect `fit.integration` for actual integration, fallback reasons and retained
attempts. Inference uses the fitted frozen-node objective; full Stage 1a parity,
recovery and coverage remain unverified. See the quickstart for controls and limits.

`fit_binomial_gllvm(Y; K=2, N=trials, aghq=5)` also exposes an ordinary
binomial candidate with logit, probit or cloglog link. Trials, masks and offsets
are retained for inference and postfit methods. `predict` returns probabilities;
`simulate` returns counts. The original five-node binomial comparison currently
fails both-engine convergence and the likelihood gate; do not infer full parity.

`fit_gaussian_gllvm(Y; K=2, aghq=3)` adds the shared-residual-SD Gaussian
candidate. Default fitting remains the exact Gaussian marginal with zero mean;
use `X[p,n,q]` for fixed effects. Recorded masks, offsets and fixed-zero
coefficients carry through prediction and inference. Gaussian AGHQ uses a real
outer quadrature fit, even though the exact marginal is available as a check.
Its interval objective is the fitted frozen-node surrogate. See the quickstart
for an executed example; this does not establish the whole parity programme.
