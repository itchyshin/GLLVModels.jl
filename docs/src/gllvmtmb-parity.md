# Capability parity with gllvmTMB

```@raw html
<div class="gllvm-route gllvm-route--evidence">
  <div>
    <span class="gllvm-route__eyebrow">What works in both packages</span>
    <p>Use this page to see which R workflows have a documented Julia counterpart, and which do not.</p>
  </div>
</div>
```

Can you fit the same biological model in Julia as in R? Start with the response
family, then check the model structure and the summaries you need. A fitter
available directly in Julia may not be available through
`gllvmTMB(..., engine = "julia")`, and similar function names can describe
different models or parameter scales.

Use the tables below to choose a model, and the R bridge section to check
whether you can call it from R. For help choosing a first analysis, see
[Choose R, Julia, or the bridge](choose-r-julia-bridge.md).

GLLVModels.jl is a Julia companion to R's `gllvmTMB`. On the shared Gaussian
and phylogenetic examples, its point estimates and likelihoods agree to **at
least six significant digits**. The largest measured differences on that grid
are `|Δ logLik| = 2.343e-07` and a relative Frobenius difference of
`4.424e-05` for `Σ_y`; this is close agreement, not machine precision.

This page is a **capability overview**: it shows which `gllvmTMB` workflows
have a documented Julia counterpart and which do not. For *speed* comparisons see
[Comparison](comparison.md) and [Benchmarks](benchmarks.md).

Legend: ✅ available · 🔨 in progress · ⬜ planned · ⚡ GLLVModels.jl advantage.

## How closely do the results agree?

"Parity" on this page means agreement in small, controlled examples
(`p ≤ 5` responses, `n ≤ 150` observations). It does not yet mean that a full
workflow will give the same result end to end. **First-order** comparisons
(log-likelihood at each optimum and agreement when evaluating the same model)
exist for five paired families: Gaussian, Poisson-log,
Binomial-logit, Beta-logit, and NB2-log. **Second-order** results
(standard errors, the fixed-effect `vcov` block, Wald CI endpoints) now cover
about 15 examples, each checked against a numerical tolerance the maintainer
signed off on 2026-09-05. As
with the first-order comparisons, each package is still evaluated at its own
optimum, not at identical parameter values; see "Matched-parameter
comparison" below. **Realistic-size examples** (p ∈ {20, 50}, n ∈ {500,
2000}) have also been checked, for three families (Gaussian, Poisson, and
negative binomial): all 12 example cells passed their tolerance, merged
2026-09-06. This check does not
cover every family, and the receipts themselves say this specifically is not
a completed-parity claim. **Interval
*coverage* is not part of parity**. It is a separate Julia-only diagnostic
study. Empirical undercoverage there is a finding, not a calibrated-coverage
certificate and not an R↔Julia
comparison. R's own 0.7.1 interval claim is based on three fixed Wald
examples; the prior total-variance “0.94 coverage floor” wording was
withdrawn.

### Standard errors and confidence intervals

**True second-order parity is not established.** A broader set of small
examples, plus a twelve-cell check for three families at realistic size,
still falls short of showing that standard errors and intervals agree across
every family and every data shape.

**Matched-parameter comparison is incomplete.** The available comparisons
evaluate each package at its own optimum. In a five-example pilot, Gaussian,
Poisson, and binomial-logit agreed at parameter values estimated in R.
Beta-logit and NB2-log are still not compared at matching parameter values,
but the reason has changed: GLLVModels.jl's bridge now defaults to a
per-trait dispersion route for these two families, the same structure R
uses by default, so the earlier mismatch (R using one dispersion value per
trait while Julia used one shared value) no longer applies as a technical
block. The maintainer instead decided, on 2026-09-15, to keep
matched-parameter comparison permanently out of scope for these two
families' default examples rather than promote it from a test-code fix alone.
This pilot is useful evidence, not a complete matched-parameter comparison.

The comparison starts from **R workflows and checks their Julia counterparts**,
using `gllvmTMB` 0.7.0 as the fixed reference. It does not check every Julia
workflow against R. At that reference point, 62 R exports
have no Julia counterpart and 91 Julia exports have no R counterpart; three
matches remain ambiguous. These counts describe the comparison, not a promise
to implement every unmatched function here.

### A separate frozen-reference check

Apart from the comparisons above, a fixed set of harder example datasets is
also run against one frozen copy of `gllvmTMB` (commit `b4d5fee6`, version
0.7.0) as an early-warning check, not a parity claim: the reference version
never moves, so a failure here can also mean the Julia side changed, rather
than that either package is wrong. Three examples in that fixed set still
fail as written and are treated as known hard cases, sometimes called
"holdouts": a negative-binomial model whose overdispersion estimate drifts
toward the edge where the model is indistinguishable from a plain Poisson
count model, a Student-t model, and a truncated negative-binomial model. The
most recent run of all three was 2026-09-24; this gap remains open. A fix
merged 2026-09-25 restarts negative-binomial fits that stall at that
Poisson-like edge and raised several fits that used to stop early, but the
negative-binomial holdout case is a deliberately hard example and is not
resolved by that fix.

### Matching functions does not establish matching analyses

Completing a feature inventory does not establish true parity. True parity
still needs comparisons of uncertainty, realistic-size examples, real-data
workflows, and grouping-level pairing. Do not infer that R workflows run
identically through Julia from a completed inventory.

### Bridge scope (what `engine = "julia"` is)

The bridge sends a subset of cross-sectional reduced-rank models **from R to
Julia** through JuliaCall. It supports 11 families, with unit-level
`latent(d=K)` terms or no latent factors. It does not cover phylogenetic,
spatial, animal, kernel, or integrated species-distribution (iSDM) structures,
the full `traits()` formula grammar, all mixed-family combinations, or
`column_coef` and slope models. The detailed restrictions appear below.

### What has not been established

In plain language, the following remain out of scope:

- Agreement for every Julia workflow when run through R
- Support for all `gllvmTMB` 0.7.1 functions and formula combinations, including
  `column_coef` and slopes
- Spatial and slope models through the R bridge
- Correct interval *coverage*, meaning that intervals contain the true value
  at the stated rate across repeated datasets
- Agreement for model extensions outside the compared examples
- Agreement with newer R versions across the full comparison set; the fixed
  reference remains 0.7.0, with individual exceptions identified below
- Agreement of fitted values, predictions, and residuals, or recovery of known
  simulated parameters, as part of the comparisons summarized here

### Capability differences

Six areas differ from the R package or remain incomplete: spatial models with
dependent trait effects (`spatial × dep`), latent-variable intervals for the
phylogenetic Model A, multinomial models, broad simulation validation,
adaptive Gauss–Hermite quadrature (AGHQ), and mixed-family models. Obtaining
point estimates for a supported mixed-family model through the bridge does
not establish agreement for its uncertainty or for other family combinations.

## Response families

| Family | GLLVModels.jl | Notes |
|--------|:---:|-------|
| Gaussian | ✅ | closed-form marginal |
| Binomial (Bernoulli / counts) | ✅ | logit / probit / cloglog |
| Beta-binomial | ✅ | overdispersed binomial; `fit_beta_binomial_gllvm`; `BetaBinomial(N, μφ, (1−μ)φ)`, precision `φ` matches gllvm family 15; `φ→∞ ⇒ Binomial` |
| Poisson | ✅ | log link |
| Negative binomial (NB2) | ✅ | size `r` jointly estimated; `Var = μ + μ²/r`. **gllvm uses dispersion `φ = 1/r`** (`Var = μ + μ²φ`) — see the bridge map below |
| Negative binomial (NB1) | ✅ | linear variance `Var = μ(1+φ)`; matches gllvm `negative.binomial1` (same `φ`) |
| Beta | ✅ | precision `φ` (matches gllvm) |
| Ordinal (cumulative) | ✅ | logit + probit links (`link=ProbitLink()` matches gllvm's default cumulative-probit); `P(y≤c)=F(τ_c−η)` convention verified == gllvm; `fit_ordinal_gllvm()` keeps the shared-cutpoint Julia route, while `fit_ordinal_gllvm_pertrait()` and the R bridge use trait-specific cutpoints for native `gllvmTMB` parity |
| Gamma | ✅ | shape `α` |
| Delta-lognormal | ✅ | first two-part family. In an intercept-only R–Julia comparison with gllvmTMB 0.7.1, the log-likelihood difference was about `1.5e-8` (relative difference `1.7e-11`) when `predictor = :shared` and `disp_group = :species`: one shared linear predictor and one residual scale per trait. |
| Delta-Gamma | ✅ | occurrence Bernoulli × positive Gamma (log-link mean). In the corresponding intercept-only comparison, the log-likelihood difference was about `7.5e-10` (relative difference `8.3e-13`) under the same settings. |
| Hurdle (Poisson / NB) | ✅ | occurrence Bernoulli × zero-truncated Poisson / NB2; `fit_gllvm(Y; family = HurdlePoisson())` / `HurdleNB()`; the marker's `r` value does not fix the fitted dispersion. Available in Julia; no matching family in the reference R package |
| Zero-inflated (ZIP / ZINB / ZIB) | ✅ | structural zero × Poisson / NB2 / Binomial; zero-inflation intercept-only (Λ_z = 0) so the coupled-zero cross-term drops out |
| Ordered-beta | ✅ | proportions / cover with point masses at 0 and 1; `fit_gllvm(Y; family = OrderedBeta())`; the marker's `c0`, `c1`, and `φ` values do not fix the fitted parameters. Available in Julia; no matching family in the reference R package |
| Beta-hurdle | ✅ | occurrence Bernoulli × positive Beta; `fit_gllvm(Y; family = BetaHurdle())`; the marker's `φ` value does not fix the fitted precision. Available in Julia; no matching family in the reference R package |
| Exponential | ✅ | positive continuous, `Var = μ²` (Gamma with shape α=1) |
| Tweedie | ✅ | compound Poisson–Gamma (1<p<2); `fit_tweedie_gllvm`, Dunn–Smyth density series |
| Conway–Maxwell–Poisson | ✅ ⚡ | under- or over-dispersed counts; `fit_gllvm(Y; family = COMPoisson())`; `ν` is always estimated rather than fixed by the marker's value. Available in Julia; no matching family in the reference R package |

## Model structure

| Capability | GLLVModels.jl | Notes |
|-----------|:---:|-------|
| Latent-variable ordination (loadings) | ✅ | any `K`; canonical SVD rotation |
| Fixed-effect covariates (`Xβ`) | ✅ Gaussian · ✅ non-Gaussian (GLM families) | Shared site-X: Poisson/Binomial via `fit_gllvm_cov`; NB2/NB1/Beta/Gamma public/bridge default via `fit_*_gllvm_grouped_cov` (per-trait φ/α + shared `γ`; NB1 = `fit_nb1_gllvm_grouped_cov`); Ordinal via `fit_ordinal_gllvm_pertrait_cov` (per-trait cutpoints τ₁=0 / K−2 + shared `γ`; checked against R's `ordinal_probit`). Shared-dispersion + X remains `fit_gllvm_cov` where that path exists (incl. shared-φ NB1 opt-in). Gaussian `β_fixed` / non-Gaussian `γ_fixed` zero masks supported. |
| Between / within (multilevel) | ✅ Gaussian | `K_W` + per-trait diagonal |
| Phylogenetic random effect | ✅ ⚡ | fast **O(p)** sparse path, benchmarked to p = 10⁴ |
| Animal model (relatedness / GRM) | ✅ Gaussian | `relatedness_cov`, via the `Σ_phy` input |
| Spatial (Matérn / exponential) | ✅ Gaussian | `spatial_cov`, via the `Σ_phy` input |
| Structured dependence × non-Gaussian | ✅ phylo · 🔨 spatial-latent / animal | phylogenetic GLM (`fit_phylo_glm`, augmented-state joint Laplace); SPDE / Matérn spatial latent field (`fit_spde_latent_gllvm`) for the non-Gaussian GLLVM; broader spatial and animal-model support remains incomplete |
| Random slopes `(1 + x \| g)` | 🔨 | this formula syntax is not yet supported |
| Per-species / grouped dispersion (`disp.group`) | ✅ all 5 dispersion families | `fit_{nb,beta,gamma,nb1,tweedie}_gllvm_grouped(Y; K, group)` give each species (or group) its own dispersion; reduces exactly to the shared fit at `G=1`. **gllvm's default is per-species** dispersion, so for parity route Julia through a grouped fitter with `group = 1:p` (or set gllvm `disp.formula = ~1` for the shared model) |
| Row effects (fixed **and random**) | ✅ | fixed per-site intercepts (`fit_roweffect_gllvm`) **and** random `ρ_s ~ N(0, σ_row²)` (`fit_row_random_gllvm`, gllvmTMB `row.eff="random"`); `σ_row→0` reduces exactly to no-row-effect |

## Post-fit & inference

| Capability | GLLVModels.jl | Notes |
|-----------|:---:|-------|
| `getLV` / `getLoadings` / `rotation` | ✅ | all families |
| `predict` / `fitted` | ✅ | all families (ordinal adds `:prob` / `:class`) |
| `residuals` (Dunn–Smyth + Pearson) | ✅ | all families |
| `simulate` (parametric draw from a fit) | ✅ selected non-Gaussian | `simulate(fit, n)` for scalar-mean GLM-style fits and Tweedie; `simulate(fit, X)` for covariate fits. Two-part bootstrap CIs use internal samplers, but public `simulate` methods are not universal. |
| `aic` / `bic` / `show` | ✅ | all families |
| Σ_y / communality / correlation / phylo signal H² | ✅ Gaussian | report-ready extractors |
| Confidence intervals (Wald / profile / bootstrap) | ✅ scalar/grouped dispersion · 🔨 per-trait ordinal | Gaussian, scalar-dispersion GLM families, grouped-dispersion NB2/NB1/Beta/Gamma, the two-part families, and shared-cutpoint ordinal via `confint(fit, Y; method=…)`; per-trait ordinal-cutpoint CI endpoints are follow-ups; bootstrap is thread-parallel |
| Ordination biplot | ✅ | |

## Interface

| Capability | GLLVModels.jl | Notes |
|-----------|:---:|-------|
| Matrix-level fit API | ✅ | `fit_gllvm(Y; family, K, …)` |
| `@formula` front-end | ✅ fixed effects (wide + long) · 🔨 rest | `gllvm(@formula(y ~ 1 + x), Y, data; …)` and `gllvm(@formula(y ~ 1 + x), long; species, site, …)`; random slopes, `traits()`/`phylo()`, categoricals deferred |
| `traits()` / `phylo()` formula terms · random slopes `(1+x\|g)` | 🔨 | these formula terms are not yet supported |

## How much faster are the measured fits?

⚡ Large per-fit speedups on the **Gaussian closed-form path**, and an O(p)
phylogenetic gradient benchmarked to p = 10,000.

!!! warning "Read the published grid, not the headline"
    The grid published in these docs ([Benchmarks](benchmarks.md)) measures
    **161.2×, 185.3×, 194.9×, 335.3×, 398.8×, 698.1×** — median **265.1×**.

    A `~340×` figure elsewhere is attributed to a different Gaussian and
    phylogenetic grid whose results are not published here. That figure remains
    unverified; use the six published measurements above.

    Agreement is **at least six significant digits**, not machine precision: the
    measured worst case across the published grid is `|Δ logLik| = 2.343e-07` and
    `Σ_y` relative Frobenius `4.424e-05`.

    None of this generalises beyond the Gaussian closed-form path. Measured
    non-Gaussian speedups include zero-truncated Poisson ≈ 2.2× and Gamma ≈ 1.6×.

Poisson, NB2, Binomial, and Beta use analytic Laplace outer gradients by default
on plain no-mask/no-offset
fits, with finite-difference fallback; Gamma and the remaining finite-difference
Laplace paths continue to use finite differences. Sparse-Cholesky / CHOLMOD
calculations cannot generally use automatic differentiation directly. The
variational approximation (VA) estimator uses analytic inner and
envelope-theorem outer gradients; its timings depend on the model and should
not be inferred from the Gaussian benchmark.

## R bridge: match the model and parameter scales

R `gllvmTMB` can call GLLVModels.jl as its default Julia fitting path through the
R-side bridge. For results to agree, the bridge must reconcile a few
**convention differences** — the underlying models are the same, but the
parameter scales/structures differ. These are translation rules for the bridge,
not bugs on either side.

| Quantity | gllvm (R) | GLLVModels.jl | Bridge rule |
|----------|-----------|----------|-------------|
| NB2 dispersion | `φ` (dispersion), `Var = μ + μ²φ`; larger `φ` ⇒ more overdispersion | `r` (size), `Var = μ + μ²/r` | **`r = 1/φ`** (invert in both directions). Also propagates to ZINB / Hurdle-NB / grouped-NB |
| NB1 dispersion | `φ`, `Var = μ + μφ` | `φ`, `Var = μ(1+φ)` | identity (maps 1:1) |
| Gamma dispersion | `φ` = **shape**, `Var = μ²/φ` | `α` = **shape**, `Var = μ²/α` | relabel `α ↔ φ` (no inversion) |
| Beta precision | `φ`, `Var = μ(1-μ)/(1+φ)` | `φ` (same) | identity |
| Tweedie | power `ν`, `Var = φ·μ^ν`, default start `ν = 1.1` | power `p`, `Var = φ·μ^p`, default start `p_init = 1.5` | identity; set `p_init = 1.1` to reproduce gllvm's optimiser path |
| Gaussian dispersion | per-species SD `φ_j` | single shared `σ` (profiled) | needs a per-species-variance Gaussian fit for exact parity |
| Dispersion **structure** | per-species by default (`disp.formula = NULL`) | shared scalar by default; per-species via the grouped fitters | route Julia through `fit_*_gllvm_grouped(Y; K, group = 1:p)`, **or** set gllvm `disp.formula = ~1` |
| Estimation method | default `method = "VA"` | default Laplace; VA available via `fit_*_gllvm_va` | pin matching methods; VA and LA differ in finite samples |

### Families, dispersion, and measured covariates

More models can be fitted directly in Julia than through the R bridge. The
current `gllvmTMB(..., engine = "julia")` bridge supports complete, balanced,
one-part reduced-rank models for Gaussian, Poisson, Binomial, NB2, NB1, Beta,
Gamma, and Ordinal-probit no-X fits. For NB2, NB1, Beta, and Gamma, the Julia
bridge default now routes through per-trait grouped-dispersion fitters
(`group = 1:p`) so the dispersion structure matches native
`gllvmTMB`/`gllvm`; grouped-dispersion Wald/profile/bootstrap intervals are
available through the same bridge for models without covariates. Ordinal and
ordinal-probit bridge models use per-trait cutpoints by default and return
`cutpoints` as a NaN-padded trait x threshold matrix plus per-trait
`n_categories`, `cutpoint_mode = "per_trait"`, and `cutpoint_link`; per-trait
ordinal CI endpoints remain unavailable. Fixed-effect
covariates (`X`) are supported for complete, balanced one-part Gaussian, Poisson,
Binomial, NB2, NB1, Beta, and Gamma fits (NB1 via per-trait
`fit_nb1_gllvm_grouped_cov`; the `nbinom1` comparison with covariates had
absolute log-likelihood difference ≈1.53e-9 at relative tolerance 1e-6, seed=48).
`GLLVModels.bridge_capabilities()` lists which Julia models the bridge
supports, with an explicit status for each. Models available only in Julia
are marked as unavailable from R.
For Gaussian covariate fits the bridge returns `mean_coef`, the full coefficient
vector for the supplied `X` array, so the R side can reconstruct in-sample
fitted values without guessing from the per-trait mean summary. When the R side
passes a fixed-zero coefficient mask through `options["coef_fixed"]`, the bridge
returns the full coefficient vector with constrained entries equal to zero plus
`mean_coef_status` (Gaussian) or `gamma_status` (non-Gaussian) so the R package
can print fixed rows without treating them as estimated parameters.

### Predictors of latent scores

Predictor-informed latent-score covariates (`X_lv`) are supported for
complete-response ordinary Gaussian, Poisson (log link), shared-dispersion NB2,
shared-shape Gamma, shared-precision Beta, binomial logit/probit/cloglog, and
native shared-cutpoint Ordinal logit point fits. The Gaussian bridge centres
responses by trait means and returns those means as `alpha`; the Poisson, NB2,
Gamma, Beta, and binomial bridges keep per-trait link-scale intercepts in
`alpha` (the NB2, Gamma, and Beta `X_lv` routes use the
shared-dispersion/shape/precision fitter, not the per-trait grouped route).
Native shared-cutpoint Ordinal `X_lv` is available only in Julia; the per-trait
ordinal R bridge does not support this extension. These models return total
latent scores
in `scores` and add `scores_mean`, `scores_innovation`, `alpha_lv`, and
rotation-stable `lv_effects = Lambda * alpha_lv'`. Native GLLVModels.jl can compute
uncertainty for the ordinary `B_lv` product, including profile intervals for
selected entries. The R bridge currently returns Wald intervals for its
documented `X_lv` routes. Response masks, simultaneous fixed-effect `X`,
mixed-family fits, grouped-dispersion `X_lv`, profile or bootstrap `X_lv`
intervals through R, per-trait ordinal `X_lv`, and two-part `X_lv` are not yet
available.

### Missing responses and predictions

Missing responses are supported only for one-part non-Gaussian bridge fits
without covariates, through an explicit `mask` (`true = observed`). Complete
R-to-Julia calls have been tested for Poisson, Bernoulli Binomial, NB2, NB1,
Beta, Gamma, and Ordinal-probit. Gaussian response masks remain unsupported
through the bridge.

Ordinal-probit comparisons check fitting, observation counts, masks, and links.
Julia returns per-trait cutpoints and category counts, but the R implementation
must also support prediction from those values before an ordinal prediction
method can use them. NB1 post-fit prediction, residual, augmentation,
and conditional simulation are routed for complete-data no-X fits and for masked
fits where the fitted means are available; masked simulation and masked
CI/profile/bootstrap refits remain rejected with explicit CI-status messages.
Combining `X` with a mask, ordinal covariate fits, structured covariance terms,
and user-selectable Julia optimizer controls remain unsupported through the bridge.

### Different response families in one model

The mixed-family R bridge supports point estimates for complete, balanced,
trait-aligned data without covariates or missing responses. Its components
can be Gaussian, Poisson, Binomial, NB2, Beta, or Gamma. The bridge stores
per-trait `families` and `link` labels in response-row order, checks the selected
R model and agreement of direct and wrapped log-likelihoods, and supports
the current in-sample post-fit methods. Confidence intervals are unavailable.
Mixed-family X, masks, cbind/weights, REML, ordinal/NB1/two-part components, and CI endpoints
remain unsupported.

REML is currently available only for Gaussian models. Non-Gaussian models use
the documented Laplace likelihood approximation; they should not be described
as REML fits.

### Further differences from R

Check these restrictions when translating an R analysis:

- **`ZNIB`** (zero-and-N-inflated binomial) — unsupported. The reference R
  implementation appears to evaluate the beta-binomial likelihood instead;
  the intended likelihood needs confirmation from its developers before a
  Julia counterpart can be compared with it.
- **corAR1 / corExp / corCS structured row effects, and `lvCor` correlated latent
  variables** — these are `gllvm` features, **not in gllvmTMB**, so they are out of
  scope for this bridge. (GLLVModels.jl does carry more general SPDE/Matérn-spatial and
  phylogenetic implementations, which gllvm/gllvmTMB lack.)
- **Per-trait nuisance-parameter intervals** — grouped NB2/NB1/Beta/Gamma CIs
  are supported; grouped Tweedie and per-trait ordinal-cutpoint CI endpoints
  remain unavailable.

## Why likelihood approximations can differ

### Laplace curvature: Fisher vs observed

`gllvmTMB` is built on TMB, whose `MakeADFun(..., random = ...)` differentiates
the coded joint negative log-likelihood. Its Laplace log-determinant therefore
uses the **observed** joint Hessian: the curvature of that likelihood at the
fitted values. GLLVModels.jl implements its Laplace calculations directly, and
several of them used the **Fisher (expected)** information in that role instead.

The two coincide at canonical links — Poisson/log and Binomial/logit, where the
curvature is free of `y`. For other families and links, the two can differ.

The documented choices and comparisons are:

- **Using observed curvature:** NB1 (grouped route), `truncated_nbinom2`,
  `Exponential`, `DeltaGamma`, **`Gamma`** through
  `fit_gllvm(Y; family = Gamma())` — and, as of 2026-08-28, the
  shared **Tweedie** route (`fit_tweedie_gllvm`) and **`Binomial`/probit**.
  TMB obtains observed curvature by differentiating the joint negative
  log-likelihood for each family.
- **Still using the Fisher weight:** only **GP-1**. A minority of tested examples
  fail badly under observed curvature, so GP-1 retains Fisher curvature. A
  log-likelihood from GP-1 will not match `gllvmTMB` to machine precision.
- `Binomial` at the **cloglog** link and the grouped **Tweedie** route both
  default to `:observed`. For cloglog, numerical quadrature agrees with R to
  7.4e-12; the prior `:fisher` default was a Julia-side error. Grouped
  Tweedie reduces exactly to the shared route under the same default when
  `G = 1`.
- **Not a uniform improvement.** Against numerical quadrature, the observed
  curvature is decisively closer for Gamma (12/12 seeds, 20–60× smaller error)
  and for NB2 (87% of 150 examples in the curvature comparison, 2026-08-27),
  but *not* for Beta, and measurably worse for GP-1's dispersion recovery. So
  evidence for one family does not establish better accuracy for another.

Where a curvature has been corrected, the previous behaviour stays reachable
through `hessian = :fisher` on the corresponding marginal.

## Choosing your next step

For a native Julia analysis, choose the response family and structure in the
tables above, then read the corresponding [response-family guide](response-families.md)
or [structured-dependence guide](structured-dependence.md). For an analysis
called from R, also check the bridge restrictions above: a native Julia
fitter does not automatically become available through `engine = "julia"`.

The principal remaining extensions are:

- **Structured dependence with non-Gaussian responses** — phylogenetic and
  spatial latent-field models are available, but a general dense species
  covariance `u ~ N(0, σ²Σ)` shared across sites and a scalable determinant for
  very many responses are still missing.
- **Formula interface** — `gllvm(@formula(y ~ 1 + covariates), Y, data; family, K)`
  accepts continuous fixed effects, with the wide and long interfaces listed
  above. Custom `traits()`/`phylo()`/`latent()` terms, categorical covariates,
  and random slopes `(1 + x | g)` are not yet available.
- **R bridge (`engine = "julia"`)** — the bridge supports documented
  models and summaries described above. Remaining extensions include
  covariates together with missing responses, mixed-family covariates and
  intervals, masked confidence intervals, structured dependence, and broader
  post-fit methods. Use the family-specific descriptions above to distinguish
  available intervals from those still unsupported.
