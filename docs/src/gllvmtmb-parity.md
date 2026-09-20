# Capability parity with gllvmTMB

```@raw html
<div class="gllvm-route gllvm-route--evidence">
  <div>
    <span class="gllvm-route__eyebrow">What works in both packages</span>
    <p>Use this page to see which R workflows have a documented Julia counterpart, and which do not.</p>
  </div>
</div>
```

GLLVModels.jl is a from-scratch Julia twin of R's `gllvmTMB`, built for fitting speed
at moderate-to-large species counts while reproducing point estimates and
likelihoods to **at least six significant digits** on the shared Gaussian +
phylogenetic path (worst case across the benchmark grid:
`|Δ logLik| = 2.343e-07`, `Σ_y` relative Frobenius `4.424e-05`). This

!!! note "Corrected 2026-08-25"
    This sentence previously claimed agreement "to machine precision". Machine
    precision is ~2.2e-16; the measured worst case is 2.343e-07 — roughly nine
    orders of magnitude larger. The Benchmarks page (linked below) always
    reported the honest figure and its comparison limits; this summary did not.

page is a **capability overview** — where GLLVModels.jl stands against the
`gllvmTMB` feature set. For *speed* comparisons see
[Comparison](comparison.md) and [Benchmarks](benchmarks.md).

Legend: ✅ available · 🔨 in progress · ⬜ planned · ⚡ GLLVModels.jl advantage.

## What parity does NOT mean

"Parity" on this page means agreement in small, controlled examples
(`p ≤ 5`, `n ≤ 150`). It does not yet mean that a full workflow will give the
same result end to end. **First-order** comparisons (log-likelihood at each
optimum, cross-objective identity) exist for five paired families: Gaussian, Poisson-log,
Binomial-logit, Beta-logit, and NB2-log. **Second-order** results
(standard errors, the fixed-effect `vcov` block, Wald CI endpoints) exist
only for five small examples, without a stated tolerance. **Realistic-size
examples** (p ≥ 20, n ≥ 500) have not yet been compared. **Interval
*coverage* is not part of parity**. It is a separate Julia-only diagnostic
study. Empirical undercoverage there is a finding, not a calibrated-coverage
certificate and not an R↔Julia
comparison. R's own 0.7.1 interval claim is based on three fixed Wald
examples; the prior total-variance “0.94 coverage floor” wording was
withdrawn.

### Second-order status

**True second-order parity is not established.** A few small examples do not
show that standard errors and intervals agree on realistic data.

**Matched-coordinates comparison: not implemented.** The available comparisons
evaluate each package at its own optimum. A five-example pilot measured
**3 pass / 2 blocked** on five cells — gaussian, poisson, and binomial_logit
pass at R-anchored θ; **beta_logit** and **nb2_log** cannot be compared on the
same coordinates because R uses per-trait dispersion and Julia uses a shared
log-dispersion. Do not read the 3/5 pilot as a completed matched-coordinates
comparison.

The qualification claim is **one-directional**: R workflows against Julia, at
the frozen `gllvmTMB` 0.7.0 reference. At that reference point, 62 R exports
have no Julia counterpart and 91 Julia exports have no R counterpart; three
matches remain ambiguous. These counts describe the comparison, not a promise
to implement every unmatched function here.

### An inventory is not true parity

Completing a feature inventory does not establish true parity. True parity
still needs second-order comparisons, realistic-size cells, real-data
workflows, and grouping-level pairing. Do not infer that R workflows run
identically through Julia from a completed inventory.

### Bridge scope (what `engine = "julia"` is)

One-way **R → Julia** only: a subset of cross-sectional reduced-rank models through
JuliaCall. It admits 11 families, unit-tier
`latent(d=K)` and no-latent paths. **Does not** cover phylo/spatial/animal/kernel/iSDM,
`traits()` formula grammar, mixed-family vectors at full depth, or column_coef / slope
families. Further bridge expansion remains open.

### Explicitly OUT of the parity claim

In plain language, the following remain out of scope:

- Two-directional qualification (Julia→R)
- Full 0.7.1 surface port (column_coef, slopes, formula grid)
- Spatial/slopes engines before phylo transport completes
- Interval *coverage* certification as an R↔Julia comparison
- Other model-development work not required to compare these routes
- Updating the frozen 0.7.0 reference before second-order comparisons land
- fitted/predict/residuals and recovery-to-truth as parity requirements

### Capability differences

Six compared capability rows differ from the R package by design or because a
route remains incomplete: `spatial × dep`, Phylo Model A `lv` intervals,
multinomial depth, broad simulation coverage, public AGHQ, and full
mixed-family vectors. A partial mixed-family point fit through the R bridge is
transport only; it does not make the whole row comparable.

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
| Delta-lognormal | ✅ | first two-part family; shared 2-block Laplace substrate. **Light RCall no-X logLik Δ PAID 2026-08-28** — Δ ≈ 1.5e-8 (rel 1.7e-11) against gllvmTMB 0.7.1, requires `predictor = :shared` + `disp_group = :species` (the twin's parameterisation: one shared η, per-trait σ); `test/parity/test_delta_lognormal_parity.jl` |
| Delta-Gamma | ✅ | occurrence Bernoulli × positive Gamma (log-link mean) on the substrate. **Light RCall no-X logLik Δ PAID 2026-08-28** — Δ ≈ 7.5e-10 (rel 8.3e-13), same two settings; `test/parity/test_delta_gamma_parity.jl` |
| Hurdle (Poisson / NB) | ✅ | occurrence Bernoulli × zero-truncated Poisson / NB2; `fit_gllvm(Y; family = HurdlePoisson())` / `HurdleNB()` (marker `r` is a tag payload). Julia-forward — twin has no hurdle family |
| Zero-inflated (ZIP / ZINB / ZIB) | ✅ | structural zero × Poisson / NB2 / Binomial; zero-inflation intercept-only (Λ_z = 0) so the coupled-zero cross-term drops out |
| Ordered-beta | ✅ | proportions / cover with point masses at 0 and 1; `fit_gllvm(Y; family = OrderedBeta())` (marker `c0`, `c1`, `φ` are tag payloads). Julia-forward — twin has no ordered-beta family |
| Beta-hurdle | ✅ | occurrence Bernoulli × positive Beta; `fit_gllvm(Y; family = BetaHurdle())` (marker `φ` is a tag payload). Julia-forward — twin has no beta-hurdle family |
| Exponential | ✅ | positive continuous, `Var = μ²` (Gamma with shape α=1) |
| Tweedie | ✅ | compound Poisson–Gamma (1<p<2); `fit_tweedie_gllvm`, Dunn–Smyth density series |
| Conway–Maxwell–Poisson | ✅ ⚡ | under- or over-dispersed counts; `fit_gllvm(Y; family = COMPoisson())`; marker `ν` is a tag payload (always estimated). Julia-forward — twin has no CMP family |

## Model structure

| Capability | GLLVModels.jl | Notes |
|-----------|:---:|-------|
| Latent-variable ordination (loadings) | ✅ | any `K`; canonical SVD rotation |
| Fixed-effect covariates (`Xβ`) | ✅ Gaussian · ✅ non-Gaussian (GLM families) | Shared site-X: Poisson/Binomial via `fit_gllvm_cov`; NB2/NB1/Beta/Gamma public/bridge default via `fit_*_gllvm_grouped_cov` (per-trait φ/α + shared `γ`; twin API B; NB1 = `fit_nb1_gllvm_grouped_cov`); Ordinal via `fit_ordinal_gllvm_pertrait_cov` (per-trait cutpoints τ₁=0 / K−2 + shared `γ`; light RCall vs `ordinal_probit`). Shared-dispersion + X remains `fit_gllvm_cov` where that path exists (incl. shared-φ NB1 opt-in). Gaussian `β_fixed` / non-Gaussian `γ_fixed` zero masks supported. |
| Between / within (multilevel) | ✅ Gaussian | `K_W` + per-trait diagonal |
| Phylogenetic random effect | ✅ ⚡ | fast **O(p)** sparse path, benchmarked to p = 10⁴ |
| Animal model (relatedness / GRM) | ✅ Gaussian | `relatedness_cov`, via the `Σ_phy` input |
| Spatial (Matérn / exponential) | ✅ Gaussian | `spatial_cov`, via the `Σ_phy` input |
| Structured dependence × non-Gaussian | ✅ phylo · 🔨 spatial-latent / animal | phylogenetic GLM landed (`fit_phylo_glm`, augmented-state joint Laplace); SPDE / Matérn spatial latent field (`fit_spde_latent_gllvm`) for the non-Gaussian GLLVM |
| Random slopes `(1 + x \| g)` | 🔨 | formula front-end (c) |
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
| `traits()` / `phylo()` formula terms · random slopes `(1+x\|g)` | 🔨 | custom StatsModels terms + RE substrate (design spec'd) |

## Performance — the differentiator

⚡ Large per-fit speedups on the **Gaussian closed-form path**, and an O(p)
phylogenetic gradient benchmarked to p = 10,000.

!!! warning "Read the published grid, not the headline"
    The grid published in these docs ([Benchmarks](benchmarks.md)) measures
    **161.2×, 185.3×, 194.9×, 335.3×, 398.8×, 698.1×** — median **265.1×**.

    A `~340×` figure appears elsewhere in this repo, attributed to a "Gaussian +
    phylogenetic" grid. **That grid is not published here**, so a reader cannot check
    it against anything in this repository, and it does not match the one grid that
    is. Treat `~340×` as unverified in-repo pending publication of its source.

    Agreement is **at least six significant digits**, not machine precision: the
    measured worst case across the published grid is `|Δ logLik| = 2.343e-07` and
    `Σ_y` relative Frobenius `4.424e-05`.

    None of this generalises beyond the Gaussian closed-form path. Measured
    non-Gaussian speedups include zero-truncated Poisson ≈ 2.2× and Gamma ≈ 1.6×. Poisson, NB2, Binomial, and
Beta use analytic Laplace outer gradients by default on plain no-mask/no-offset
fits, with finite-difference fallback; Gamma and the remaining finite-difference
Laplace paths stay conservative until their analytic gradients meet the runtime
accuracy criterion. The sparse-Cholesky / CHOLMOD marginals are not generic-AD-friendly;
the VA estimator adds analytic inner and envelope-theorem outer gradients for
further fit-time gains.

## R bridge: parameterization map

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

Engine-side parity is broader than the current R bridge admission surface. The
current `gllvmTMB(..., engine = "julia")` bridge admits complete, balanced,
one-part reduced-rank models for Gaussian, Poisson, Binomial, NB2, NB1, Beta,
Gamma, and Ordinal-probit no-X fits. For NB2, NB1, Beta, and Gamma, the Julia
bridge default now routes through per-trait grouped-dispersion fitters
(`group = 1:p`) so the point-fit nuisance structure matches native
`gllvmTMB`/`gllvm`; grouped-dispersion Wald/profile/bootstrap CI payloads are
routed through the same no-X bridge contract. Ordinal and
ordinal-probit bridge rows now use per-trait cutpoints by default and return
`cutpoints` as a NaN-padded trait x threshold matrix plus per-trait
`n_categories`, `cutpoint_mode = "per_trait"`, and `cutpoint_link`; per-trait
ordinal CI endpoints remain unavailable-status rows until a per-trait cutpoint
CI engine lands. Fixed-effect
covariates (`X`) are admitted for complete, balanced one-part Gaussian, Poisson,
Binomial, NB2, NB1, Beta, and Gamma fits (NB1 via per-trait
`fit_nb1_gllvm_grouped_cov`; light RCall `nbinom1`+X cell abs Δ ≈1.53e-9 @
rtol 1e-6, seed=48).
`GLLVModels.bridge_capabilities()` exposes the current Julia bridge surface as
a JuliaCall-friendly capability list. It lets the R side verify that each
admitted route has an explicit status, while Julia-only routes remain clearly
marked as unavailable from R.
For Gaussian covariate fits the bridge returns `mean_coef`, the full coefficient
vector for the supplied `X` array, so the R side can reconstruct in-sample
fitted values without guessing from the per-trait mean summary. When the R side
passes a fixed-zero coefficient mask through `options["coef_fixed"]`, the bridge
returns the full coefficient vector with constrained entries equal to zero plus
`mean_coef_status` (Gaussian) or `gamma_status` (non-Gaussian) so the R package
can print fixed rows without treating them as estimated parameters.
Predictor-informed latent-score covariates (`X_lv`) are admitted for
complete-response ordinary Gaussian, Poisson (log link), shared-dispersion NB2,
shared-shape Gamma, shared-precision Beta, binomial logit/probit/cloglog, and
native shared-cutpoint Ordinal logit point fits. The Gaussian bridge centres
responses by trait means and returns those means as `alpha`; the Poisson, NB2,
Gamma, Beta, and binomial bridges keep per-trait link-scale intercepts in
`alpha` (the NB2, Gamma, and Beta `X_lv` routes use the
shared-dispersion/shape/precision fitter, not the per-trait grouped route).
Native shared-cutpoint Ordinal `X_lv` is Julia-side only for now; it does not
promote the per-trait ordinal R bridge. These routes return total latent scores
in `scores` and add `scores_mean`, `scores_innovation`, `alpha_lv`, and
rotation-stable `lv_effects = Lambda * alpha_lv'`. Native GLLVModels.jl can compute
uncertainty for the ordinary `B_lv` product, including selected-entry
profile-likelihood canaries, but the R bridge still transports only the Wald
`X_lv` payload for promoted rows. Response masks, simultaneous fixed-effect `X`,
mixed-family fits, grouped-dispersion `X_lv`, bridge profile/bootstrap `X_lv`
intervals, per-trait ordinal `X_lv`, and two-part `X_lv` remain deliberate
follow-ups rather than inferred parity.
Initial response-missing masks are admitted only for no-X one-part non-Gaussian
bridge fits through an explicit `mask` (`true = observed`); the R bridge
live-tests Poisson, Bernoulli Binomial, NB2, NB1, Beta, Gamma, and
Ordinal-probit routes end to end. Gaussian response masks remain an explicit
follow-up.
Ordinal-probit is fit/nobs/mask/link-tested, and the Julia payload carries
per-trait cutpoints plus category counts so R-side prediction can be gated
explicitly by the paired `gllvmTMB` branch. NB1 post-fit prediction, residual, augmentation,
and conditional simulation are routed for complete-data no-X fits and for masked
fits where the fitted means are available; masked simulation and masked
CI/profile/bootstrap refits remain rejected with explicit CI-status messages.
X+mask fits, ordinal covariate fits, structured covariance terms, and
user-selectable Julia-side optimizer controls remain explicit bridge follow-ups,
not silently supported cells.

The mixed-family R bridge is guarded and intentionally limited: complete
balanced trait-aligned no-X/no-mask/no-CI Julia-engine point fits are admitted
for Gaussian, Poisson, Binomial, NB2, Beta, and Gamma components. The bridge
stores row-aligned per-trait `families` and `link` labels, validates the native
`gllvmTMB` selector used for the comparison, checks direct-wrapper logLik equality, and routes
current in-sample post-fit methods with unavailable-CI status. Mixed-family X,
masks, cbind/weights, REML, ordinal/NB1/two-part components, and CI endpoints
remain rejected deliberately.

REML is a Gaussian-only bridge/engine claim in this project. HSquared's very fast
AI-REML work is useful design input for exact Gaussian variance-component cells,
but it is not terminology to use for non-Gaussian Laplace GLLVMs. Non-Gaussian
speedups should be described as observed-information, Fisher/natural-gradient,
reverse-mode, or implicit-Laplace-adjoint work, each gated by reference-gradient,
point-estimate, and CI/status evidence.

The engine still carries additional gllvm/gllvmTMB parity rows that are not all
public through the R bridge yet:

- **`ZNIB`** (zero-and-N-inflated binomial) — deferred: the gllvm TMB template's
  `case ZNIB` appears to fall through (missing `break;`) into beta-binomial, so its
  likelihood needs upstream confirmation before building to it.
- **corAR1 / corExp / corCS structured row effects, and `lvCor` correlated latent
  variables** — these are `gllvm` features, **not in gllvmTMB**, so they are out of
  scope for this bridge. (GLLVModels.jl does carry more general SPDE/Matérn-spatial and
  phylogenetic substrates, which gllvm/gllvmTMB lack.)
- **Per-trait nuisance-parameter intervals** — grouped NB2/NB1/Beta/Gamma CIs
  are routed; grouped Tweedie and per-trait ordinal-cutpoint CI endpoints remain
  follow-up work.

## Honest gaps

### Laplace curvature: Fisher vs observed

`gllvmTMB` is built on TMB, whose `MakeADFun(..., random = ...)` differentiates
the coded joint negative log-likelihood. Its Laplace log-determinant therefore
uses the **observed** joint Hessian, structurally and without ever making a
choice about it. GLLVModels.jl hand-codes its Laplace kernels, and several of them
used the **Fisher (expected)** information in that role instead.

The two coincide at canonical links — Poisson/log and Binomial/logit, where the
curvature is free of `y` — which is why the launch families were unaffected and
why the discrepancy went unnoticed. They differ everywhere else.

**Status, stated plainly rather than as a capability claim:**

- **Using observed curvature:** NB1 (grouped route), `truncated_nbinom2`,
  `Exponential`, `DeltaGamma`, **`Gamma`** — the one that sat on the public
  default path `fit_gllvm(Y; family = Gamma())` — and, as of 2026-08-28, the
  shared **Tweedie** route (`fit_tweedie_gllvm`) and **`Binomial`/probit**.
  TMB structurally differentiates the joint nll, so its log-det is observed
  for every family it ships, not a per-family exception.
- **Still using the Fisher weight:** only **GP-1**. A minority of tested cells
  fail badly under observed curvature, so GP-1 retains Fisher curvature. A
  log-likelihood from GP-1 will not match `gllvmTMB` to machine precision.
- `Binomial` at the **cloglog** link and the grouped **Tweedie** route both
  default to `:observed`. For cloglog, numerical quadrature agrees with R to
  7.4e-12; the prior `:fisher` default was a Julia-side error. Grouped
  Tweedie reduces exactly to the shared route under the same default when
  `G = 1`.
- **Not a uniform improvement.** Against numerical quadrature, the observed
  curvature is decisively closer for Gamma (12/12 seeds, 20–60× smaller error)
  and for NB2 (87% of 150 curvature-adjudication study cells, 2026-08-27),
  but *not* for Beta, and measurably worse for GP-1's dispersion recovery. So
  each family is decided on its own evidence. Matching TMB is the goal;
  "the numbers get better" would be an overstatement.

Where a curvature has been corrected, the previous behaviour stays reachable
through `hessian = :fisher` on the corresponding marginal.

### Other gaps

The rows above describe engine capabilities and the narrower R bridge admission
surface separately. Engine-side work now covers the major response-family rows,
fixed-effect covariates for the GLM families, the VA estimator, ordination
extractors, SPDE / Matérn spatial latent fields, phylogenetic GLMs, and
confidence-interval machinery. Those are not automatically public
`gllvmTMB(..., engine = "julia")` claims: each bridge row still needs its own
R-side admission, parity test, CI-status handling, and documentation.

The remaining gaps are each scoped by an execution-ready spec in
`docs/superpowers/specs/` (design + slice plan + verifiable goals), so they can
be built *with* validation rather than shipped unverified:

- **Structured dependence × non-Gaussian (animal / spatial extensions)** — the
  phylogenetic GLM has landed (`fit_phylo_glm`, an augmented-state joint Laplace),
  and the SPDE / Matérn spatial latent field is wired into the non-Gaussian GLLVM
  (`fit_spde_latent_gllvm`). The remaining work is the general dense-`S_u`
  species random effect `u ~ N(0, σ²Σ)` shared across sites and the scalable
  large-`p` determinant. Spec:
  `2026-05-31-nongaussian-structured-dependence-design.md`.
- **`@formula` front-end** — **v1 landed**: `gllvm(@formula(y ~ 1 + covariates), Y,
  data; family, K)` for continuous fixed effects over wide data routes to the
  engine (StatsModels + Tables added). Still deferred (design spec'd in
  `2026-05-31-formula-frontend-random-slopes-design.md`): long-format data, the
  `traits()`/`phylo()`/`latent()` custom terms, categorical covariates, and the
  headline random slopes `(1 + x | g)` (which need the new RE engine substrate).
- **R bridge (`engine = "julia"`)** — in progress through the R package bridge.
  Complete-data one-part fits, selected fixed-effect-X rows, selected
  missing-response-mask rows including NB1, scalar-CI transport, and NB1
  post-fit methods are admitted only where live R tests cover them. Mixed-family
  point-fit metadata, grouped-dispersion CI endpoints, NB1-X, masked CIs,
  structured dependence, and broader post-fit methods remain bridge follow-ups.
