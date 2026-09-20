# Response families

A GLLVM links its latent factors to the responses through a **response family**
and a **link**. GLLVModels.jl follows the Julia convention (as in GLM.jl): the family
is a `Distributions.jl` distribution, chosen with the `family =` keyword to
`fit_gllvm`.

## The unified entry point

```julia
using GLLVModels, Distributions

# Gaussian responses (continuous) — exact closed-form marginal
fit_gllvm(Y;  family = Normal(),   K = 2)

# Binary / binomial responses — Laplace marginal
fit_gllvm(Yb; family = Binomial(), K = 2, link = LogitLink())

# Count data — Laplace marginal
fit_gllvm(Yc; family = Poisson(), K = 2)

# Overdispersed counts — per-species r by default (NBGroupedFit)
fit_gllvm(Yc; family = NegativeBinomial(), K = 2)

# Proportions in (0,1) — per-species φ by default (BetaGroupedFit)
fit_gllvm(Yp; family = Beta(), K = 2)

# Ordered categories — Laplace marginal
fit_gllvm(Yo; family = Ordinal(), K = 2)

# Heavy-tailed continuous — outlier-robust alternative to Normal()
fit_gllvm(Y;  family = StudentTFamily(4.0), K = 2)

# two-part continuous (occurrence × positive continuous)
fit_gllvm(Yd; family = DeltaLogNormal(), K = 2)
fit_gllvm(Yd; family = DeltaGamma(), K = 2)
```

`fit_gllvm` dispatches on the family: `Normal()` uses the exact closed-form
Gaussian marginal; all non-Gaussian families use a Laplace approximation,
because the latent integral is non-conjugate for non-Gaussian families.

## Links

For binomial responses you can choose the link:

| Link | `linkinv(η)` | Use |
|------|--------------|-----|
| `LogitLink()` *(default)* | logistic | log-odds; the canonical binary link |
| `ProbitLink()` | `Φ(η)` | latent-Gaussian threshold models |
| `CLogLogLink()` | `1 − exp(−eη)` | asymmetric; rare-event / occupancy |

```julia
fit_gllvm(Yb; family = Binomial(), K = 2, link = ProbitLink())
```

For `Poisson`, `NegativeBinomial`, and `Gamma` the default and only supported
link is `LogLink()`. For `StudentTFamily` it is `IdentityLink()` (the response is
a location on the real line). For `Beta` the default is `LogitLink()`. `Ordinal` defaults
to a cumulative `LogitLink()` and also supports `ProbitLink()`. Beta-binomial
supports `LogitLink()` (default), `ProbitLink()`, and `CLogLogLink()`.

## Supported families

| Family | Status | Link | Marginal | Extra parameter | Notes |
|--------|--------|------|----------|-----------------|-------|
| `Normal()` | ✅ available | identity | closed form | — | continuous; the original engine |
| `Binomial()` | ✅ available | logit / probit / cloglog | Laplace | — | binary (Bernoulli) and binomial counts |
| `Poisson()` | ✅ available | log | Laplace | — | counts |
| `TruncatedPoisson()` | ✅ available | log | Laplace | — | zero-truncated counts (`y ≥ 1`); the log link applies to the *untruncated* μ; `fit_gllvm` / named `fit_truncated_poisson_gllvm` |
| `CensoredPoisson()` | ✅ available | log | Laplace | — | right-censored counts; pass `censored::BitMatrix` (or `lower`/`upper`); `fit_gllvm` / named `fit_censored_poisson_gllvm` |
| `NegativeBinomial()` | ✅ available | log | Laplace | dispersion `r` (Var = μ + μ²/r) | overdispersed counts; `r` jointly estimated |
| `NB1()` | ✅ available | log | Laplace | dispersion `φ` (Var = μ(1+φ)) | linear-variance (quasi-Poisson-like) overdispersed counts; `fit_gllvm` default is per-species `φ`; shared `φ` via `fit_nb1_gllvm` |
| `TruncatedNegBin2()` | ✅ available | log | Laplace | dispersion `r` (Var = μ + μ²/r) | zero-truncated NB2 (`y ≥ 1`); shared `r`, or per-trait `r_t` via `fit_truncated_nbinom2_gllvm_pertrait` (twin `log_phi_truncnb2`) |
| `Beta()` | ✅ available | logit | Laplace | precision `φ` (Var = μ(1−μ)/(1+φ)) | proportions in (0,1); `φ` jointly estimated |
| `Ordinal()` | ✅ available | cumulative logit / probit | Laplace | cutpoints `τ` | ordered categories `1:C`; `fit_ordinal_gllvm()` uses shared cutpoints, `fit_ordinal_gllvm_pertrait()` uses trait-specific cutpoints for R-bridge parity |
| `GLLVModels.Multinomial()` | ✅ available | baseline-category softmax (`η₁ ≡ 0`) | fixed effects only — **no latent variables** | — | one *unordered* categorical trait per fit (`1×n` row, length-`n` vector, or `K×n` one-hot); categories `1:K` with `K ≥ 3`; qualify the marker — the bare name clashes with `Distributions.Multinomial` |
| `Gamma()` | ✅ available | log | Laplace | shape `α` (Var = μ²/α) | positive continuous; `α` jointly estimated |
| `Lognormal()` | ✅ available | log | **closed form** (Gaussian on `log y`) | log-SD `σ` | one-part positive continuous, `log y ~ Normal(η, σ²)`; reuses the closed-form Gaussian fitter on `log(Y)`, not a Laplace approximation; `loglik` is reported on the *y* scale; distinct from the two-part `DeltaLogNormal()` |
| `Exponential()` | ✅ available | log | Laplace | — | positive continuous, `Var = μ²` (Gamma with α=1) |
| `StudentTFamily(ν)` | ✅ available | identity | Laplace | scale `σ`; FIXED df `ν` on the marker | heavy-tailed continuous, `(y − η)/σ ~ t_ν`; outlier-robust alternative to `Normal()`; `σ` estimated, `ν` held fixed; → `Normal(η, σ²)` as `ν → ∞` |
| Tweedie | ✅ available | log | Laplace | dispersion `φ`, power `p` (1<p<2) | compound Poisson–Gamma; biomass / abundance with true zeros; `fit_tweedie_gllvm` |
| `COMPoisson()` | ✅ available | log | Laplace | dispersion exponent `ν` (tag payload) | counts with under- or over-dispersion (`ν>1` / `ν<1`; `ν=1 ⇒ Poisson`); `fit_gllvm` / named `fit_compoisson_gllvm`; Julia-forward |
| `GeneralizedPoisson1(α)` | ✅ available | log | Laplace | signed dispersion `α` (tag payload) | counts with over- **or** under-dispersion (`α>0` / `α<0`); `fit_gllvm` / named `fit_gp1_gllvm`; Julia-forward (no twin counterpart) |
| `OrderedBeta()` | ✅ available | logit | Laplace | precision `φ`, cutpoints `c₀<c₁` (tag payloads) | proportions / cover with point masses at 0 and 1; `fit_gllvm` / named `fit_ordered_beta_gllvm`; Julia-forward |
| `DeltaLogNormal()` | ✅ available | logit × identity(log) | two-part Laplace | log-SD `σ` (tag payload) | occurrence × positive lognormal; `fit_gllvm` / named `fit_delta_lognormal_gllvm` |
| `DeltaGamma()` | ✅ available | logit × log | two-part Laplace | shape `α` (tag payload) | occurrence × positive Gamma; `fit_gllvm` / named `fit_delta_gamma_gllvm` |
| `BetaHurdle()` | ✅ available | logit × logit | two-part Laplace | precision `φ` (tag payload) | occurrence × positive Beta; `fit_gllvm` / named `fit_beta_hurdle_gllvm`; Julia-forward |
| `HurdlePoisson()` | ✅ available | logit × log | two-part Laplace | — | occurrence × zero-truncated Poisson; `fit_gllvm` / named `fit_hurdle_poisson_gllvm` |
| `HurdleNB()` | ✅ available | logit × log | two-part Laplace | dispersion `r` (tag payload) | occurrence × zero-truncated NB2; `fit_gllvm` / named `fit_hurdle_nb_gllvm`; Julia-forward |
| ZIP | ✅ available | logit × log | two-part Laplace | — | zero-inflated Poisson; `fit_zip_gllvm` / shared site-X via `fit_zip_gllvm_cov` (separate `γz`/`γc`, `Λz=0`; Julia-forward) |
| ZINB | ✅ available | logit × log | two-part Laplace | shared scalar `r` | zero-inflated NB2; `fit_zinb_gllvm` / shared site-X via `fit_zinb_gllvm_cov` (separate `γz`/`γc`, `Λz=0`, shared `r`; Julia-forward) |
| ZIB | ✅ available | logit × logit | two-part Laplace | — | zero-inflated Binomial; `fit_zib_gllvm` |
| `BetaBinom()` | ✅ available | logit / probit / cloglog | Laplace | precision `φ` (`a = μφ, b = (1−μ)φ`) | overdispersed binomial counts; `fit_gllvm` default is per-species `φ` and requires the p×n `N`; shared `φ` via `fit_beta_binomial_gllvm`; → Binomial as `φ → ∞` |

The single-block families with a plain `Distributions` marker — `Normal`,
`Binomial`, `Poisson`, `NegativeBinomial` (NB2), `Beta`, `Ordinal`, `Gamma`,
`Exponential` — are reached through the unified `fit_gllvm` entry, as are `NB1`,
`BetaBinom`, `StudentTFamily`, `DeltaLogNormal`, `DeltaGamma`,
`HurdlePoisson`, `HurdleNB`, `BetaHurdle`, `COMPoisson`, and `OrderedBeta`
via the package's own exported markers.
Tweedie currently has a dedicated `fit_tweedie_gllvm` driver (the power is
not yet a public `fit_gllvm` marker).
A **mixed-family response vector** — a different family per trait in one fit —
is available through the dedicated `fit_mixed_gllvm` driver rather than a
`family =` marker; see its section below. Calling `fit_gllvm` with an unimplemented
family raises a clear error listing what is currently available.

**Phylogenetic GLM.** For a per-species phylogenetic random intercept under a
non-Gaussian family, `fit_phylo_glm(Y, phy; family = …)` fits the augmented-state
joint Laplace marginal (Poisson / NB / Binomial, with a dispersion parameter for
the dispersion families) over the sparse phylogenetic precision.

## Family details

### Gaussian — `Normal()`

```julia
fit = fit_gllvm(Y; family = Normal(), K = 2)
```

The Gaussian GLLVM admits a **closed-form marginal** (no Laplace approximation).
The latent integral is conjugate, so the optimiser works directly on the exact
log-likelihood. This is the fastest and most accurate path. The response matrix
`Y` is `p × n` (responses × sites).

### Binomial — `Binomial()`

```julia
fit = fit_gllvm(Yb; family = Binomial(), K = 2)                    # Bernoulli
fit = fit_gllvm(Yb; family = Binomial(), K = 2, N = trials)        # binomial counts
fit = fit_gllvm(Yb; family = Binomial(), K = 2, link = ProbitLink())
```

For binary responses (Bernoulli), `Y` is a `p × n` integer matrix of 0/1.
For binomial *counts*, pass the trial counts as `N` — a `p × n` integer matrix;
the default is all-ones (Bernoulli). Link choices: `LogitLink()` (default),
`ProbitLink()`, `CLogLogLink()`.

### Poisson — `Poisson()`

```julia
fit = fit_gllvm(Yc; family = Poisson(), K = 2)
```

For count data (`Y` a `p × n` integer matrix). Uses a log link and a Laplace
marginal. Poisson GLLVMs are a natural starting point for species-abundance
matrices before considering overdispersion.

### Zero-truncated Poisson — `TruncatedPoisson()`


```julia
fit = fit_gllvm(Yc; family = TruncatedPoisson(), K = 2)   # counts; y ≥ 1; log link on untruncated μ
fit = fit_truncated_poisson_gllvm(Yc; K = 2)              # same driver, called directly
```

For counts with **no observable zeros** — e.g. counts recorded only where a
species was detected at all. The log link applies to the *untruncated*
Poisson mean `μ = exp(η)`; the per-cell density is `Poisson(y; μ) / (1 −
e^{−μ})` for `y ≥ 1`. Every observed cell of `Yc` must be `≥ 1`:
`fit_truncated_poisson_gllvm` (and `fit_gllvm`, which delegates to it) scans
the observed cells first and throws an `ArgumentError` —
`"truncated_poisson requires y ≥ 1; …"` — on any cell below 1, so zeros and
negatives both fail loud rather than being silently dropped. Masked cells are
skipped by that scan. Only `LogLink()` is supported, and the guard is
explicit:

```julia
link isa LogLink || throw(ArgumentError(
    "fit_truncated_poisson_gllvm: only LogLink is supported (twin truncated_poisson)"))
```

Fitting is Laplace + L-BFGS with a finite-difference outer gradient. Beyond
`K` (required) and `link`, the fitter takes `mask` and `offset`, warm-start
`β_init` / `Λ_init`, and the usual `g_tol` / `iterations` /
`newton_maxiter` / `newton_tol` controls. There is **no** `X`-covariate
keyword on this route.

`TruncatedPoisson()` is distinct from the `HurdlePoisson()` occurrence ×
zero-truncated two-part family: `HurdlePoisson()` models the zero/non-zero
split with its own Bernoulli part, whereas `TruncatedPoisson()` has no
occurrence part at all — it treats zeros as structurally unobservable, not
as a competing process to be estimated.

Current limits: `TruncatedPoissonFit` is not in the package's Wald
confidence-interval union, so there is no CI route for this family, and the
R-bridge arm (`family = "truncated_poisson"`) is no-X only — `X`, `X_lv`,
missing-response masks, and CI are loud rejects there rather than silent
no-ops. The bridge requires finite positive integer responses exactly
representable as `Int`. Fractional counts and values that would lose precision
during conversion are rejected before fitting; the bridge never rounds this
family’s response.

### Right-censored Poisson — `CensoredPoisson()`


```julia
fit = fit_censored_poisson_gllvm(Y; K = 2, censored = falses(size(Y)))   # no censored cells
fit = fit_censored_poisson_gllvm(Y; K = 2, censored = cens)              # cens::BitMatrix; Y holds count/limit
fit = fit_censored_poisson_gllvm(Y; K = 2, lower = L, upper = U)         # interval-ready (lower, upper)
fit = fit_gllvm(Y; family = CensoredPoisson(), K = 2, censored = cens)   # same route via fit_gllvm
```

Right-censored counts — e.g. an abundance recorded only as "at least `C`"
above some detection ceiling. Log link on the untruncated mean `μ =
exp(η)`; `P(y ≥ C | μ)` is evaluated as `logcdf(Gamma(C, 1), μ)` for numerical
stability rather than `1 − cdf`. Encode censoring either as `(lower, upper)`
integer matrices — uncensored `lower == upper == y`; right-censored `lower == C`,
`upper == typemax(Int)` — or as a Boolean `censored` matrix with `Y` holding
the observed count (uncensored cells) or the censoring limit (censored
cells) at each position. Pass one encoding or the other, never both
(`"censored_poisson: pass either (lower,upper) or censored, not both"`).
Left-censoring and finite (non-right) intervals are rejected — a documented
future extension. Only `LogLink()` is supported; there is no dispersion
parameter, so packing is Poisson-identical, `θ = [β; pack(Λ)]`; there is no
`X`-covariate keyword.

Both encodings fail loud on out-of-range cells rather than silently
recoding them: a censored cell needs `C ≥ 1`
(`"censored_poisson: right-censored C ≥ 1; got C=…"` — `C = 0` is
uninformative, since `P(Y ≥ 0) = 1`), and an uncensored cell needs `y ≥ 0`
(`"censored_poisson: uncensored y ≥ 0; got y=…"`). A non-log link throws
before any fitting (`"fit_censored_poisson_gllvm: only LogLink is supported
(Identity lock)"`), because the censored η-derivatives are hand-coded
against `μ = exp(η)`.

`CensoredPoisson` is **Julia-forward**: the twin `gllvmTMB` exposes a
constructor for it but has no working density behind it, so there is no
twin-parity claim for this family. `CensoredPoissonFit` is not among the fit
types handled by the package's Wald confidence-interval dispatch, and there
is no R-bridge route for it.

### Negative Binomial — `NegativeBinomial()`

```julia
fit = fit_gllvm(Yc; family = NegativeBinomial(), K = 2)   # per-species r (default)
```

For overdispersed counts. The NB2 variance function is Var = μ + μ²/r. The
public `fit_gllvm` default estimates **per-species** dispersion (returns
`NBGroupedFit`; `fit.r_group`), matching gllvmTMB's length-`p` `log_phi_nbinom2`.
With shared site covariates (`@formula` / bridge `X`), the default is
[`fit_nb_gllvm_grouped_cov`](@ref) (per-trait `r` + shared `γ`). As `r → ∞` the
negative binomial collapses to Poisson. For a single shared `r` across species,
call [`fit_nb_gllvm`](@ref) (no-X) or [`fit_gllvm_cov`](@ref) (with X).

The recorded NB2 comparison currently fails the stricter likelihood and
fit-health checks: its Julia convergence flag accompanies an unstable numerical
gradient. A scalar-density precision problem at large size has been identified
and is awaiting a tested repair. Do not infer verified parity or parameter
recovery from the convergence flag alone.

### Negative binomial type-1 — `NB1()`

```julia
fit = fit_gllvm(Yc; family = NB1(), K = 2)   # per-species φ (default)
```

The NB1 variance function is linear in the mean, Var = μ(1+φ) — quasi-Poisson-like
overdispersion, the alternative to NB2's quadratic tail. `NB1` is GLLVModels.jl's own
exported marker (there is no `Distributions` counterpart) and matches R gllvm's
`negative.binomial1` with the same `φ`. The public `fit_gllvm` default estimates
**per-species** dispersion (returns `NB1GroupedFit`; vector `fit.φ`), matching
gllvmTMB's length-`p` `log_phi_nbinom1` and the estimand already used with shared
site covariates ([`fit_nb1_gllvm_grouped_cov`](@ref)). For a single shared `φ`,
call [`fit_nb1_gllvm`](@ref).

The marker's `φ` field is a tag payload: `NB1()` and `NB1(2.5)` give the same fit,
because `φ` is always estimated and never seeded from the marker. Pass `φ_init` to
the named fitters to set a starting value.

### Zero-truncated Negative Binomial (NB2) — `TruncatedNegBin2()`


```julia
fit = fit_gllvm(Yc; family = TruncatedNegBin2(), K = 2)   # counts, y ≥ 1; shared r
fit = fit_truncated_nbinom2_gllvm(Yc; K = 2, r_init = 5.0) # same route, explicit seed
fit = fit_truncated_nbinom2_gllvm_pertrait(Yc; K = 2)      # per-trait r_t (twin log_phi_truncnb2)
```

The zero-truncated counterpart of `NegativeBinomial()` (twin `truncated_nbinom2()`,
family_id 11) — for counts where zeros are structurally unobservable. Support is
`{1, 2, …}`; the log link applies to the *untruncated* mean `μ = exp(η)`, with the
NB2 variance function `Var = μ + μ²/r` before truncation. The per-cell density is
`NB2(y; μ, r) / (1 − p₀)` with `p₀ = (r/(r+μ))^r`.

`Yc` must be integer-valued counts, and **every observed cell must be `≥ 1`**:

```
ArgumentError: truncated_nbinom2 requires y ≥ 1; found y=0 at (2,1)
```

Masked cells are exempt from that check. Both routes accept a `mask`, derive one
automatically when `Yc` contains `missing`, and accept an `offset`. Only
`LogLink()` is supported — anything else throws
(`"only LogLink is supported (twin truncated_nbinom2)"`). There is no
`X`-covariate keyword on either route.

#### Dispersion: shared vs per-trait

With `disp_group` omitted, `fit_gllvm` calls [`fit_truncated_nbinom2_gllvm`](@ref),
which estimates a single **shared** `r` across species and packs
`[β; pack(Λ); log r]` (length `p+rr+1`). Explicit `disp_group = :species` calls
[`fit_truncated_nbinom2_gllvm_pertrait`](@ref), estimating per-trait `r_t`
(twin `log_phi_truncnb2`) and packing `[β; pack(Λ); log r_1 … log r_p]`
(length `p+rr+p`). A length-`p` vector of distinct positive integer group IDs
also selects this per-trait model; repeated IDs and partial grouping are unsupported.

```@setup truncated_nb2_dispatch
using GLLVModels, Random, Distributions
_TNB2_SEED = 58
function parity_loadings_p5k2()
    return [
        0.8   0.0
        0.5   0.6
        0.3  -0.4
       -0.2   0.5
        0.1   0.3
    ]
end
Random.seed!(_TNB2_SEED)
p, K, n = 5, 1, 120
# Intercepts chosen so μ ≳ 3: keeps p₀ small (cheap rejection sampling), the
# truncation correction well conditioned, and the Laplace modes stable.
β = log.([4.0, 5.0, 3.5, 4.5, 4.0])
r_true = 4.0
Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
Z = randn(K, n)
η = β .+ Λ * Z
Y = Matrix{Int}(undef, p, n)
for t in 1:p, s in 1:n
    μ = exp(clamp(η[t, s], -3.0, 3.5))
    while true                      # zero-truncated draw by rejection
        v = rand(Distributions.NegativeBinomial(r_true, r_true / (r_true + μ)))
        if v >= 1
            Y[t, s] = v
            break
        end
    end
end

Yc = Y
```

```@example truncated_nb2_dispatch
fit = fit_gllvm(Yc; family = TruncatedNegBin2(), K = 1, disp_group = :species)
wide = gllvm(@formula(y ~ 1), Yc, (site = 1:size(Yc, 2),);
             family = TruncatedNegBin2(), K = 1, disp_group = :species)
@assert fit.converged && wide.converged
@assert isapprox(fit.loglik, wide.loglik; atol=1e-10, rtol=0)
(length(wide.r), wide.converged)
```

A complete long table with `y`, `species` and `site` columns accepts the same keywords.

This formula route admits intercept-only models. It does not add site covariates,
new interval methods, or R-bridge qualification.

The `TruncatedNegBin2` marker carries an `r` field (`TruncatedNegBin2()` fills in
`10.0`), but **no fit path reads it** — `r` is always jointly estimated. To seed
the search, pass `r_init` to the named fitters: a scalar on the shared route, a
scalar or a length-`p` vector on the per-trait route. Left unset, both start from
`r = 10`.

Equal `r_t` reproduces the shared-`r` log-likelihood **by construction**, not by
coincidence: `truncated_nbinom2_marginal_loglik_laplace` is implemented as the
equal-`r_t` special case of `truncated_nbinom2_pertrait_marginal_loglik_laplace`,
so the two routes cannot drift apart.

#### Scope of the recorded R comparison

The recorded seed-58 per-trait comparison has a verified result using a
public R BFGS continuation from its default fit. Both engines pass the recorded
fit-health checks, with absolute log-likelihood difference below `1e-7`.
The default R fit still reports unsuccessful convergence. This result applies
to that explicit continuation recipe; it does not establish default-R health,
parameter-recovery performance, or complete family parity. The required parity case now records this policy explicitly and preserves
the default R fit's unsuccessful convergence in its evidence.

#### Laplace curvature: `hessian = :observed` (default) or `:fisher`

Both routes take a `hessian` keyword. `:observed` (the default) builds the Laplace
log-determinant term from the exact conditional curvature `−∂²ℓ/∂η²` of the
zero-truncated NB2 at the log link; `:fisher` uses the expected-information
weight instead. The two differ here because the NB2 term is **`y`-dependent**
through `−(y+r)·log(μ+r)` — unlike zero-truncated Poisson, where `y` enters `η`
linearly and the two curvatures coincide pointwise. The observed weight
(`_truncnb2_observed_weight`) is checked against ForwardDiff to a max relative
error of 1.8e-13 over 125 `(μ, r, y)` cells spanning `μ ∈ [0.5, 25]`,
`r ∈ [0.3, 50]`, `y ∈ [1, 40]`.

The keyword selects only the weight in that log-det term; the mode solve itself is
a Fisher-scoring iteration in both cases, which changes *how* the mode is found,
not the objective. An unrecognised symbol is rejected up front with a clear
`ArgumentError` rather than inside the objective, whose `try`/`catch` would
otherwise convert a typo into a converged-looking garbage fit.

#### Limits

The outer optimisation is Optim LBFGS with a **finite-difference** gradient over
the packed vector, not a hand-coded analytic outer gradient. Neither
`TruncatedNegBin2Fit` nor `TruncatedNegBin2PerTraitFit` has a `confint` dispatch,
so `confint(fit, Y)` is not available for this family, and there is no R-bridge
route for it.

### Beta — `Beta()`

```julia
fit = fit_gllvm(Yp; family = Beta(), K = 2)   # per-species φ (default)
```

For proportions strictly inside (0,1) — e.g. cover fractions, frequencies.
The per-observation law is Beta(μφ, (1−μ)φ), so Var = μ(1−μ)/(1+φ). The
public `fit_gllvm` default estimates **per-species** precision (returns
`BetaGroupedFit`; vector `fit.φ`), matching gllvmTMB's length-`p` `log_phi_beta`.
With shared site covariates, the default is [`fit_beta_gllvm_grouped_cov`](@ref)
(per-trait `φ` + shared `γ`). For a single shared precision, call
[`fit_beta_gllvm`](@ref) (no-X) or [`fit_gllvm_cov`](@ref) (with X).

### Ordinal — `Ordinal()`

```julia
fit = fit_gllvm(Yo; family = Ordinal(), K = 2)
```

For ordered categorical responses coded `1:C` (e.g. Likert scales, abundance
classes). Uses a proportional-odds cumulative-logit model with `C−1` ordered
cutpoints `τ` shared across species. There is no species intercept — the
cutpoints carry the category levels. The fitted cutpoints are available as
`fit.τ`. The cumulative link is `LogitLink()` by default; pass
`link = ProbitLink()` for a cumulative-probit (ordered-probit) model.

For native `gllvmTMB` bridge parity, use `fit_ordinal_gllvm_pertrait()` (no-X)
or `fit_ordinal_gllvm_pertrait_cov()` / `@formula`+X / bridge X (shared site-X):
per-trait cutpoints with τ₁=0 fixed and K−2 free log-spacings, plus shared `γ`
under X (twin API B). The shared-cutpoint `fit_ordinal_gllvm()` route remains a
Julia-side comparator and is **not** the public X default.

The shared-cutpoint route also admits predictor-informed latent-score means:

```julia
fit_xlv = fit_ordinal_gllvm(Yo; K = 1, X_lv = X_lv)
extract_lv_effects(fit_xlv)
confint_lv_effects(fit_xlv, Yo, X_lv; method = :profile,
                   profile_indices = [1])
```

Those intervals target the native Julia `B_lv = Λ * alpha_lv'` product. They do
not promote per-trait ordinal bridge CI parity.

### Multinomial — `GLLVModels.Multinomial()`


```julia
Y = reshape(y, 1, n)                                    # y::Vector{Int}, values in 1:K, K ≥ 3

fit = fit_multinomial_gllvm(Y; n_categories = K)        # 1×n matrix, or a length-n vector
fit = fit_gllvm(Y; family = GLLVModels.Multinomial())        # same route; Y must be a MATRIX here
fit = fit_multinomial_gllvm(Y; X = X, n_categories = K) # + site covariates (n×p, no intercept column)
```

Unordered categorical responses (twin `multinomial()`, family_id 16) via
fixed-effects baseline-category softmax: `η₁ ≡ 0`, `η_k = β_k + xᵀγ_k` for
`k = 2,…,K`, `P(y=k) = softmax(η)_k`. **Unlike every other family in this
table, `Y` here is one categorical trait, not a species-by-site matrix**:
pass a `1×n` integer row, a length-`n` integer vector, or a `K×n` one-hot
matrix for a single categorical variable. Stacking several categorical traits
into one call is rejected — `"multinomial v1 is one unordered trait per fit
(1×n integer categories or K×n one-hot); do not expand TMB K−1 pseudo-rows"`
— so fit each trait separately. Note the vector form is accepted only by
`fit_multinomial_gllvm`; `fit_gllvm` takes a matrix, so reshape to `1×n`
before routing through it.

Categories must be coded `1:K` with **`K ≥ 3`**. Both constraints fail loud:
any `y < 1` (or `y > K`) throws `"multinomial requires y ∈ {1, …, K}; found
y=$v"`, and `K = 2` throws `"multinomial requires K ≥ 3 categories; K = 2 is
binomial-logit — use Binomial() / LogitLink()"`. `K` itself comes from
`n_categories` when you pass it; **when `n_categories` is left unset, `K` is
inferred as `maximum(y)`** — so pass it explicitly whenever the top category
may be unobserved in your sample, or you will silently fit a smaller `K`.
Only `LogitLink()` is supported, and that is an explicit guard, not just a
default: any other link throws `"fit_multinomial_gllvm: only LogitLink is
supported (twin multinomial)"`.

**v1 is fixed-effects softmax only — no latent variables.** `fit_gllvm`
rejects any `K` / `num_lv` other than `nothing` or `0`, and rejects
`row_eff`, `disp_group`, or `pervar`; `fit_multinomial_gllvm` carries the
same `K` / `num_lv` rejection itself. There is no dispersion parameter. The
packed vector is contrast-major, `[β₂…β_K; γ₂; …; γ_K]`, of length
`(K−1)(1+p)`. `X` (site covariates, `n×p`, no intercept column) is supported
directly on `fit_multinomial_gllvm` and passes through `fit_gllvm`. The
returned `MultinomialFit` carries `β` (length `K−1`), `γ` (`(K−1)×p`),
`n_categories`, `link`, `loglik`, `converged`, `iterations`, and
`theta_packed`; there are no loadings in v1. Multinomial fits have no Wald
confidence-interval dispatch and no R-bridge route.

The marker is `GLLVModels.Multinomial`, not `Distributions.Multinomial` (the
count-vector law). GLLVModels.jl deliberately excludes `Multinomial` from its
`using Distributions` list so the identity marker can bind unqualified
*inside* the package — but both packages **export** the name, so in user code
that does `using GLLVModels, Distributions` the bare name `Multinomial` is
undefined rather than resolving to either one. Always write
`GLLVModels.Multinomial()` (and `Distributions.Multinomial(...)` for the count
law). Because v1 has no latent variables, `Multinomial` is **not yet a general
GLLVM workflow**. It is a fixed-effects model for one unordered categorical
response.

### Gamma — `Gamma()`

For positive-continuous data with Var = μ²/α (constant coefficient of variation),
the no-X public default remains shared-shape [`fit_gamma_gllvm`](@ref) /
`fit_gllvm(...; family = Gamma())`. With shared site covariates, the public /
bridge default is [`fit_gamma_gllvm_grouped_cov`](@ref) (per-trait shape `α` +
shared `γ`; twin API B). Shared-α + X remains the opt-in
[`fit_gllvm_cov`](@ref). Per-species shape without X is available via
[`fit_gamma_gllvm_grouped`](@ref) / `disp_group`.

```julia
fit = fit_gllvm(Yp; family = Gamma(), K = 2)   # Yp > 0; shared α (no-X)
```

!!! warning "Cloglog fits can saturate the Laplace approximation (diagnostic added 2026-08-28)"
    The complementary log-log link's doubly-exponential upper tail saturates at
    η ≈ 3.3 (logit needs ~35), so a Binomial/cloglog fit can converge onto a
    "saturation ridge" where the Laplace log-determinant penalty is locally
    deleted and the reported log-likelihood strongly overstates the exact
    marginal (measured: +74.8 loglik units with ‖Λ̂‖ ≈ 27 against a simulated
    truth of 0.9, `converged = true`). This is an intrinsic property of the
    Laplace approximation at this link — the implementation is FD-verified —
    so `fit_binomial_gllvm` now computes a post-fit saturation health record
    (`fit.saturation`), emits a warning when any cell reaches a saturation
    threshold or its log-det weight collapses, and tags `show` output with
    `SATURATED (k cells)`. The diagnostic reports; it never alters the
    objective or the `converged` flag. Treat saturated cloglog fits with
    suspicion, especially the loadings.

!!! note "Laplace curvature: Beta, NB1 and Student-t use the observed Hessian"
    The shared routes of Beta/logit, NB1/log and Student-t/identity now
    default to the **observed** conditional curvature in the Laplace
    log-determinant, matching TMB / `gllvmTMB` (their grouped fitters already
    did). In a 900-cell comparison with the exact-ML optimum, the observed
    curvature's estimates are closer in
    90–100% of realistic cells for all three; its reported log-likelihood is
    measurably more biased for these families — that trade-off was accepted
    deliberately. **Reported logliks, AIC/BIC and Wald SEs change**;
    `hessian = :fisher` restores the previous objective. Beta's and
    Student-t's observed curvature can be genuinely negative; the
    positive-definiteness guard at the Laplace assembly handles that case.

!!! note "Laplace curvature: NB2 uses the observed Hessian (changed 2026-08-27)"
    The shared-dispersion NB2 route (`fit_nb_gllvm`, and `fit_gllvm` when it
    does not coerce to the grouped route) now uses the **observed** conditional
    curvature `−∂²ℓ/∂η² = μ·r·(r+y)/(r+μ)²` in the Laplace log-determinant,
    matching TMB / `gllvmTMB`. The grouped per-trait route already did.
    **Reported log-likelihoods and Wald SEs change**; point estimates move
    little. A 900-cell comparison (2026-08-27) found that NB2 preferred the
    observed curvature on both the
    estimator-quality and approximation-accuracy metrics. The previous
    behaviour stays reachable via `hessian = :fisher`.

!!! note "Laplace curvature: Gamma uses the observed Hessian (changed 2026-08-25)"
    Gamma's Laplace log-determinant now uses the **observed** conditional
    curvature `−∂²ℓ/∂η² = α·y/μ`, matching what TMB — and therefore
    `gllvmTMB` — computes. It previously used the Fisher (expected)
    information, the constant `α`, which is exactly the *expectation* of the
    correct weight.

    **This changes reported log-likelihoods for Gamma fits.** Point estimates
    move very little (the conditional mode is determined by the score, which is
    unaffected), but `loglik`, and anything derived from the Hessian such as
    Wald standard errors, will differ from earlier versions.

    Measured against high-resolution numerical quadrature across 12 seeds, the
    observed curvature is closer to the exact marginal in 12/12 cases, with
    20–60× smaller error. The previous behaviour remains available:

    ```julia
    ll = gamma_marginal_loglik_laplace(Y, Λ, β, α; hessian = :fisher)
    ```

    Other families are **unchanged** and keep the Fisher default. This was a
    per-family decision made on per-family evidence, not a global switch — for
    some families the observed curvature is *not* closer to the exact marginal,
    so each is decided on its own measurements.

!!! note "Laplace curvature: TweedieED and Binomial-probit use the observed Hessian (changed 2026-08-28)"
    A 2026-08-28 maintenance decision established that
    the shared Tweedie route (`fit_tweedie_gllvm`) and Binomial/probit both
    default to the **observed** conditional curvature in the Laplace
    log-determinant, matching TMB / `gllvmTMB` structurally — TMB
    differentiates the joint negative log-likelihood, so its log-det is
    observed for *every* family it ships, not a per-family exception.
    `hessian = :fisher` restores the previous objective for both.

    - TweedieED/log: `−∂²ℓ/∂η² = μ^(1−p)·[(2−p)·μ + (p−1)·y] / φ`. The
      normalising series in the Tweedie density is μ-free, so it contributes
      no η-curvature and the closed form above is exact, not an
      approximation. Always non-negative (μ, φ > 0, p ∈ (1,2), y ≥ 0).
    - Binomial/probit: `−∂²ℓ/∂η² = η·φ(η)·(y−nμ)/(μ(1−μ)) + φ(η)²·[y/μ² +
      (n−y)/(1−μ)²]`, μ = Φ(η). Provably non-negative for every (η, y, n) —
      the probit binomial log-likelihood is globally concave in η (Pratt 1981,
      *JASA*), so unlike Beta/Student-t the positive-definiteness guard is not
      expected to fire for this family.
    - Binomial/**cloglog** is explicitly excluded and stays `:fisher` — the
      diagnosed Laplace saturation pathology above, not a pending flip.
    - The Tweedie **grouped** route accepts `hessian = :observed` (default)
      or `:fisher`. This selects the Laplace curvature; the mode-search policy
      stays separate. Comparisons must use the same curvature and power model.

### Lognormal — `Lognormal()`


```julia
fit = fit_gllvm(Y; family = Lognormal(), K = 2)   # Y > 0; log link on E[log y] = η
fit = fit_lognormal_gllvm(Y; K = 2)               # same route, called directly
```

A one-part lognormal GLLVM (twin `lognormal()`, family_id 3): `log(y) ~
Normal(η, σ²)` with `η = β + Λz`. On the log scale this is exactly the
Gaussian GLLVM, so `fit_lognormal_gllvm` **reuses the closed-form Gaussian
fitter** on `log(Y)` rather than a Laplace approximation: per-trait intercepts
`β_t = mean_s log(Y[t,s])` are removed first, then `fit_gaussian_gllvm`
estimates `(Λ, σ)` on the centred log scale. The reported `loglik` is on the
*y*-scale — the Gaussian log-scale marginal plus the change-of-variables
Jacobian `−Σ log y`.

`Y` must be strictly positive; a non-positive cell throws
`ArgumentError("lognormal requires y > 0; found non-positive response")`.
Only `LogLink()` is supported — any other link throws
`ArgumentError("fit_lognormal_gllvm: only LogLink is supported (twin lognormal)")`.

`Lognormal` is distinct from `Distributions.LogNormal` and from the two-part
`DeltaLogNormal()` hurdle (occurrence × positive lognormal, fid 12) — this is
the one-part law with no zero mass.

`LognormalFit` carries `β` (length `p`, mean of `log y`), `Λ` (`p×K`, on the
log scale), `σ` (`Var(log y) = σ²`), `link`, `loglik`, `converged`,
`iterations`, and the free-σ reference packing
`theta_packed = [β; pack(Λ); log σ]`. To evaluate the y-scale marginal at
explicit parameters, use the exported `lognormal_marginal_loglik(Y, Λ, β, σ)`.

The response mean uses the standard lognormal bias correction —
`E[y | η] = exp(η + σ²/2)`, not `exp(η)` (the median). `lognormal_response_mean`
is a **scalar** function of `(η, σ)`; broadcast it for an array of linear
predictors:

```julia
lognormal_response_mean(fit.β[1], fit.σ)    # E[y] for trait 1 at z = 0
lognormal_response_mean.(fit.β, fit.σ)      # broadcast over the p intercepts
```

Note that you must supply `η` yourself: `LognormalFit` defines no `getLV`,
`predict`, or `link_residual` method, so latent scores and fitted linear
predictors are not extracted from the fit object on this engine.

**Surface limits.** There is no Wald confidence-interval route —
`LognormalFit` is not in the native `confint` family union — and the R bridge
rejects CI requests loudly, requiring `ci_method = "none"`. On the bridge,
lognormal is a **no-X** family: fixed-effect `X`, `X_lv`, and missing-response
masks are all fenced as follow-ups, and bridge `scores` come back empty with
`Sigma` / `correlation` assembled from the shared block `ΛΛᵀ` alone
(communality `1`). `fit_lognormal_gllvm` itself declares no `X` keyword; it
forwards remaining keywords to `fit_gaussian_gllvm`, but no test exercises an
X-covariate route through the lognormal entry point, so treat it as
unsupported.

### Tweedie — `fit_tweedie_gllvm`

Compound Poisson–Gamma for biomass / abundance with true zeros: an exact point
mass at 0 plus a positive continuous part, `Var = φ μ^p` with `1 < p < 2`. The shared
`fit_tweedie_gllvm` estimates dispersion and power. The grouped fitter also
supports fixed common power and per-species estimated power:

```julia
fit = fit_tweedie_gllvm(Y; K = 2)                    # Y ≥ 0; φ and p estimated
fit = fit_tweedie_gllvm_grouped(Y; K = 2)            # φ per species, shared p
fit = fit_tweedie_gllvm_grouped(Y; K = 2, power = 1.5) # fixed common p
fit = fit_tweedie_gllvm_grouped(Y; K = 2, power_group = :species) # p per species
```

`φ` and the power sit on a nearly flat joint ridge, so the reported `converged`
flag is deliberately strict: on top of the optimiser's own verdict, the fit is
flagged as **not** converged if the Laplace marginal could not be evaluated at
the returned point (`loglik` is then `-Inf`, never an internal sentinel), if the
power has run to the closed end of `(1, 2)`, or if the gradient residual is large
relative to the objective's own scale. Treat `converged = false` as a real
result: try a different `p_init` / `power_init`, or a cell with more sites.
The same contract applies to [`fit_tweedie_gllvm_grouped`](@ref) (and therefore
to the already-shipped `fit_gllvm(...; disp_group = :species)` route). The
bare-marker `fit_gllvm` admit is still closed.

See [Tweedie power contracts](tweedie-power.md) for parameter counts and the
distinction between a shared reference constraint and the R default model.

### Student-t — `StudentTFamily(ν)`

```julia
fit = fit_gllvm(Y; family = StudentTFamily(4.0), K = 2)   # heavy-tailed continuous
fit = fit_gllvm(Y; family = StudentTFamily(), K = 2)      # estimate ν (per-species default)
```

An outlier-robust drop-in for `Normal()` on the identity link: the
per-observation law is the location–scale t, `(y − η)/σ ~ t_ν`. The heavy tail
bounds the influence of extreme cells — the score `(ν+1)r/(νσ² + r²)` *decreases*
in the residual `r`, so a handful of gross outliers barely move `β̂`, where a
Gaussian fit would chase them. As `ν → ∞` the family tends to `Normal(η, σ²)`.

The two marker fields play different roles. A numeric `ν` fixes the degrees
of freedom; `StudentTFamily()` carries `ν = nothing` and estimates it.
`fit_gllvm` selects per-species scales and degrees of freedom for that default.
The named fitter also supports `disp_group = :shared`. Set degrees of freedom
on the family marker when using `fit_gllvm`; use the `nu` keyword on
[`fit_studentt_gllvm`](@ref). The scale `σ` is a **tag
payload** — it is always estimated (returned as `fit.σ`), so `StudentTFamily(4.0)`
and `StudentTFamily(4.0, 9.0)` give the same fit; seed it with `σ_init` on the
named fitter instead.

Student-t is a **no-X** surface: `fit_gllvm` and `gllvm(@formula(y ~ 1), …)` are
admitted. Shared and per-species dispersion are supported; this does not
establish arbitrary dispersion-group, covariate or row-effect support. The fit
records `estimated_nu`, so AIC counts free degrees-of-freedom parameters only
when they were estimated. See [Student-t parity limits](studentt-parity.md):
the recorded R comparison still does not converge reliably.

### Conway–Maxwell–Poisson — `COMPoisson()`

```julia
fit = fit_gllvm(Y; family = COMPoisson(), K = 2)      # counts; ν estimated
fit = fit_gllvm(Y; family = COMPoisson(2.0), K = 2)   # same — marker ν never read
```

A one-parameter-dispersion count family that handles **both** under- and
over-dispersion via the exponent `ν` (`ν = 1` ⇒ Poisson, `ν > 1` ⇒
underdispersion, `ν < 1` ⇒ overdispersion). The named fitter
[`fit_compoisson_gllvm`](@ref) always estimates `ν` (seeded by `ν_init`,
default 1). The marker's `ν` field is a **tag payload** — it is never read —
unlike [`StudentTFamily`](@ref), where a numeric `ν` is a fixed model control
and `nothing` requests estimation. `COMPoisson()` is the usual call; `COMPoisson(9.0)` gives the same fit.

COM-Poisson is Julia-forward (the twin has no CMP family) and a **no-X**
surface: `fit_gllvm` and `gllvm(@formula(y ~ 1), …)` are admitted, but
covariates, `disp_group`, and row effects are not. There is no twin light Δ.

### Generalized Poisson type-1 — `GeneralizedPoisson1(α)`


```julia
fit = fit_gllvm(Y; family = GeneralizedPoisson1(0.3), K = 2)   # counts; α estimated (tag payload)
fit = fit_gp1_gllvm(Y; K = 2, α_init = 0.3)                    # same driver, explicit seed
```

Counts with a **signed** scalar dispersion `α` (Famoye / Consul–Jain
mean-parameterisation, log link): `μ = exp(η)`, `E[y] = μ`,
`Var = μ(1+αμ)²`. `α = 0` is the Poisson limit; `α > 0` overdisperses and is
exactly normalised for any `α ≥ 0`; `α < 0` underdisperses, with finite
support `y < 1/|α|` and a small loss of tail mass once `|α|μ` is large — an
intrinsic property of the underdispersed GP-1, not an implementation defect.
As for `NB1()` and `COMPoisson()`, the marker's `α` field is a **tag
payload**: it is required by the constructor but `fit_gllvm` never reads it,
so `α` is always estimated from the data by [`fit_gp1_gllvm`](@ref); seed the
search with `α_init` rather than the marker value.

`fit_gp1_gllvm` fits by **profiling over `α`** — a fixed grid of trial values,
each one refit for `(β, Λ)` at that *fixed* `α` (warm-start-chained), then a
Brent refinement of `α` on the profile and a final `(β, Λ)` refit — rather
than one joint L-BFGS over `[β; Λ; α]`, because `α` and the latent-factor
variance both absorb overdispersion and a naive joint fit collapses onto a
degenerate optimum (`Λ → 0`, `α` pinned at its bound). `α_bound` (default
`2.0`) caps `|α|`; raise it if a fit saturates near the cap. `α_init`, if
given, is clamped into `±0.99·α_bound` and added to the profile grid as a
seed. `Y` is a `p×n` integer count matrix and may contain `missing` (pass a
`mask`, or leave `missing` entries in `Y` — masked cells are dropped from the
marginal and the warm start); an `offset` keyword (`η = β + offset + Λz`) is
accepted. GP-1 has no `X`-covariate keyword — it is a **no-X surface**.

`link` defaults to `LogLink()` and the family pieces above are derived for
`μ = exp(η)`. Note that — unlike `TruncatedPoisson()`, which throws on any
other link — GP-1 has **no link guard**: another `link` is accepted without
an error, so pass one only deliberately.

Post-fit, a `GP1Fit` supports `getLV`, `predict(fit, Y; type = :link` or
`:response)`, `residuals(fit, Y; type = :dunnsmyth` or `:pearson)`, and
`aic` / `bic` (parameter count `β` + `Λ` + `α`). Confidence intervals go
through the unified family layer — `confint(fit, Y; method = :wald)`,
`:profile`, or `:bootstrap` — with `α` reported on its raw signed scale (not
a log scale) under the term name `"alpha"`.

### Delta-lognormal — `DeltaLogNormal()`

```julia
fit = fit_gllvm(Y; family = DeltaLogNormal(), K = 2)   # occurrence × positive lognormal
fit = fit_gllvm(Y; family = DeltaLogNormal(9.0), K = 2)  # same — marker σ never read
```

Two-part Laplace: Bernoulli occurrence (`π = logistic(β^z)`, `Λ_z = 0` in v1) times
a positive lognormal with meanlog `η^c = β^c + Λ_c z` and shared sdlog `σ`. The
marker's `σ` is a **tag payload** — always estimated (returned as `fit.σ`). Named
fitter [`fit_delta_lognormal_gllvm`](@ref) remains available.

Delta-lognormal is a **no-X** surface: `fit_gllvm` and `gllvm(@formula(y ~ 1), …)`
are open; covariates, `disp_group`, and `row_eff` are not admitted. No bridge /
R-parity claim.

**`predictor` mode (2026-08-28):** [`fit_delta_lognormal_gllvm`](@ref) takes a
`predictor::Symbol` kwarg, `:separate` (default, the behaviour above) or
`:shared`. gllvmTMB's `delta_lognormal()` ties occurrence and the positive
part to ONE shared linear predictor (`gllvmTMB.cpp:2816-2830`); `predictor =
:shared` reproduces that — `βz ≡ βc`, `Λz ≡ Λc` — as a twin-parity-oriented
mode, not a general recommendation over `:separate`: occurrence log-odds and
log-abundance move together by construction under `:shared`.

### Delta-Gamma — `DeltaGamma()`

```julia
fit = fit_gllvm(Y; family = DeltaGamma(), K = 2)   # occurrence × positive Gamma
fit = fit_gllvm(Y; family = DeltaGamma(4.0), K = 2)  # same — marker α never read
```

Two-part Laplace: Bernoulli occurrence times a positive Gamma with log-link mean
`μ = exp(η^c)` and shared shape `α` (`Var = μ²/α`). The marker's `α` is a **tag
payload** — always estimated (returned as `fit.α`). Named fitter
[`fit_delta_gamma_gllvm`](@ref) remains available.

Delta-Gamma is a **no-X** surface: same fence as Delta-lognormal (no +X, no
`disp_group`, no `row_eff`, no bridge / R-parity claim).

**`predictor` mode (2026-08-28):** same kwarg as Delta-lognormal above —
[`fit_delta_gamma_gllvm`](@ref)'s `predictor::Symbol`, `:separate` (default)
or `:shared` (twin-identity mode, `gllvmTMB.cpp:2831-2844`). DeltaGamma is
the one two-part family whose observed count-part weight is implemented, so
`hessian = :observed` vs `:fisher` genuinely differ under `:shared` too.

### Beta-binomial — `BetaBinom()`

```julia
fit = fit_gllvm(Yb; family = BetaBinom(), K = 2, N = trials)   # per-species φ (default)
```

For binomial counts that are **over-dispersed** relative to `Binomial(N, μ)` — the
per-trial success probability is itself random, `p ~ Beta(a, b)` with `a = μφ`,
`b = (1−μ)φ`. `Y` is a `p × n` matrix of integer successes; `N` the matching trial
counts. The Beta precision `φ` (the shape-sum `a + b`) is jointly estimated and
available as `fit.φ`; as `φ → ∞` the family collapses to `Binomial(N, μ)`. Links:
`LogitLink()` (default), `ProbitLink()`, `CLogLogLink()`.

`BetaBinom` is GLLVModels.jl's own exported marker, named to avoid colliding with
`Distributions.BetaBinomial`. The public `fit_gllvm` default estimates
**per-species** `φ` (returns `BetaBinomialGroupedFit`; vector `fit.φ`), matching
gllvmTMB's length-`p` `log_phi_betabinom` and the estimand already used with shared
site covariates ([`fit_beta_binomial_gllvm_grouped_cov`](@ref)). For a single shared
`φ`, call [`fit_beta_binomial_gllvm`](@ref).

Unlike the named fitters, which default `N` to all-ones, **`fit_gllvm` and `gllvm`
require `N`** as a p×n matrix. At `N = 1` the beta-binomial collapses to
`Bernoulli(μ)` and `φ` is unidentifiable, so there is no safe default at a public
entry point; a missing `N` — or a scalar, which is not broadcast — raises a clear
error. As for `NB1`, the marker's `φ` field is a tag payload: `BetaBinom()` and
`BetaBinom(7.5)` give the same fit. Pass `φ_init` to the named fitters to set a
starting value.

### Per-species and grouped dispersion

For `NegativeBinomial`, `Beta`, `NB1`, and `BetaBinom`, the public `fit_gllvm`
default already uses per-species dispersion (`disp_group = :species`). Pass an
integer `disp_group` vector for custom grouping, or call the named shared fitters
(`fit_nb_gllvm` / `fit_beta_gllvm` / `fit_nb1_gllvm` /
`fit_beta_binomial_gllvm`) for one dispersion across all species.

The remaining dispersion families (`Gamma`, Tweedie) still default to a shared
parameter on `fit_gllvm` / their named drivers; use a `_grouped` driver or
`disp_group = :species` to vary by species (gllvm's `disp.group`):

```julia
fit_nb_gllvm_grouped(Yc;  K = 2, group = group)   # NB2 dispersion r per group
fit_nb1_gllvm_grouped(Yc; K = 2)                  # NB1 dispersion φ, default per-species
fit_beta_gllvm_grouped(Yp;    K = 2)              # Beta precision φ per species
fit_beta_binomial_gllvm_grouped(Yb; K = 2, N = trials)  # beta-binomial φ per species
fit_gamma_gllvm_grouped(Yc;   K = 2)              # Gamma shape α per species
fit_tweedie_gllvm_grouped(Yc; K = 2)             # Tweedie dispersion φ per species (shared power p)
```

(`fit_nb_gllvm_grouped` requires an explicit `group`; the other five default to
per-species. The beta-binomial drivers additionally need the trial counts `N`.)

```julia
fit_gllvm(Yc; family = NegativeBinomial(), K = 2)                 # default = per-species
fit_gllvm(Yc; family = NB1(), K = 2)                              # default = per-species
fit_gllvm(Yb; family = BetaBinom(), K = 2, N = trials)            # default = per-species; N required
fit_gllvm(Yp; family = Beta(), K = 2, disp_group = group)         # custom groups
fit_gllvm(Yp; family = Gamma(), K = 2, disp_group = :species)     # Gamma opt-in
```

`disp_group = :species` means one dispersion per species; an integer vector
assigns species to shared dispersion groups. Grouped dispersion is a single
specialised route: it is not combined with `row_eff` or Gaussian `pervar` in the
same call, and unsupported families fail with an `ArgumentError`.

### Gaussian with per-species variance — `fit_gaussian_pervar_gllvm`

```julia
fit = fit_gaussian_pervar_gllvm(Y; K = 2)   # heteroscedastic Gaussian
```

A heteroscedastic Gaussian GLLVM with a **separate residual variance per species**
(gllvm's heteroscedastic default), in contrast to the single shared `σ_eps` of
`fit_gaussian_gllvm`. With `X=nothing`, trait intercepts are profiled as row means of the `p × n`
response matrix. Supply `X` of shape `(p, n, q)` to fit a complete fixed-effect
design; no intercept is added automatically. Its `q` coefficients are profiled
by generalized least squares at each covariance trial and returned in `fit.β`.
Explicit designs use L-BFGS with direct covariance Cholesky (O(p³) factorization)
to avoid cancellation near a zero residual variance; the default intercept-only
path can use EM. A
zero-column design specifies zero mean; nonfinite or rank-deficient designs
are rejected. This is maximum likelihood, not REML.

The development option `fixed_residual_sd=c` fits
`Λ*Λ′ + Diagonal(ψ² .+ c^2)` and retains `fit.ψ²` (unique variances),
`fit.φ²` (total diagonal variances) and `fit.fixed_residual_sd`. The fixed scale
is not an estimated parameter. Positive `c` uses L-BFGS even with `method=:em`;
the default `c=0` leaves the existing model unchanged. It does not automatically
choose R's data-dependent small scale. The formula route is available with
`family=Normal(), pervar=true`; R bridge and intervals for this decomposition
remain unverified.

```@example pervar_design
using GLLVModels, Random
rng = MersenneTwister(7093)
p, n = 4, 80
site_x = collect(range(-1, 1; length=n))
Y = [0.7, -0.4, 0.5, 0.3] * randn(rng, 1, n) .+
    [0.5, 0.7, 0.9, 0.6] .* randn(rng, p, n) .+
    1.2 .* reshape(site_x, 1, n)
X = zeros(p, n, p + 1)
for j in 1:p
    X[j, :, j] .= 1                # one intercept per trait
end
X[:, :, end] .= reshape(site_x, 1, n)
fit_x = fit_gaussian_pervar_gllvm(Y; K=1, X=X)
@assert fit_x.converged # hide
coef(fit_x)                        # p intercepts, then one shared slope
```

For example, a known residual SD of `0.2` is kept fixed while the unique
variances and requested fixed effects are estimated:

```@example pervar_design
fit_fixed = fit_gaussian_pervar_gllvm(Y; K=1, X=X, fixed_residual_sd=0.2)
@assert fit_fixed.converged # hide
@assert isapprox(fit_fixed.φ², fit_fixed.ψ² .+ 0.2^2; atol=1e-12) # hide
println("Fixed residual SD: ", fit_fixed.fixed_residual_sd)
for (trait, variance) in enumerate(fit_fixed.ψ²)
    println("Unique variance ", trait, ": ", round(variance; digits=4))
end
```

The formula interface can build the same complete design. `y ~ 1 + site_x`
includes one intercept per trait and a shared slope. `y ~ 0 + site_x` removes
the intercepts; `y ~ 0` is a zero-mean model. Omitting the intercept marker
(`y ~ site_x`) includes trait intercepts. This applies to `pervar=true`; the
existing shared-variance formula route is unchanged. Complete long tables and
categorical contrast choices use the same per-variance route.

```@example pervar_design
fit_formula = gllvm(@formula(y ~ 1 + site_x), Y, (site_x=site_x,);
    family=GLLVModels.Normal(), K=1, pervar=true, fixed_residual_sd=0.2)
@assert fit_formula.converged # hide
@assert isapprox(fit_formula.loglik, fit_fixed.loglik; atol=1e-7) # hide
@assert isapprox(fit_formula.β, fit_fixed.β; atol=1e-6) # hide
println("Formula coefficients: ", length(fit_formula.β))
println("Formula/matrix likelihood agreement: ",
    isapprox(fit_formula.loglik, fit_fixed.loglik; atol=1e-7))
```

The model with a fixed residual and unique effects is outside AGHQ Stage 1a's
loadings-only domain. `aghq=3` or `aghq=:auto` therefore warns and retains the
exact Gaussian/Laplace fit. `aghq=1` follows the same path without an ignored-
request warning. No ridge or quadrature approximation is introduced, and the
fit reports which method actually ran:

```@example pervar_design
fallback = gllvm(@formula(y ~ 1 + site_x), Y, (site_x=site_x,);
    family=GLLVModels.Normal(), K=1, pervar=true, fixed_residual_sd=0.2, aghq=3)
@assert fallback.loglik == fit_formula.loglik # hide
@assert fallback.ψ² == fit_formula.ψ² # hide
println("Requested nodes: ", fallback.integration.requested_k)
println("Actual integration: ", fallback.integration.actual)
println("Reason: ", fallback.integration.reason)
```

The warning is intentional: requesting AGHQ does not remove the unique effects.
Without the fixed-residual decomposition (`fixed_residual_sd=0`), the plain
heteroscedastic AGHQ adapter remains unimplemented and reports the distinct
reason `pervar_aghq_unimplemented`; this is not a verified R structural rejection.
The default `aghq=false` leaves `integration=nothing`. Invalid node requests and
integration controls fail clearly. These examples check model equivalence and
reporting, not recovery or interval calibration.

### Mixed-family response vector — `fit_mixed_gllvm`


```julia
using GLLVModels, Distributions   # the family markers come from Distributions, not GLLVModels

families = [Normal(), Poisson(), Binomial()]
links    = [IdentityLink(), LogLink(), LogitLink()]
fit = fit_mixed_gllvm(Y; families = families, links = links, K = 2)

correlation(fit, Y)    # cross-family latent-scale trait correlation
sigma_y_site(fit, Y)   # Σ_latent = ΛΛᵀ + diag(σ²_d)
communality(fit, Y)    # per-trait communality on the latent scale
getLV(fit, Y)          # per-site latent scores
```

Each of the `p` rows (traits) of `Y` can carry its **own** response family and
link, while all traits share one `K`-dimensional latent block `Λ` — one factor
structure spanning, for example, a Poisson count trait, a Binomial binary
trait, and a Beta proportion trait, so that `correlation` returns a trait
correlation across those families on the common latent (link) scale.
`fit_mixed_gllvm` is a **named fitter, not a `fit_gllvm` family**: there is no
`Mixed` marker, and `fit_gllvm` does not dispatch to it.

`families` and `K` are required keywords; `links`, `N`, and the `*_init` seeds
are optional. v1 supports six per-trait families — `Normal()`, `Poisson()`,
`Binomial()`, `NegativeBinomial()`, `Gamma()`, `Beta()` — passed as a length-`p`
vector to `families`; anything else (e.g. `Ordinal()`) is rejected up front with
an `ArgumentError` naming the supported set, before any optimisation starts.
`Normal`, `NegativeBinomial`, `Gamma`, and `Beta` traits each carry one scalar
dispersion, optimised on the log scale and reported back on the natural scale in
`fit.dispersion` (`NaN` for `Poisson` and `Binomial` traits, which carry none);
`fit.disp_index` records which packed slot each one occupies. `links` defaults
to each family's canonical link if omitted. `N` (Binomial trial counts, `p×n`)
defaults to all-ones. The outer gradient is a direct `ForwardDiff` gradient
taken straight through the per-site Laplace mode solve, not a hand-coded
analytic kernel. There is no `X`-covariate keyword on this route.

Note that the family markers (`Normal`, `Poisson`, …) are `Distributions` types
that GLLVModels.jl uses internally but does not re-export, so a mixed-family script
needs `using Distributions` alongside `using GLLVModels`; the links
(`IdentityLink`, `LogLink`, `LogitLink`, …) are exported by GLLVModels.jl.

Unlike the single-family extractors elsewhere on this page, the mixed
extractors — `correlation`, `sigma_y_site`, `communality`, `getLV`, `predict`,
`link_residual` — all take **`(fit, Y)`**, not `fit` alone. `MixedFamilyFit`
stores only the parameters (`β`, `Λ`, `families`, `links`, `dispersion`, plus
the usual `loglik` / `converged` / `iterations`), not the latent scores, so each
extractor re-solves the per-site Laplace mode from `Y` and works forward from
there. Mixed-family fits have **no confidence-interval engine yet**: no
`MixedFamilyFit` method appears in the `confint` dispatch, and the R-bridge CI
route skips such fits with an explicit "not routed" note rather than returning
intervals.

## Two-part and mixture families (occurrence/zero × value)

Two-part and mixture families model a response with a point mass at zero (or at
the boundary) plus a distribution over the non-zero, count, or continuous part.
The hurdle, delta, and zero-inflated fits share a single latent `z` that loads on
the value/count part (`Λ_c`); the occurrence / zero-inflation part is a
per-species intercept (`β_z`, i.e. `Λ_z = 0`). Each has a dedicated fitter:

```julia
fit = fit_delta_lognormal_gllvm(Y; K = 2)   # Y ≥ 0; positive part lognormal, log-SD σ
fit = fit_delta_gamma_gllvm(Y;     K = 2)   # Y ≥ 0; positive part Gamma, shape α
fit = fit_gllvm(Y; family = BetaHurdle(), K = 2)     # proportions; occurrence × positive Beta
fit = fit_gllvm(Y; family = BetaHurdle(80.0), K = 2) # same — marker φ never read
# named fitter remains: fit_beta_hurdle_gllvm
fit = fit_gllvm(Y; family = HurdlePoisson(), K = 2)  # counts; occurrence × zero-truncated Poisson
# named fitter remains: fit_hurdle_poisson_gllvm
fit = fit_gllvm(Y; family = HurdleNB(), K = 2)       # counts; occurrence × zero-truncated NB2
fit = fit_gllvm(Y; family = HurdleNB(99.0), K = 2)   # same — marker r never read
# named fitter remains: fit_hurdle_nb_gllvm
fit = fit_zip_gllvm(Y;             K = 2)   # counts; structural zero × Poisson
fit = fit_zip_gllvm_cov(Y; X = X,  K = 2)   # ZIP + shared site-X (separate γz/γc; Λz=0)
fit = fit_zinb_gllvm(Y;            K = 2)   # counts; structural zero × NB2, shared r
fit = fit_zinb_gllvm_cov(Y; X = X, K = 2)   # ZINB + shared site-X (separate γz/γc; Λz=0; shared r)
fit = fit_zib_gllvm(Y;             K = 2, N = N)  # binomial counts; structural zero × Binomial(N, μ)
fit = fit_gllvm(Y; family = OrderedBeta(), K = 2)          # proportions; 0/1 masses + Beta interior
fit = fit_gllvm(Y; family = OrderedBeta(0.0, 2.0, 3.0), K = 2)  # same — marker tags never read
# named fitter remains: fit_ordered_beta_gllvm
```

All two-part fitters (including the `_cov` variants) accept the
`hessian = :observed | :fisher` curvature selector introduced for the one-part
families. Honest scope: the observed count-part weight is currently implemented
only for DeltaGamma, so for every other two-part family the two selectors
produce the identical objective until their observed weights land (the recorded
two-part curvature gap); the kwarg is exposed now as the measurement
prerequisite for closing that gap.

**Hurdle vs zero-inflated.** A *hurdle* model treats every zero as a
non-occurrence and the positive part as a **zero-truncated** count. A
*zero-inflated* model mixes a structural-zero process with an **ordinary** count
that can itself produce zeros: `P(y=0) = π + (1−π)·P_count(0)`. ZIP → Poisson as
the zero-inflation `π → 0`; ZINB → ZIP as `r → ∞`.

`predict` exposes the parts: `:occurrence` / `:zeroinfl` (the Bernoulli
probability), `:positive` / `:mean` (the value-part mean), and `:response` (the
unconditional mean). `residuals` gives randomized-quantile (Dunn–Smyth) residuals
under the correct two-part CDF.

### ZIP / ZINB / ZIB — Julia-beyond, evidenced by recovery not parity

`fit_zip_gllvm`, `fit_zinb_gllvm`, and `fit_zib_gllvm` are **Julia-beyond**
capabilities: no `gllvmTMB` twin exists for any of the three at the frozen
0.7.0 oracle, so there is nothing to run a parity comparison against (the
maintainer has since decided the R side will gain `zip`/`zinb`/`zib` as
native families, planned on the R side, no date set). The evidence backing
these three fitters is therefore an **ADEMP simulation-based recovery study**,
not a parity receipt (K = 1 latent factor,
intercept-only zero-inflation, i.e. `Λ_z = 0` by construction; 500 seeds per
cell):

| family | p | n | conv (n) | conv rate (MCSE) | βz bias mean/med | βz RMSE mean/med | βc bias mean/med | βc RMSE mean/med | Cerr med/p90 | fit_s med/max |
|---|---|---|---|---|---|---|---|---|---|---|
| zip  | 5  | 50  | 500/500 | 100.0% (0.00pp) | −1.213 / −0.342 | 2.834 / 0.901 | −0.177 / −0.173 | 0.467 / 0.455 | 1.245 / 2.035 | 1.09 / 4.69 |
| zip  | 5  | 200 | 500/500 | 100.0% (0.00pp) | 0.001 / 0.056  | 0.486 / 0.356 | −0.033 / −0.032 | 0.179 / 0.173 | 0.466 / 0.774 | 4.19 / 10.11 |
| zip  | 25 | 50  | **175/500** | **35.0% (2.13pp)** | −2.463 / −0.797 | 7.155 / 3.857 | −0.059 / −0.061 | 0.367 / 0.365 | 1.034 / 1.358 | 68.3 / 173.2 |
| zip  | 25 | 200 | 481/500 | 96.2% (0.86pp) | 0.009 / 0.014  | 0.568 / 0.365 | 0.147 / 0.000 | 0.307 / 0.167 | 0.313 / 0.935 | 172.2 / 932.5 |
| zinb | 5  | 50  | 500/500 | 100.0% (0.00pp) | −0.681 / 0.015 | 2.229 / 0.908 | −0.142 / −0.119 | 0.605 / 0.580 | 1.889 / 3.022 | 3.26 / 12.76 |
| zinb | 5  | 200 | 496/500 | 99.2% (0.40pp) | −0.349 / 0.003 | 1.238 / 0.515 | −0.079 / −0.076 | 0.305 / 0.298 | 0.833 / 1.302 | 9.17 / 43.18 |
| zinb | 25 | 50  | **350/500** | **70.0% (2.05pp)** | −0.981 / −0.419 | 4.457 / 3.487 | 0.009 / 0.004 | 0.468 / 0.458 | 1.483 / 1.906 | 129.8 / 373.7 |
| zinb | 25 | 200 | 493/500 | 98.6% (0.53pp) | 0.237 / 0.131 | 0.493 / 0.448 | 0.262 / 0.036 | 0.434 / 0.239 | 0.435 / 0.951 | 162.3 / 1515.2 |
| zib  | 5  | 50  | 500/500 | 100.0% (0.00pp) | 0.024 / 0.049 | 0.373 / 0.318 | −0.032 / −0.017 | 0.269 / 0.220 | 0.911 / 2.319 | 0.63 / 4.97 |
| zib  | 5  | 200 | 500/500 | 100.0% (0.00pp) | 0.079 / 0.082 | 0.177 / 0.177 | 0.002 / 0.003 | 0.102 / 0.099 | 0.408 / 0.611 | 2.27 / 6.41 |
| zib  | 25 | 50  | 500/500 | 100.0% (0.00pp) | −0.014 / −0.010 | 0.357 / 0.338 | −0.025 / −0.024 | 0.231 / 0.227 | 1.009 / 1.425 | 12.4 / 58.2 |
| zib  | 25 | 200 | 500/500 | 100.0% (0.00pp) | 0.028 / 0.030 | 0.172 / 0.170 | −0.009 / −0.006 | 0.116 / 0.105 | 0.297 / 0.444 | 55.4 / 686.8 |

**Small-n limitation, stated plainly.** `zib` recovers cleanly across the
whole grid (100.0% convergence in all four cells). `zip` and `zinb` recover
well except at the smallest-n/largest-p corner tested: **zip at p=25, n=50
converges only 35.0% of the time**, and among the fits that do converge the
zero-inflation intercept `βz` is badly biased (median −0.80, RMSE median
3.86); **zinb at the same corner converges 70.0% of the time**, with a
similarly degraded `βz`. The count/conditional part (`βc`) and the
rotation-invariant loading crossproduct stay reasonable at that corner — the
difficulty is concentrated in the zero-inflation intercept, not a global fit
collapse. Treat `zip`/`zinb` at `p ≈ 25, n ≈ 50` as a documented limitation,
not a supported capability. This study used K = 1 and intercept-only
zero-inflation only, evaluated point-estimate bias/RMSE and the fitter's own
convergence diagnostic — **no coverage or SE evaluation was done**, so it says
nothing about interval calibration for any of the three families.

## Extractors

The same post-fit extractors (`communality`, `correlation`, `sigma_y_site`, …)
work for all implemented families:

```julia
communality(fit)   # shared-variance fraction per response
correlation(fit)   # cross-response correlation matrix
getLV(fit)         # latent variable scores (sites × K)
```

See [Working with a fit](working-with-a-fit.md) for the full extractor reference.

See also: [Get started](quickstart.md) · [Covariance and correlation](covariance-correlation.md) · [Reference](api.md).
