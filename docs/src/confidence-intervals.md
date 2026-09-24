# Confidence intervals

GLLVModels.jl provides three interval constructions — **Wald**, **profile
likelihood**, and **parametric bootstrap** — for the Gaussian engine and the
admitted non-Gaussian CI rows. They quantify different approximations to
uncertainty; none by itself guarantees calibrated coverage for a particular fit
or data set.

## Non-Gaussian families — one entry point

For a fitted non-Gaussian model, all three methods are reached through a single
R-style call:

```julia
using GLLVModels, Distributions

fit = fit_gllvm(Y; family = Poisson(), K = 2)

confint(fit, Y; method = :wald)                          # observed-information Wald
confint(fit, Y; method = :profile,  parm = "beta[1]")    # profile-likelihood (LRT)
confint(fit, Y; method = :bootstrap, n_boot = 500)       # parametric bootstrap
```

`Y` is the same response matrix you fitted — it is needed to reconstruct the
marginal likelihood. The call returns a `NamedTuple` with `term`, `estimate`,
`lower`, `upper`, and `method` (plus method-specific extras below).

The generic entry point currently accepts the fitted types in its `_CIFit`
dispatch: ordinary family fits (including Poisson, Binomial, NB/NB1, GP1,
Beta, Gamma, Exponential, Tweedie, Beta-Binomial, row-random, lognormal,
truncated-count, and Student-t routes); two-part fits; grouped-dispersion and
grouped-dispersion-covariate fits; `OrdinalFit`, per-trait ordinal fits,
`MultinomialFit`, `GllvmCovFit`, `ZIPCovFit`, `ZINBCovFit`, `OrderedBetaFit`,
`QuadraticFit`, and `RowEffectFit`. Availability of a meaningful endpoint still
depends on the selected fit, term, convergence, and curvature diagnostics.
`ZIPCovFit` / `ZINBCovFit` need the design via `X`
(`confint(fit, Y; method=…, X=X)`); term names add free dual slopes
`gammaz[k]` / `gammac[k]` (`ZINBCovFit` also reports shared scalar `r` on the
log scale). `GaussianPerVarFit` is not in this dispatch and currently has no
public CI method.

### Term names

| Family group | Names |
|--------------|-------|
| GLM families | `beta[t]`, `Lambda[i,k]`, and a dispersion `r` / `phi` / `alpha` |
| Grouped NB2/NB1/Beta/Gamma | `beta[t]`, `Lambda[i,k]`, and group-level dispersion `r[g]` / `phi[g]` / `alpha[g]` |
| Two-part families | `betaz[t]` (occurrence / zero-inflation logits), `betac[t]` (value / count intercepts), `Lambda[i,k]`, and `sigma` / `alpha` / `r` |
| Ordinal (`OrdinalFit`, shared cutpoints) | `Lambda[i,k]`, `tau[c]` (cutpoints) |

Dispersion parameters are estimated on the log scale internally; their interval
**bounds are reported on the natural (positive) scale**.

`parm` subsets the terms: an exact name (`"beta[1]"`, `"r"`), a group (`"beta"`,
`"Lambda"`, `"betac"`, `"tau"`), or a vector of these.

## The three methods

### Wald — `method = :wald`

The Hessian of the negative Laplace log-likelihood is formed by **central finite
differences** at the MLE (the Laplace inner mode-finder is not forward-AD-
friendly, matching how the fitters themselves are optimised), then inverted for
the asymptotic covariance; `lower/upper = θ̂ ± z·SE`. Returns an extra
`pd_hessian::Bool` flagging whether the FULL joint observed information was
positive definite. Cheapest method; assumes approximate normality on the
working scale.

**Per-parameter degradation at a boundary (T14 F1, 2026-09-02).** If the fit flags a
grouped NB2/NB1/Beta/Gamma dispersion parameter at its `dispersion_boundary` (the
Poisson / near-Bernoulli / near-deterministic limit) — whether or not the joint
Hessian happens to pass a Cholesky — or the joint Hessian is not PD for any other
numerically flat direction, `pd_hessian` is `false`,
but the CI no longer NaNs *every* parameter. The known-boundary parameters
(plus any further direction needed for the remaining sub-Hessian to pass a
positive-definiteness check) are conditioned out; the rest of the parameters
(β, γ, Λ, and any non-boundary dispersion) get finite bounds from that reduced
Hessian's inverse. The extra `boundary_terms::Vector{String}` names exactly
which terms were conditioned out (empty when `pd_hessian == true`). This
mirrors R's `sdreport()`, which NaNs only the degenerate block rather than the
whole covariance matrix. Note this per-parameter degradation applies to the
family/grouped-dispersion route (`confint(fit::_CIFit, Y; ...)`,
`src/confint_family.jl`); the Gaussian path (`confint(fit::GllvmFit; ...)`,
`src/confint.jl`) is unchanged and still reports all-NaN on a non-PD Hessian.

### Profile likelihood — `method = :profile`

Uses a likelihood-ratio construction: the deviance `D(c) = 2(ℓ̂ − ℓ_p(c))` is
compared with a χ²₁ cutoff, and the reported interval is
`{c : D(c) ≤ qchisq(level, 1)}`. Each side is located by
**bracket-then-bisection**, re-optimising the other parameters at every
candidate. This can represent asymmetry that a local Wald approximation misses,
but it remains conditional on successful constrained refits and the usual
likelihood-ratio approximation; it is not an exact or coverage-guaranteed
interval. Returns a per-term `status` (`:profile` / `:partial` / `:failed`).
Use `profile_iterations`, `profile_g_tol`, `profile_max_expand`, and
`profile_max_bisect` to tune the constrained-refit and bracketing budget when a
profile canary needs tighter or cheaper refits.

### Parametric bootstrap — `method = :bootstrap`

Simulates `n_boot` datasets from the fitted model, refits each, and takes
percentile bounds. It can be useful where a local quadratic approximation is
unpersuasive, but it is not a gold standard or a coverage guarantee: results
also depend on the fitted model, the number of replicates, and successful
refits.

```julia
confint(fit, Y; method = :bootstrap, n_boot = 1000, parallel = true)
```

Set `parallel = true` to run replicates over `Threads.@threads`. **Each
replicate seeds its own RNG (`seed + b`)**, so the result is independent of
thread scheduling — multi-core and single-core give identical bounds. Returns an
extra `n_converged::Int` (replicates whose refit failed or changed dimension are
dropped). Inspect it before interpreting the percentile bounds. Start Julia
with `julia -t auto` to use multiple threads.

## Gaussian engine

The Gaussian fit keeps its own dedicated functions (it has the richest parameter
structure — `σ_eps`, between/within tiers, phylogenetic blocks):

```julia
fit = fit_gaussian_gllvm(y; K = 2)
confint(fit; y = y)                       # Wald (observed information)
profile_ci(fit, "sigma_eps"; y = y)       # profile likelihood
bootstrap_ci(fit; y = y, n_boot = 500)    # parametric bootstrap
```

and derived-quantity CIs (Σ_y entries, communality, correlation, phylogenetic
signal H²) via [`confint_derived`-family helpers](covariance-correlation.md).

### Confirmatory fits and `loading_profile` (D3 Stage 1)

`fit_gaussian_gllvm(y; K, lambda_constraint = M)` fits a **confirmatory**
ordinary Gaussian model: `M` is a `p × K` matrix of raw `Λ` values (`NaN` =
free, numeric = pinned at that value), mirroring R gllvmTMB's
`lambda_constraint = list(unit = M)`. `loading_profile(fit; y, ...)` then
profiles a grid over each **free** entry of such a fit, refitting with that
one entry additionally pinned at each grid value — the confirmatory mirror of
R's `loading_profile()`. It requires a confirmatory fit and refuses an
ordinary (unpinned) one; see `loading_profile_exploratory` for the existing
penalty-based profile CI on an unpinned fit's raw loadings.

```julia
fit = fit_gaussian_gllvm(y; K = 2, lambda_constraint = M)   # M: p × K, NaN = free
loading_profile(fit; y = y, n_grid = 11)
```

**Current scope.** Confirmatory loading profiles are available for the ordinary Gaussian latent-variable model only: no phylogenetic or diagonal terms and no fixed-effect covariates `X`. `lambda_constraint` cannot yet be combined with `aghq`, `mask`, `offset`, or predictor-informed latent scores; such fits are refused with an error. The profile grid spacing follows a Wald-standard-error heuristic rather than R's exact spacing rule, and no cross-package numeric comparison against R's `loading_profile()` output has been published yet.

## Predictor-informed latent-score effects

For fits with `X_lv`, `confint_lv_effects(fit, Y, X_lv)` targets the induced,
rotation-stable trait-effect matrix `B_lv = Lambda * alpha_lv'`. Wald,
profile-likelihood, and bootstrap intervals are native Julia uncertainty
routes for admitted ordinary `X_lv` fits; bootstrap remains a cost-bounded
diagnostic complement rather than the default engine.
The admitted ordinary set currently covers Gaussian, Poisson, Binomial, NB2,
Gamma, Beta, and shared-cutpoint Ordinal fits. Per-trait ordinal bridge
intervals, source-specific structural `X_lv`, mixed-family `X_lv`, and
response-mask `X_lv` intervals remain separate gates.

```julia
ci_all = confint_lv_effects(fit, Y, X_lv; method = :profile)
ci_some = confint_lv_effects(fit, Y, X_lv; method = :profile,
                             profile_indices = [2, 4])
```

`profile_indices` selects entries of `vec(B_lv)` in column-major order, matching
returned names such as `B_lv[2,1]` and `B_lv[4,1]`. It is intentionally only
accepted with `method = :profile`; Wald/bootstrap calls return their full
supported surface.

See also: [Response families](response-families.md) · [Working with a fit](working-with-a-fit.md) · [Reference](api.md).
