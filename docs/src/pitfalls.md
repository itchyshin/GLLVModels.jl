# Common pitfalls

A short guide to the things that most often trip people up when fitting GLLVMs.

## Latent factors are identified only up to rotation and sign

The loadings `Λ` and the factor scores are not unique: any rotation
`Λ → Λ R` (with `R` orthogonal) gives the same fit. So **do not compare raw
`Λ` across fits or against a known truth** — compare rotation-invariant
quantities instead:

- the model-implied covariance `Σ_y = Λ Λᵀ + Ψ` (`sigma_y_site`),
- per-response communalities (`communality`),
- cross-response correlations (`correlation`).

For a fixed, interpretable orientation, rotate the loadings (e.g. varimax)
*after* fitting.

## Choose the number of factors `K` deliberately

Too few factors miss real structure; too many invite over-fitting and weak
identifiability. Compare a few values of `K` by log-likelihood / information
criteria, and prefer the smallest `K` that captures the covariance you care
about.

## Check convergence

`fit_gaussian_gllvm` warm-starts from PPCA and usually converges in a step or
two; the non-Gaussian fitters use L-BFGS over a Laplace marginal. Always check
the `converged` flag — if it is `false`, try a different start or more
iterations. Standardising responses to a common scale helps the Gaussian path.

## Fewer sites than species (`n < p`)

The closed-form Gaussian fitter `fit_gaussian_gllvm` does not require
`n_sites ≥ p`. It requires the number of latent axes `K` (plus `K_W`) to be
below the rank of the data it sees, unless that rank is already `p`; otherwise
the Gaussian likelihood is unbounded and the fit throws an `ArgumentError` that
names the rank and `K`. When `X` is supplied the data are first residualised on
`X`, so per-trait intercepts centre them. With `n_sites < p` the rank is at
most `n_sites`, so a small `K` fits and `K ≥ n_sites` is refused; rank-deficient
data with `n_sites ≥ p` (for example a trait that is a sum of two others) are
refused for `K` at or above the rank. Every route built on the fitter applies
the same rule: `fit_gllvm(Y; family = Normal(), K)` (also through `@formula`
and `select_lv`), the masked / offset / `aghq` Gaussian route,
`family = Lognormal()` (a Gaussian fit to the centred `log(Y)`), and the R
bridge's `"gaussian"` and `"lognormal"` families. The Laplace-fitted families
(Poisson, binomial, negative binomial and the others) have no such condition
and fit with `n_sites < p`. Treat such fits as weakly identified: check
`converged`, expect binomial fits to reach the Laplace saturation region sooner
(the fit then warns and reports it through `saturation`), and prefer a small `K`.

## Use the O(p) path for large phylogenies

The dense Gaussian fit with a phylogenetic covariance is `O(p³)` and assumes
`n ≥ p`. For many species, use the O(p) phylogenetic fitter
[`fit_phylo_gaussian`](@ref), which scales linearly (≈ 0.8 ms per gradient at
p = 10,000) by never forming the dense `p×p` covariance.

## Binary data: watch for separation

With binary responses, a response that is all-0 or all-1 drives its intercept
to ±∞ (complete separation). `fit_binomial_gllvm` clamps the linear predictor
for numerical safety, but a non-convergence flag or an extreme intercept is a
sign to inspect that response.

See also: [Get started](quickstart.md) · [Response families](response-families.md) · [Reference](api.md).

## Loading StatsBase or MixedModels alongside GLLVModels breaks six verbs

GLLVModels.jl defines its own `confint`, `aic`, `bic`, `predict`, `fitted` and
`residuals`. It does not extend the StatsAPI generics of the same name, so if
you load another modelling package in the same session, Julia sees two
unrelated functions sharing one name and refuses to pick:

```julia
using GLLVModels, StatsBase

confint(fit, Y; method = :wald)
# WARNING: both StatsBase and GLLVModels export "confint";
#          uses of it in module Main must be qualified
# ERROR: UndefVarError: `confint` not defined
```

All six behave this way, and **`StatsBase` is not the only trigger** —
`MixedModels.jl` exports the same six, so the comparison workflow the README
suggests runs into it too. `DataFrames` and `Distributions` do not.

**The fix is to qualify the call:**

```julia
using GLLVModels, StatsBase

GLLVModels.confint(fit, Y; method = :wald)     # GLLVModels's interval machinery
GLLVModels.predict(fit, Y; type = :link)
GLLVModels.aic(fit)
```

Or import only what you need from the other package
(`using StatsBase: mean, sample`) so the names never collide.

Nothing is broken about the fit itself — this is a name-resolution problem, and
both functions remain reachable when qualified. The examples elsewhere in these
docs are written unqualified because they assume `using GLLVModels` alone.

!!! note "This is a known wart, not a design choice"
    Re-rooting these six onto the StatsAPI generics would remove the clash
    entirely and let generic ecosystem code work with a `GllvmFit`. That is an
    API change, so the present behaviour remains explicit and predictable.

The one genuinely deliberate name clash is `Multinomial`, which is documented
separately under [Response families](response-families.md) — always write
`GLLVModels.Multinomial()` for the family marker and `Distributions.Multinomial(...)`
for the count-vector law.
