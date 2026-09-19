# Structured-term fitting

`fit_gaussian_structured` is a public entry point for GLLVModels.jl's
structured-covariance source grammar — the same
`indep`/`dep`/`scalar`/`kernel_indep`/`kernel_dep`/`kernel_scalar`/
`kernel_latent` term vocabulary that `fit_gaussian_sources` already fits, now
reachable without hand-assembling `SourceCovariance` objects yourself.

## Why a `structure=` keyword, not a formula macro

StatsModels' `@formula` macro parses its right-hand side into `Term`s at
macro-expansion time, and it has no hook for the `lhs | group`-headed call
syntax these structured terms use (`indep(0 + trait | g)`, `kernel_latent(g,
K = K, d = 2)`, ...) — the bar is a parse error before any GLLVModels code runs.
`structure=` sidesteps this: each term is a raw, unevaluated `Expr` the caller
quotes with `:(...)`, and GLLVModels's own recognizer walks the `Expr` tree. A
macro front door that lets you write `structure(indep(0 + trait | g))`
directly, without quoting, is a separate, later, maintainer-approved grammar
decision — this wrapper does not attempt it.

## Scope

Gaussian only: `family` must be `Normal()`, matching `fit_gaussian_sources`
(which takes no `family` argument at all). Any other value raises a named
`ArgumentError`. Every error the underlying recognizer/gate pipeline raises —
augmented-LHS rejection, non-literal flag rejection, an unknown term name, a
Step-4 mutual-exclusion violation, a non-PD kernel matrix — surfaces
unchanged through this public wrapper; it is a thin, documented pass-through,
not a re-implementation.

## A first structured fit

This small Gaussian example adds one independent grouped term. `Y` has traits
in rows and observations in columns; `g` identifies the group for each
observation.

```@example structured_term_first_fit
using GLLVModels, Random
rng = MersenneTwister(70100)
Y = randn(rng, 3, 24)
data = (g = repeat(1:6; inner = 4),)
fit = fit_gaussian_structured(Y, data;
    structure = [:(indep(0 + trait | g))], sigma_eps_fixed = 0.5)
isfinite(fit.loglik)
```

The recognizer internals this wrapper drives (`_recognize_source_term`,
`_source_term_covariance`, `SourceTermSpec`, and friends) are internal
and documented on the [Low-level reference](low-level-reference.md) page —
they are not part of the public contract and can change without notice.

See also: [Structured dependence](structured-dependence.md) ·
[Mathematical model](model.md).
