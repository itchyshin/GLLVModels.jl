# Gaussian formulas with covariates estimate trait intercepts and shared slopes

Status: PROPOSED (draft PR, stacked on the trait-intercept PR #519). Changes
healthy-fit results; needs maintainer sign-off before merge.

## Finding

`gllvm(@formula(y ~ x), Y, data; family = Normal(), K)` took the default
shared-variance branch of `src/formula.jl`. `_build_site_modelmatrix` drops the
constant term, and the branch passed the site-only design `X[t, s, k] = mm[s, k]`
to `fit_gaussian_gllvm`, which treats `X` as the complete mean. So the fit had one
shared slope per covariate and no trait intercepts, and shifting Y moved the
log-likelihood.

Reproduction (`MersenneTwister(11)`, p = 5, n = 90, K = 1, one covariate `temp`,
`Y1 = Y0 .+ (4:8)`):

| data | before: logLik, length(β) | after: logLik, length(β) |
|---|---|---|
| `Y0` | -400.115, 1 | equal for both, 6 |
| `Y1` | -749.358, 1 | equal for both, 6 |

## Intent

- The `gllvm` docstring says: "The intercept (`1`) is the engine's built-in
  per-species intercept; each covariate column ... becomes a coefficient shared
  across species." `fit_gaussian_gllvm` has no built-in intercept, so the
  Gaussian branch did not do what the docstring says.
- `2026-08-31-core070-pervar-formula.md` left this route unchanged on purpose
  for scope and wrote that its "broader legacy intercept documentation needs its
  own audit". This record is that audit.
- The other Gaussian formula routes (pervar, sources, grouping, phylo) already
  build trait intercepts plus shared slopes with `_pervar_formula_design`.
- The non-Gaussian `_cov` fitters (`fit_gllvm_cov`, `fit_nb_gllvm_grouped_cov`,
  ...) use per-species intercepts `β` and shared covariate coefficients `γ`.
- gllvmTMB's equivalent is `value ~ 0 + trait + x`: trait intercepts, one shared
  slope. No parity fixture exercises Gaussian `y ~ x`.

## Decision

The Gaussian shared-variance branch builds its design with
`_pervar_formula_design`, so it follows StatsModels' intercept and rank rules:

| formula | mean |
|---|---|
| `y ~ x`, `y ~ 1 + x` | trait intercepts + shared slope (changed) |
| `y ~ 0 + x`, `y ~ -1 + x` | shared slope, no intercepts (unchanged) |
| `y ~ 1` | trait intercepts (PR #519) |
| `y ~ 0` | zero mean (unchanged) |

Slopes stay **shared**, not per-trait, to match the docstring, the non-Gaussian
`_cov` routes, the other Gaussian formula routes and gllvmTMB's
`value ~ 0 + trait + x`. Per-trait slopes are a separate model
(`value ~ 0 + trait + (0 + trait):x` in gllvmTMB) and are not added here.
Explicit-`X` calls to `fit_gaussian_gllvm` are unchanged.

## Not covered

- Post-fit helpers (`getLV`, `predict`, `residuals`, `confint`, ...) take `X`
  from the caller. For a Gaussian formula fit with covariates the caller must
  rebuild the same design, and omitting it silently uses a zero mean. This was
  already true before this change; only the intercept-only fit is tagged
  (PR #519). Storing the design on the fit is left for a separate decision.
- CHANGELOG.md is leased by lane auto-d; the entry is added at merge time.
