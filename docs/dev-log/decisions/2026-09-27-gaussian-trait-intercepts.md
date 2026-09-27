# Public Gaussian route estimates trait intercepts when X is omitted

Status: PROPOSED (draft PR). Changes healthy-fit results; needs maintainer
sign-off before merge.

## Finding

Lane auto-d-20260926 cross-check, 2026-09-27. `fit_gllvm(Y; family = Normal(), K)`
with no `X` routed to `fit_gaussian_gllvm`, whose docstring defines `X = nothing`
as a zero mean. So the public Gaussian route fitted no trait intercepts:
`fit.pars.β == Float64[]` and `_nparams` counted `q = 0`.

Reproduction (`Random.seed!(3)`, p = 6, n = 100,
`Y0 = 0.8 .* randn(p,1) * randn(1,n) .+ randn(p,n)`, K = 1):

| data | before: logLik, length(β) | after: logLik, length(β) |
|---|---|---|
| Y0 | -962.334, 0 | -959.151, 6 |
| Y0 .+ 5 | -1297.089, 0 | -959.151, 6 |

gllvmTMB's `value ~ 0 + trait + latent(0 + trait | unit, d, unique = FALSE)` fits
one intercept per trait. The two had agreed only because the test data had mean 0.

## Was the zero mean intended?

For `fit_gaussian_gllvm`, yes: its docstring says so, and the R bridge centres Y
before calling it (`src/bridge.jl`, "Preserve the no-X Gaussian bridge
convention"). For the public `fit_gllvm` route, no evidence of intent was found:

- every non-Gaussian family on `fit_gllvm` estimates per-trait intercepts with no X;
- `fit_gllvm(...; family = Normal(), pervar = true)` with no X uses the trait
  means as intercepts (`fit_gaussian_pervar_gllvm` docstring);
- `2026-08-31-fixed-residual-unique-gaussian.md` records "X=nothing means trait
  intercepts" for that route;
- no docstring tells users to centre Gaussian data before `fit_gllvm`;
- no decision in `docs/design/`, `docs/dev-log/decisions/` or the maintainer's
  notes chooses a zero mean for the public route.

## Contract

| Call | Mean |
|---|---|
| `fit_gllvm(Y; family = Normal(), K)`, no X | one estimated intercept per trait |
| `gllvm(@formula(y ~ 1), Y, data; family = Normal(), K)` | same as above |
| `fit_gllvm(...; X = X)` | `X` is the complete mean, unchanged |
| `fit_gaussian_gllvm(Y; K)` | zero mean, unchanged (bridge relies on it) |
| `lambda_constraint = M` | zero mean, unchanged (Stage 1 requires X = nothing) |

The intercepts use the design `X[t, s, t] = 1` and are optimised jointly with
`Λ` and `σ_eps` by the existing fitter. With complete data the ML intercepts are
the trait means for any covariance, because every site shares the design `I_p`.
They are counted in `_nparams`. The added count is the same for every K, but
the log-likelihoods change: under the old zero-mean model a latent factor could
absorb the trait means. So `select_lv` choices on uncentred data can move.

The fit records `pars.mean_design = :trait_intercepts`. `_mean_X(fit, X, n)`
returns the intercept design when a tagged fit is used without `X`. It is
called in `_fitted_mean` (`getLV`, `predict`, `residuals`), `simulate`,
`confint`/`vcov`, `_confint_reconstruct_nll` (diagnostics, derived Wald),
`profile_ci`, `bootstrap_ci`, `bootstrap_ci_derived` and `profile_ci_derived`
(which serves `loading_profile_exploratory` and the variance profiles). Without
these calls, `confint`, `vcov`, `confint_inspect` and
`loading_profile_exploratory` returned silently wrong intervals for a tagged fit,
and the bootstraps and `check_gllvmTMB` raised errors.

`cv_gllvm(...; family = Normal())` refits through the same intercept route. The
random-cell imputation now adds the fitted intercept to the latent prediction
for those fits.

## Not covered

- `gllvm(@formula(y ~ x), ...; family = Normal())` builds a site-only design
  with no trait intercepts. That is the same class of gap, but fixing it changes
  what a formula design means, so it is left for a separate decision.
- CHANGELOG.md is leased by lane auto-d; the entry is added at merge time.
