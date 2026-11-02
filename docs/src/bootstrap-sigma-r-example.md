# Worked example: `bootstrap_Sigma_r`

This page runs `bootstrap_Sigma_r` end to end on a small simulated data set:
the call, the fields of the returned `BootstrapSigmaR`, how to read the bounds
and `n_effective`, the warnings it can raise, and the fits it refuses. Every
code block below is executed when the documentation is built, so the printed
output is the real output.

!!! warning "These numbers are for illustration only"
    The example uses `n_boot = 39` so that the page builds quickly. That is the
    smallest value for which a 95% percentile interval can reach its nominal
    coverage, and it is far too small for reporting. Use the default
    `n_boot = 999` or more for real work. No output of `bootstrap_Sigma_r`
    has been compared with R's `bootstrap_Sigma()`; this page shows how the
    Julia function behaves, not that it agrees with R.

## 1. Simulate data and fit the model

Six traits (rows), 100 observations (columns), two shared patterns, and a
separate residual variance for each trait. The fit uses `has_diag = true`, so
that `communality` is informative (for a fit without `has_diag` the
communality is exactly 1 for every trait).

```@example bsig
using GLLVModels, Distributions, Random, Statistics, Logging

# Run a call with the log silenced, so that warnings are not interleaved with the
# printed values. Section 5 shows the messages explicitly.
quiet(f) = with_logger(f, NullLogger())

Random.seed!(20261102)
p, n, K = 6, 100, 2
Λ_true = [0.9 0.0; 0.7 0.3; 0.5 0.6; 0.2 0.8; 0.0 0.7; 0.4 0.4]
ψ_true = [0.30, 0.40, 0.35, 0.30, 0.45, 0.50]
y = Λ_true * randn(K, n) .+ sqrt.(ψ_true) .* randn(p, n)
y .-= mean(y; dims = 2)

fit = quiet(() -> fit_gaussian_gllvm(y; K = K, has_diag = true))
fit.converged
```

With one observation per column, `fit_gaussian_gllvm(...; has_diag = true)`
prints a one-time warning that the between-unit and within-unit variances are
not separately identified. The total covariance `Σ_B = ΛΛ' + diag(σ²_B +
σ²_W + σ²_eps)` is identified, and that total is what `bootstrap_Sigma_r`
reports; do not interpret the split.

## 2. The call

```@example bsig
res = quiet() do
  bootstrap_Sigma_r(fit;
    n_boot   = 39,
    seed     = 1,                       # one seeded stream; same seed, same result
    progress = false,                   # otherwise one @info line per replicate
    level    = :unit,                   # a GllvmFit has one tier, called :unit (= :B)
    what     = (:Sigma, :R, :communality),
    n_sites  = n)                       # a GllvmFit does not store n; pass it (or y)
end
res
```

The printed summary shows only the diagonal of each matrix summary. The full
matrices are in `res.point_est`, `res.ci_lower` and `res.ci_upper`.

`n_sites` (or `y`, a `p x n` matrix of which only the number of columns is
read) is required for a `GllvmFit`, because the fit does not store the number
of observations. Passing `level = :unit` and an explicit `what` avoids the two
informational messages the defaults produce (see section 5).

## 3. The returned fields

```@example bsig
fieldnames(BootstrapSigmaR)
```

```@example bsig
(res.level, res.what, res.n_boot, res.n_failed, res.conf, res.ci_method,
 res.coverage_ceiling)
```

- `point_est`, `ci_lower`, `ci_upper` are `NamedTuple`s with one entry per
  requested summary and tier, named `Sigma_B`, `R_B` and `communality_B`.
  `Sigma_B` and `R_B` are `p x p` matrices; `communality_B` is a length-`p`
  vector.
- `n_effective` is a `NamedTuple` with entries for the vector summaries only
  (here `communality_B`). Matrix summaries have no `n_effective`; this mirrors
  R and is kept on purpose.
- `n_failed` counts replicates whose refit threw or did not converge.
- `coverage_ceiling` is `(n_boot - 1) / (n_boot + 1)`: the highest coverage a
  percentile interval from `n_boot` draws can reach.
- `draws` is `nothing` unless you pass `keep_draws = true`. `boot_median` is
  always an empty `NamedTuple` for these Gaussian fits.
- `level` holds the internal tier names (`:B`, `:W`, `:phy`), never `:unit`
  or `:unit_obs`.

```@example bsig
(keys(res.point_est), size(res.point_est.Sigma_B), keys(res.n_effective),
 res.draws, res.boot_median)
```

## 4. Reading the bounds and `n_effective`

Bounds are type-7 percentiles (the default of `Statistics.quantile`) of the
finite bootstrap draws, at `(1 - conf) / 2` and `1 - (1 - conf) / 2`.

```@example bsig
for i in 1:p
    println("trait ", i, ": communality ",
            round(res.point_est.communality_B[i]; digits = 3),
            "  [", round(res.ci_lower.communality_B[i]; digits = 3), ", ",
            round(res.ci_upper.communality_B[i]; digits = 3), "]",
            "   n_effective = ", res.n_effective.communality_B[i])
end
```

`n_effective[i]` is the number of replicates that contributed a finite draw to
entry `i`. A failed replicate is left out, so here every entry should equal
`n_boot - n_failed`:

```@example bsig
(all(res.n_effective.communality_B .== res.n_boot - res.n_failed),
 res.n_boot - res.n_failed)
```

A few things to keep in mind when reading the intervals:

- The estimate is the original fit's value, not a bootstrap mean, and it is not
  forced to lie inside its interval. Check it:
  `all(res.ci_lower.Sigma_B .<= res.point_est.Sigma_B .<= res.ci_upper.Sigma_B)`.
- A bound at, or numerically next to, a boundary (a communality bound of about
  `1`, such as trait 1 above, or a correlation bound of about `±1`) means that
  some replicates landed on the boundary of the parameter space. With so few
  draws this is common.
- Correlation bounds in `R_B` are clamped to `[-1, 1]` only when they overshoot
  by rounding error (at most `1e-8`); a larger excursion is left visible.
- There is no minimum-survivor rule: if most replicates fail you still get
  percentiles, from fewer draws. Watch `n_failed` and the warnings below.

```@example bsig
all(res.ci_lower.Sigma_B .<= res.point_est.Sigma_B .<= res.ci_upper.Sigma_B)
```

```@example bsig
res.point_est.R_B[1, 2], res.ci_lower.R_B[1, 2], res.ci_upper.R_B[1, 2]
```

## 5. Warnings and messages

Warnings go to the Julia log. The block below uses a small logger that prints
only the level and the message, so that they show up on this page. Two calls
with different arguments:

```@example bsig
struct MessageLogger <: AbstractLogger end     # prints "Level: message" only
Logging.min_enabled_level(::MessageLogger) = Logging.Info
Logging.shouldlog(::MessageLogger, args...) = true
Logging.catch_exceptions(::MessageLogger) = false
Logging.handle_message(::MessageLogger, level, message, args...; kwargs...) =
    println(level, ": ", message)

logged(f) = with_logger(f, MessageLogger())

logged() do
    # (a) default `level`: an @info names the tiers a GllvmFit does not have.
    bootstrap_Sigma_r(fit; n_boot = 39, seed = 1, progress = false,
                      what = :Sigma, n_sites = n)
    # (b) n_boot below min_boot = ceil(2 / (1 - conf)) - 1 = 39 for conf = 0.95:
    #     the coverage ceiling warning.
    bootstrap_Sigma_r(fit; n_boot = 19, seed = 1, progress = false,
                      level = :unit, what = :Sigma, n_sites = n)
end
```

- `n_boot` below 999 gives "percentile bounds are noisy". This is the usual
  warning in this page.
- `n_boot` below `min_boot` gives the coverage-ceiling warning instead (never
  both): with `n_boot = 19` the best achievable coverage is `0.9`, not `0.95`.
- If any replicate is lost, a warning names the count out of `n_boot` and says
  that the bounds rest on the survivors (the count in `n_failed` is the failed
  refits; a replicate whose summaries could not be extracted is lost too but
  is not in `n_failed`). If more than 20% of replicates are lost for a vector
  summary, an additional "attrition" warning names the worst one.
- With `progress = true` (the default) one `@info` line is printed per
  replicate, about 1000 at the default `n_boot`.
- For a fit without `has_diag`, `communality_B` is exactly 1 with a zero-width
  interval and an `@info` says so.

## 6. What it refuses

Anything without a verified simulate-and-refit route throws an
`ArgumentError` before any simulation is run. The message names the
supported alternative where one exists.

```@example bsig
function refusal(f)
    try
        quiet(f)
        "no error"
    catch e
        sprint(showerror, e)
    end
end

fitW = quiet(() -> fit_gaussian_gllvm(y; K = 1, K_W = 1))          # a W tier
M = fill(NaN, p, 1); M[1, 1] = 1.0
fitpin = quiet(() -> fit_gaussian_gllvm(y; K = 1, lambda_constraint = M))   # a pinned loading
Yc = Float64.(rand(Poisson(3.0), p, n))
fitpois = quiet(() -> fit_gllvm(Yc; family = Poisson(), K = 1))    # a non-Gaussian fit

cases = [
    "K_W > 0"            => () -> bootstrap_Sigma_r(fitW; n_boot = 39, progress = false, n_sites = n),
    "pinned loadings"    => () -> bootstrap_Sigma_r(fitpin; n_boot = 39, progress = false, n_sites = n),
    "Poisson fit"        => () -> bootstrap_Sigma_r(fitpois; n_boot = 39, progress = false, n_sites = n),
    "n_cores = 2"        => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, n_sites = n, n_cores = 2),
    "what = :cross_corr" => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, n_sites = n, what = :cross_corr),
    "level = :phy"       => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, n_sites = n, level = :phy),
    "no n_sites"         => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, level = :unit),
]
for (label, f) in cases
    println(label, ":\n    ", refusal(f), "\n")
end
```

Summary of what is supported:

| Fit | Result |
|---|---|
| `GllvmFit`, no W tier, no pins, no phylogenetic block, no retained Gaussian record | tier `:B` only: `Sigma`, `R`, `communality` |
| `TwoLevelFit` (pass `individual`) | tiers `:B` and `:W`, plus `ICC_site` |
| `GllvmFit` with `K_W > 0`, with pinned loadings, or with a phylogenetic block | refused |
| non-Gaussian fits, `PrecisionMultivariateFit`, `JointPhyloGroupedGaussianFit`, any other type | refused |
| `n_cores > 1` | refused |

`cross_corr` is never available (R needs a multinomial trait), and `ICC` needs
a `TwoLevelFit` with both tiers; on a `GllvmFit` it is dropped with an
`@info`.

## Where next

[SE and profile machinery](se-profile-machinery.md) states what
`bootstrap_Sigma_r` is and is not (including the legacy `bootstrap_Sigma`,
which is a different estimand). [Confidence intervals](confidence-intervals.md)
and [Derived confidence intervals](derived-confidence-intervals.md) cover the
other interval routes.
