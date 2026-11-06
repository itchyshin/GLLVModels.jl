# Worked example: `bootstrap_Sigma_r` at the phylogenetic tier

This page runs `bootstrap_Sigma_r` on a small simulated phylogenetic fit. It
covers the call, how to read the `Sigma_phy` and `communality_phy` bounds, the
message you get when the communality carries no information, the warnings for
lost replicates, and the refusals. A short second example uses a joint
phylogenetic plus grouped fit. Every code block is executed when the
documentation is built, so the printed output is the real output of that build.
The companion page for an ordinary `GllvmFit` is
[Worked example: bootstrap_Sigma_r](bootstrap-sigma-r-example.md).

!!! warning "What this page does and does not show"
    The evidence behind the `:phy` tier is structural plus Julia descriptive:
    the code was read against the R source and tested on constructed cases, and
    its failure rates were measured on Julia fits. Agreement with R has been
    shown only as a point-estimate check: `fit_precision_multivariate` was
    compared with R gllvmTMB on two data sets, and the differences were at most
    about 2e-6 (see the repository's check-log note `2026-11-02-bootstrap-sigma-phy-fuzz-and-r-point.md`,
    section "R point-estimate agreement"). No bootstrap interval from this
    function has been compared with an R bootstrap, and no coverage or recovery
    study has been run. The example uses `n_boot = 39` so that the page builds
    quickly; that is far too small for reporting, so use the default
    `n_boot = 999` or more. Failure counts in the examples below depend on the
    platform and Julia version (the fitter's convergence checks are
    gradient based), so the numbers on your build can differ from the numbers in
    the text.

## 1. Simulate data and fit the model

Three traits, 16 tips of a balanced tree, three observations per tip, one shared
phylogenetic pattern, and a phylogenetic unique variance for each trait. The
helper below draws the data from the model itself, using the tree's precision
matrix.

```@example bsigphy
using GLLVModels, Random, LinearAlgebra, Logging

# Run a call with the log silenced. Later sections show the messages explicitly.
quiet(f) = with_logger(f, NullLogger())

phy = PrecisionPhy(GLLVModels.random_balanced_tree(16; branch_length = 0.35))
sid = repeat(1:16, inner = 3)          # tip of each observation, in the tree's leaf order

function simulate_phy(rng, phy, sid; beta, loading, unique, residual)
    d = length(beta)
    F = cholesky(Symmetric(Matrix(phy.Q))).U
    g = F \ randn(rng, phy.n_aug, size(loading, 2))     # shared phylogenetic effects
    h = F \ randn(rng, phy.n_aug, d)                    # trait-specific phylogenetic effects
    Y = Matrix{Float64}(undef, d, length(sid))
    for o in eachindex(sid), t in 1:d
        node = phy.species_aug_id[sid[o]]
        Y[t, o] = beta[t] + sum(loading[t, k] * g[node, k] for k in axes(loading, 2)) +
                  (unique === nothing ? 0.0 : sqrt(unique[t]) * h[node, t]) +
                  sqrt(residual[t]) * randn(rng)
    end
    return Y
end

Y = simulate_phy(MersenneTwister(20261102), phy, sid;
    beta = [0.4, -0.3, 0.7], loading = reshape([0.9, -0.6, 0.7], 3, 1),
    unique = [0.40, 0.30, 0.35], residual = [0.10, 0.12, 0.10])

fit = quiet(() -> GLLVModels.fit_precision_multivariate(Y, phy;
        rank = 1, mode = :explicitunique, species_id = sid))
(typeof(fit), fit.converged, fit.mode, fit.rank, size(fit.response))
```

`mode = :explicitunique` gives each trait its own phylogenetic unique variance,
so `communality_phy` can be informative. Two cautions about the inputs.
`species_id` indexes tips in the tree's leaf order, not by name; a permuted map
is a valid index vector, so no check can catch it, and the fit would silently
describe the wrong model. Also, the fit stores its response, tree and mean
design, which is why `bootstrap_Sigma_r` needs no `y` or `n_sites` for this
class.

## 2. The call

```@example bsigphy
res = quiet() do
  bootstrap_Sigma_r(fit;
    n_boot   = 39,
    seed     = 1,                       # one seeded stream; same seed, same result on one platform
    progress = false,                   # otherwise one @info line per replicate
    level    = :phy,                    # the only tier this class has
    what     = (:Sigma, :R, :communality))
end
res
```

The printed summary shows only the diagonal of each matrix. The full matrices
are in `res.point_est`, `res.ci_lower` and `res.ci_upper`. The diagonal of
`R_phy` is 1 by construction, so read the off-diagonal entries of that matrix.

Each replicate draws a new dataset from the fitted model with every
phylogenetic effect redrawn, holds the loadings fixed, refits from a cold start,
and counts as lost if the refit throws or does not converge.

## 3. The returned fields

```@example bsigphy
fieldnames(BootstrapSigmaR)
```

```@example bsigphy
(res.level, res.what, res.n_boot, res.n_failed, res.conf, res.ci_method,
 res.coverage_ceiling)
```

```@example bsigphy
(keys(res.point_est), size(res.point_est.Sigma_phy), keys(res.n_effective),
 res.diagnostics)
```

The fields are the same as in the `GllvmFit` example, with these differences.
The entries are named `Sigma_phy`, `R_phy` and `communality_phy`, and
`res.level` is `[:phy]`. `n_effective` exists for `communality_phy` only.
`diagnostics` is empty for this class; it is filled only for the joint fit in
section 7.

## 4. Reading `Sigma_phy` and `communality_phy`

`Sigma_phy` is `ΛΛ' + diag(unique)`: the shared phylogenetic pattern plus the
trait-specific phylogenetic variances. It does not include the residual
variance. Check this against the fit's own fields:

```@example bsigphy
S = fit.loading * fit.loading' + Diagonal(fit.phylo_unique_variance)
(res.point_est.Sigma_phy ≈ S,
 res.point_est.communality_phy ≈ vec(sum(abs2, fit.loading; dims = 2)) ./ diag(S))
```

`communality_phy[i]` is the share of trait `i`'s phylogenetic variance that is
shared across traits, `(ΛΛ')ᵢᵢ / Σ_phyᵢᵢ`. Bounds are type-7 percentiles of the
finite draws, and the estimate is the original fit's value, not a bootstrap
mean:

```@example bsigphy
for i in 1:3
    println("trait ", i, ": communality ",
            round(res.point_est.communality_phy[i]; digits = 3),
            "  [", round(res.ci_lower.communality_phy[i]; digits = 3), ", ",
            round(res.ci_upper.communality_phy[i]; digits = 3), "]",
            "   n_effective = ", res.n_effective.communality_phy[i])
end
```

```@example bsigphy
println("point"); display(round.(res.point_est.Sigma_phy; digits = 3))
println("lower"); display(round.(res.ci_lower.Sigma_phy; digits = 3))
println("upper"); display(round.(res.ci_upper.Sigma_phy; digits = 3))
all(res.ci_lower.Sigma_phy .<= res.point_est.Sigma_phy .<= res.ci_upper.Sigma_phy)
```

```@example bsigphy
res.point_est.R_phy[1, 2], res.ci_lower.R_phy[1, 2], res.ci_upper.R_phy[1, 2]
```

Points to keep in mind:

- A communality upper bound of `1` means that some replicates put a trait's
  unique variance at zero, which is the edge of the parameter space. With 16
  tips and 39 replicates this is common, and it does not by itself mean
  something went wrong.
- Intervals are wide here because the data set is small. The page is about
  reading the output, not about how wide a real interval would be.
- There is no minimum-survivor rule. If most replicates fail you still get
  percentiles from the survivors, so read `n_failed` and `n_effective` together
  with the warnings in section 6.

## 5. A communality that is exactly 1

With `mode = :barelowrank` the phylogenetic tier is `ΛΛ'` with no diagonal, so
shared over total is 1 for every trait. The function says so in an `@info`, and
the interval has zero width. The block below uses a small logger that prints
the level and message of everything the call logged.

```@example bsigphy
struct MessageLogger <: AbstractLogger
    lines::Vector{String}
end
Logging.min_enabled_level(::MessageLogger) = Logging.Info
Logging.shouldlog(::MessageLogger, args...) = true
Logging.catch_exceptions(::MessageLogger) = false
Logging.handle_message(l::MessageLogger, level, message, args...; kwargs...) =
    push!(l.lines, string(level, ": ", message))

# Run `f`, then print the level and message of everything it logged.
function logged(f)
    lg = MessageLogger(String[])
    with_logger(f, lg)
    foreach(println, lg.lines)
end

fitb = quiet(() -> GLLVModels.fit_precision_multivariate(Y, phy;
        rank = 1, mode = :barelowrank, species_id = sid))
resb = nothing
logged() do
    global resb = bootstrap_Sigma_r(fitb; n_boot = 39, seed = 1, progress = false,
                                    level = :phy, what = (:Sigma, :communality))
end
(fitb.phylo_unique_variance, resb.point_est.communality_phy,
 resb.ci_lower.communality_phy, resb.ci_upper.communality_phy)
```

Refit with `mode = :explicitunique` if you want a communality that carries
information.

## 6. Warnings and messages

Two calls with different arguments. The first leaves `level` at its default,
which logs an `@info` that the `:B` and `:W` tiers do not exist for this class.
The second uses `n_boot = 19`, below `min_boot = ceil(2 / (1 - conf)) - 1 = 39`
for `conf = 0.95`, which gives the coverage-ceiling warning instead of the usual
"noisy" one (never both).

```@example bsigphy
logged() do
    bootstrap_Sigma_r(fit; n_boot = 39, seed = 1, progress = false, what = :Sigma)
    bootstrap_Sigma_r(fit; n_boot = 19, seed = 1, progress = false,
                      level = :phy, what = :Sigma)
end
```

A real refit can fail to converge, and the failure rate depends on the data and
the platform, so this page cannot promise a failure in any one build. To show the
warnings, the block below forces every fourth refit to stop after one iteration,
so it is not converged. It uses the internal `_refit` keyword, which exists for
the package's own tests and is not part of the public interface.

```@example bsigphy
calls = Ref(0)
flaky(Y, phy; kw...) = begin
    calls[] += 1
    iters = calls[] % 4 == 0 ? 1 : 400
    quiet(() -> GLLVModels.fit_precision_multivariate(Y, phy; kw..., iterations = iters))
end
resf = nothing
logged() do
    global resf = bootstrap_Sigma_r(fit; n_boot = 39, seed = 1, progress = false,
                                    level = :phy, what = (:Sigma, :communality),
                                    _refit = flaky)
end
(resf.n_failed, resf.n_effective.communality_phy)
```

- The "lost" warning names the count out of `n_boot` and says the bounds rest on
  the survivors (and are all NaN if every replicate was lost).
- If a vector summary keeps under 80% of the replicates, an "attrition" warning
  names the worst one.
- A converged refit whose summaries come out NaN is not counted in `n_failed`;
  only the attrition warning then shows it.
- If the original fit did not converge, the bootstrap still runs without a
  warning about the point-estimate fit. Check `fit.converged` yourself.
- The fitter may also print its own optimiser warnings for individual refits.
- `progress = true` (the default) logs one `@info` per replicate.

## 7. What it refuses

Arguments that would be silently ignored, tiers the fit does not have, and
unavailable summaries throw an `ArgumentError` before any simulation is run.

```@example bsigphy
function refusal(f)
    try
        quiet(f)
        "no error"
    catch e
        sprint(showerror, e)
    end
end

cases = [
  "level = :B"  => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, level = :B),
  "what = :ICC" => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, level = :phy, what = :ICC),
  "n_sites"     => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, level = :phy, n_sites = 48),
  "y"           => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, level = :phy, y = Y),
  "n_cores = 2" => () -> bootstrap_Sigma_r(fit; n_boot = 39, progress = false, level = :phy, n_cores = 2),
]
for (label, f) in cases
    println(label, ":\n    ", refusal(f), "\n")
end
```

Two details of the refusal text. The `what = :ICC` message is written for a
`GllvmFit` and names that class even though the fit here is a
`PrecisionMultivariateFit`. And `what` still accepts `:cross_corr`, but no
`cross_corr` entry is ever produced (`keys(res.point_est)` shows what you got).

## 8. A joint phylogenetic plus grouped fit

`JointPhyloGroupedGaussianFit` adds ordinary independent grouping terms to the
phylogenetic model. `bootstrap_Sigma_r` supports it at the `:phy` tier only. The
ordinary terms are redrawn in each replicate, but they are not reported: the
`:B` and `:W` tiers are dropped. The public route to this fit class is
`fit_gllvm(Y; family = Normal(), phylo = ..., grouping = ...)`; this page calls
the fitter that route wraps, `fit_joint_phylo_grouped_gaussian`, directly, with
its keywords spelled out.

```@example bsigphy
phyj = PrecisionPhy(GLLVModels.random_balanced_tree(8; branch_length = 0.35))
sidj = repeat(1:8, inner = 6)          # 48 observations
unit = repeat(1:12, inner = 4)         # 12 groups of 4 observations

rngj = MersenneTwister(20261103)
Yj = simulate_phy(rngj, phyj, sidj; beta = [0.4, -0.3, 0.7],
        loading = reshape([0.9, -0.6, 0.7], 3, 1),
        unique = [0.40, 0.30, 0.35], residual = [0.30, 0.40, 0.35])
Yj .+= (sqrt.([0.25, 0.20, 0.30]) .* randn(rngj, 3, 12))[:, unit]   # group effects

fitj = quiet(() -> GLLVModels.fit_joint_phylo_grouped_gaussian(Yj, phyj;
        rank = 1, phylo_mode = :explicitunique,
        terms = GroupingTerm[GroupingTerm(:unit; mode = :indep, common = false)],
        unit = unit, species_id = sidj))
(typeof(fitj), fitj.converged)
```

```@example bsigphy
resj = nothing
logged() do
    global resj = bootstrap_Sigma_r(fitj; n_boot = 39, seed = 1, progress = false,
                                    level = :phy, what = (:Sigma, :communality))
end
resj
```

```@example bsigphy
resj.diagnostics
```

`resj.diagnostics` is filled only for this class:

| Field | Meaning |
|---|---|
| `n_thrown`, `n_not_converged`, `n_non_identified` | Why replicates failed. Their sum is `n_failed`. A replicate whose variance components are not identified counts as a failure, which is stricter than R's rule (convergence only). |
| `n_converged` | Converged replicates, identified or not. |
| `n_hessian_not_pd` | Of those, how many have a Hessian that is not positive definite. They are kept, and an `@info` says so, because such replicates often sit on a variance boundary and can truncate interval ends. |
| `original_hessian_pd` | Whether the original fit's Hessian is positive definite. If `false`, a warning says so and the run proceeds. |

The share of converged replicates with a non-positive-definite Hessian can be
large on small data sets, as in the output above. Whether the original
Hessian is positive definite can differ between platforms for the same data.

Two further cautions. Convergence uses an absolute gradient tolerance from fixed
start values, so rescaling the same data, or the branch lengths of the tree, can
change how many replicates fail, and very large or very small scales can make the
original fit itself refused. And the identification check can report a group term
as identified when its variance cannot in fact be identified (for example a term
with only one or two groups); the bootstrap then runs without a refusal. Neither
point has been fixed. Both are recorded in the check-log note cited at the top of this page.

## Summary of what is supported at `:phy`

| Fit | Result |
|---|---|
| `PrecisionMultivariateFit` | tier `:phy` only: `Sigma_phy`, `R_phy`, `communality_phy` |
| `JointPhyloGroupedGaussianFit` | tier `:phy` only, same entries, plus `diagnostics` |
| either class with `y`, `n_sites`, `X`, `individual` or `Σ_phy` | refused (the fit stores what it needs) |
| `n_cores > 1`, `:B` or `:W` tiers, `ICC`, `cross_corr` | refused or not produced |

## Where next

[Worked example: bootstrap_Sigma_r](bootstrap-sigma-r-example.md) shows the same
function on an ordinary `GllvmFit`. [SE and profile machinery](se-profile-machinery.md)
states what `bootstrap_Sigma_r` is and is not, and
[Confidence intervals](confidence-intervals.md) covers the other interval routes.
