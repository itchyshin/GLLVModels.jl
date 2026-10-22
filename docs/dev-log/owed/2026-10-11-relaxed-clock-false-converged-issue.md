# GitHub issue (not filed — API 403)

`issue_write` against `itchyshin/GLLVModels.jl` returned
`403 Resource not accessible by personal access token`.
Paste the block below as a new issue. Do not use closing keywords.

**Title:** `fit_relaxed_clock` reports `converged = true` at the τ² floor [sibling of #753]

**Labels:** `code-review`

---

## Where

- `src/relaxed_clock.jl:499` (`fit_relaxed_clock` M-step): `τ² = max(var(ρ_mle) - samp_var, τ²_floor)` with default `τ²_floor = 1e-4`
- `src/relaxed_clock.jl:501`: `σ²_eps = max(recon / (n_rep * p), 1e-8)` — absolute `1e-8`, in y² units
- `src/relaxed_clock.jl:510-512`: `converged = true` when the penalised-EB objective increment falls below `tol * (|obj| + tol)`

`RelaxedClockFit` (`src/relaxed_clock.jl:375-388`) has `converged` and `n_iter` but no boundary field.

## What the code does and why it is a defect

The free-τ² update clamps to `τ²_floor` when the log-rate MLE dispersion is within sampling noise of a single rate (the honest weak-data / single-rate case). The residual update clamps to an absolute `1e-8` that does not scale with `y`. The loop then sets `converged = true` from the objective increment alone. A clamp is not an interior optimum; the same class of defect as `em_fit_phylo`'s silent `σ_eps²` floor (refs #753). Draft PR #868 covers `em_fit_phylo` / SQUAREM / `fit_branch_re` / ZINB `iterations=0` only — not this fitter.

## Minimal repro

Constant-rate BM on a 4-leaf balanced tree (no rate heterogeneity, so τ² hits the floor). Julia 1.10.12, this tree at `4fa4848c9`:

```julia
using GLLVModels, Random
phy = edge_phy("((A:0.3,B:0.3):0.3,(C:0.3,D:0.3):0.3);")
rng = MersenneTwister(1)
y, _, _ = simulate_relaxed_bm(phy, fill(1.0, phy.n_edges), 0.05, 80; rng = rng)
fit = fit_relaxed_clock(phy, y; max_iter = 2000, tol = 1e-4)
println(fit.converged, " ", fit.n_iter, " ", fit.τ², " ", fit.σ²_eps)
```

Output:

```
true 115 0.0001 0.08035768474171161
```

`τ²` equals the default `1e-4` floor exactly; `converged = true` after 115 iterations.

## What a user observes

A clean "converged" fit whose rate-heterogeneity variance is the numerical floor, not an estimated interior `τ²`. Single-rate or weak-replicate data look like a successful relaxed clock. The absolute `1e-8` `σ²_eps` clamp is also unit-dependent: rescaling `y` can flip whether the residual sits on that floor (the same review note that blocked #868's first floors).

## What the code would need to guarantee

Record the clamp (a field such as `tau2_boundary` / `sigma_eps_boundary`) and leave `converged = false` when `τ²` is at `τ²_floor` or `σ²_eps` is at its floor. Make the `σ²_eps` floor a fraction of the data scale (`mean(y.^2)` or `var(y)`), as #868 now does for branch-RE and phylo EM. Do not add an absolute gradient/objective tolerance; do not hard-code `converged = true`.

## Cause tag

silent-clamp / flag-ignores-fit

## Severity

Medium. Reachable with constant-rate or weak-`n_rep` data; result looks valid.

refs #753
