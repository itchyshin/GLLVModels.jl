# D3 Stage 1 confirmatory `loading_profile` export.
#
# Kept in its own file (not `src/confint_derived.jl`, which defines the
# generic `loading_profile(args...; kwargs...)` deprecation shim this method
# coexists with via multiple dispatch) to stay clear of the concurrent
# reader-facing cleanup on `src/confint_derived.jl` /
# `docs/src/derived-confidence-intervals.md` (PR #437). Included from
# `src/GLLVModels.jl` right after `confint_derived.jl`, so the shim is defined
# first and this adds a second, more specific method to the same generic
# function — dispatch is unaffected by file/include order either way (Julia
# resolves multiple dispatch by method specificity, not definition order),
# but this keeps the narrative order (shim, then Stage 1 mirror) legible.

"""
    loading_profile(fit::GllvmFit; level::Symbol = :unit,
                     entries::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                     n_grid::Integer = 11, grid_extent::Real = 6.0,
                     conf_level::Real = 0.95, y::AbstractMatrix)
        -> NamedTuple

**Confirmatory profile-likelihood grid** for the free entries of a **pinned**
`Λ` (`level = :unit`, the shared/between tier), mirroring R gllvmTMB's
`loading_profile()`. Requires `fit` to be a **confirmatory** fit — built via
`fit_gaussian_gllvm(y; K, lambda_constraint = M)` — so this refuses plain
exploratory fits (opposite of R's `loading_ci()`, which refuses unpinned fits
for the same reason: distinct estimands need distinct fit metadata). Use
[`loading_profile_exploratory`](@ref) for a penalty-based profile CI on a
single raw `Λ` entry of an ordinary (unpinned) fit.

# Keywords
- `level`: `:unit` (equivalently `:B`) is the only tier currently supported —
  the shared/between-tier `Λ`. `:unit_obs`/`:W` are not yet supported (no
  within-tier block exists on the ordinary fits this function admits).
- `entries`: an `n × 2` integer matrix of `(trait, axis)` pairs to profile: one
  row per requested entry. `nothing` (default) profiles every free entry.
- `n_grid`: number of grid points per profiled entry (must be `≥ 3`).
- `grid_extent`: profiling half-width in Wald-SE units around the confirmatory
  MLE for each entry (falls back to `|Λ̂|/2 + 0.5` when the Hessian-based SE is
  non-finite, matching the fallback in `loading_profile_exploratory`).
- `conf_level`: confidence level recorded alongside each row (not itself used
  to trim the grid — grid width is controlled by `n_grid` / `grid_extent`).
- `y`: the response matrix used to fit `fit` (required — same convention as
  `loading_profile_exploratory`).

# Return
A `NamedTuple` `(table, level, n_grid, grid_extent, conf_level, entries)`.
`table` is a `Vector` of row `NamedTuple`s (`Tables.jl`-compatible row table)
with fields `trait`, `axis`, `i`, `k`, `profile_value`, `objective`
(`-logLik` at that grid point), `delta_deviance` (`2 * (fit.logLik - ll)`,
`NaN` when the refit did not converge), `estimate` (the confirmatory MLE for
that entry), `conf_level`, and `converged`.

# Current limits
Available for the ordinary Gaussian latent-variable model only: no
phylogenetic or diagonal random-effect terms, and no fixed-effect
covariates. `fit` must come from `fit_gaussian_gllvm` with `lambda_constraint`
set and without `aghq`, `mask`, `offset`, or predictor-informed latent scores
(`X_lv`) — those combinations are refused with a clear error rather than
silently fitting the wrong model. Each free entry is refit one at a time,
holding it at every grid value while re-optimising everything else. The grid
itself follows a Wald-standard-error heuristic rather than R's own
grid-spacing rule, and no cross-package numeric comparison against R's
`loading_profile()` output has been published yet.
"""
function loading_profile(fit::GllvmFit;
                          level::Symbol = :unit,
                          entries::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                          n_grid::Integer = 11,
                          grid_extent::Real = 6.0,
                          conf_level::Real = 0.95,
                          y::AbstractMatrix)
    level === :unit || level === :B ||
        throw(ArgumentError(
            "loading_profile Stage 1 supports level = :unit (component = :B) only; got $(level)"))
    haskey(fit.pars, :lambda_constraint) ||
        throw(ArgumentError(
            "loading_profile requires a confirmatory fit: refit with " *
            "fit_gaussian_gllvm(y; K, lambda_constraint = M) before profiling"))
    n_grid isa Integer && n_grid >= 3 ||
        throw(ArgumentError("n_grid must be an integer ≥ 3"))
    0.0 < conf_level < 1.0 || throw(ArgumentError("conf_level must lie in (0, 1)"))
    grid_extent > 0 || throw(ArgumentError("grid_extent must be positive"))

    p, K = fit.model.p, fit.model.K
    M_user = fit.pars.lambda_constraint
    free = _enumerate_free_lambda_entries(M_user, p, K; entries = entries)
    isempty(free) &&
        throw(ArgumentError("no free Λ entries to profile (all pinned/structural)"))

    Λ̂ = fit.pars.Λ
    rows = NamedTuple[]
    for (i, k) in free
        θ_idx = _lambda_b_theta_index(fit, i, k)
        se = _profile_wald_se(fit, θ_idx, y, nothing, nothing)
        c_hat = Λ̂[i, k]
        half_width = isfinite(se) && se > 0 ? grid_extent * se :
            grid_extent * (abs(c_hat) / 2 + 0.5)
        grid = range(c_hat - half_width, c_hat + half_width; length = n_grid)
        for c in grid
            ll, ok = _confirmatory_profile_refit_lambda_pin(
                fit, y, M_user, i, k, c; require_paste = false)
            obj = ok ? -ll : NaN
            delta_dev = ok ? 2 * (fit.logLik - ll) : NaN
            push!(rows, (trait = i, axis = k, i = i, k = k,
                          profile_value = c, objective = obj,
                          delta_deviance = delta_dev, estimate = c_hat,
                          conf_level = conf_level, converged = ok))
        end
    end
    return (table = rows, level = :unit, n_grid = n_grid, grid_extent = grid_extent,
            conf_level = conf_level, entries = free)
end
