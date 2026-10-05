# flag_unreliable_loadings: the Julia twin of gllvmTMB's flag_unreliable_loadings()
# (R/loading-ci.R at P1). Given per-entry raw Wald intervals on a confirmatory Λ,
# flag the entries whose interval does not exclude a "negligible" band around zero.
#
# The pinned rows and columns are dropped from the observed information before it is
# inverted, which is R's `sd_report$cov.fixed` (the pinned entries are mapped off in R,
# not parameters). `loading_ci` and `confint` use the same free-parameter covariance
# (#794), so their raw Wald SEs agree with the ones computed here.

"""
    flag_unreliable_loadings(fit::GllvmFit, y; null_region = (-0.1, 0.1),
                             level = :unit, conf_level = 0.95) -> Vector{NamedTuple}
    flag_unreliable_loadings(rows::AbstractVector; null_region = (-0.1, 0.1))
        -> Vector{NamedTuple}

Flag loadings whose confidence interval overlaps a band of negligible values, the
Julia counterpart of gllvmTMB's `flag_unreliable_loadings()`. An interval
`[lower, upper]` overlaps `null_region = (a, b)` when `upper >= a` and `lower <= b`;
such an entry is flagged `unreliable = true`, meaning the data do not show that the
trait responds non-trivially to that axis. Pinned entries get `unreliable = missing`,
because no inference is made about them.

The first method takes a confirmatory Gaussian fit, made with
`fit_gaussian_gllvm(y; K, lambda_constraint = M)`, and the response matrix `y` that
was passed to that call. Like R, it refuses a fit without pins, since a single entry
of an unpinned `Λ` is identified only up to rotation. The intervals are symmetric
Wald intervals on the raw loading scale (R's default `method = "wald"`,
`loading_scale = "raw"`) at `conf_level`, from the observed information of the free
parameters: pinned loadings are not parameters, so their rows and columns are
removed before the information is inverted, as in R's `sd_report$cov.fixed`. A
pinned entry has `se = 0` and `lower = upper = estimate`. If the reduced
information is not positive definite, `se`, `lower` and `upper` are `NaN`,
`pd_hessian = false`, and every `unreliable` is `missing` (R returns `NA`). Only
`level = :unit`, the shared tier, is available, because `lambda_constraint` fits
have no other loading tier.

`pinned` marks the user's numeric entries of `M` and the structural zeros above the
diagonal of the first `K` rows (`k > i`). R marks only the user's entries, so a
structural zero the user left as `NA` is pinned here but reported by R as an unpinned
zero-width interval; stating that zero in `M`, as R's `confirmatory_lambda()` does,
gives the same rows on both sides.

The second method takes rows that already carry `estimate`, `lower`, `upper` and
`pinned` fields (for example the rows of [`loading_ci`](@ref)) and adds the flag
columns to each, which is R's data-frame input. A `missing` or `NaN` bound gives `unreliable = missing`.

Each returned row has `trait`, `axis`, `estimate`, `se`, `lower`, `upper`,
`conf_level`, `pinned`, `pd_hessian`, `unreliable`, `null_region_lo` and
`null_region_hi` (the second method keeps the input fields and adds the last three).
Rows run over traits first, then axes, as in R.

```julia
M = fill(NaN, 6, 2); M[1, 2] = 0.0; M[2, 1] = 0.0
fit = fit_gaussian_gllvm(y; K = 2, lambda_constraint = M)   # y centred by trait
flag_unreliable_loadings(fit, y)
flag_unreliable_loadings(fit, y; null_region = (-0.5, 0.5))
```
"""
function flag_unreliable_loadings(fit::GllvmFit, y::AbstractMatrix;
                                  null_region = (-0.1, 0.1),
                                  level::Symbol = :unit,
                                  conf_level::Real = 0.95)
    a, b = _flag_null_region(null_region)
    level in (:unit, :B) || throw(ArgumentError(
        "flag_unreliable_loadings: level = :unit is the only tier of a lambda_constraint fit; got :$level"))
    0 < conf_level < 1 || throw(ArgumentError("conf_level must lie in (0, 1); got $conf_level"))
    M = hasproperty(fit.pars, :lambda_constraint) ? fit.pars.lambda_constraint : nothing
    (M === nothing || all(isnan, M)) && throw(ArgumentError(
        "flag_unreliable_loadings: per-entry Wald intervals on Λ are defined only for a " *
        "confirmatory fit; refit with fit_gaussian_gllvm(y; K, lambda_constraint = M). " *
        "Without pins Λ is identified only up to rotation."))
    p, K = fit.model.p, fit.model.K
    size(y, 1) == p || throw(DimensionMismatch("y must have p = $p rows; got $(size(y, 1))"))
    Λ = fit.pars.Λ
    pinned = _lambda_constraint_is_pinned(M, p, K)

    # Observed information of the free parameters: drop the pinned loadings.
    θ̂ = fit.pars.θ_packed
    pin_idx = _lambda_constraint_pinned_theta_indices(fit)
    free = setdiff(1:length(θ̂), pin_idx)
    nll = _confint_reconstruct_nll(fit, y, nothing, nothing)
    V = nothing
    try
        H = ForwardDiff.hessian(nll, θ̂)[free, free]
        if all(isfinite, H)
            C = cholesky(Symmetric((H .+ H') ./ 2); check = false)
            issuccess(C) && (V = inv(C))
        end
    catch
        V = nothing
    end
    pd = V !== nothing
    pos = Dict(j => r for (r, j) in enumerate(free))
    z = quantile(Normal(), 0.5 + conf_level / 2)

    rows = Vector{NamedTuple}(undef, p * K)
    r = 0
    for k in 1:K, i in 1:p
        r += 1
        est = Λ[i, k]
        se = if pinned[i, k]
            0.0
        elseif pd
            v = V[pos[_lambda_b_theta_index(fit, i, k)], pos[_lambda_b_theta_index(fit, i, k)]]
            v > 0 ? sqrt(v) : NaN
        else
            NaN
        end
        lo, hi = isfinite(se) ? (est - z * se, est + z * se) : (NaN, NaN)
        rows[r] = (; trait = i, axis = k, estimate = est, se = se, lower = lo, upper = hi,
                   conf_level = Float64(conf_level), pinned = pinned[i, k], pd_hessian = pd,
                   unreliable = _flag_overlap(lo, hi, pinned[i, k], a, b),
                   null_region_lo = a, null_region_hi = b)
    end
    return rows
end

function flag_unreliable_loadings(rows::AbstractVector; null_region = (-0.1, 0.1))
    a, b = _flag_null_region(null_region)
    out = Vector{NamedTuple}(undef, length(rows))
    for (j, row) in enumerate(rows)
        all(hasproperty(row, f) for f in (:estimate, :lower, :upper, :pinned)) ||
            throw(ArgumentError(
                "flag_unreliable_loadings: each row needs the fields estimate, lower, upper " *
                "and pinned (the rows of loading_ci)"))
        flag = _flag_overlap(row.lower, row.upper, row.pinned, a, b)
        out[j] = merge(NamedTuple(row), (; unreliable = flag, null_region_lo = a, null_region_hi = b))
    end
    return out
end

function _flag_null_region(null_region)
    length(null_region) == 2 || throw(ArgumentError(
        "null_region must have two numbers with null_region[1] < null_region[2]"))
    a, b = Float64(null_region[1]), Float64(null_region[2])
    isfinite(a) && isfinite(b) && a < b || throw(ArgumentError(
        "null_region must have two numbers with null_region[1] < null_region[2]; got $null_region"))
    return a, b
end

# R: overlaps <- upper >= a & lower <= b; NA for pinned entries and for NA bounds
# (a `missing` or `NaN` bound gives a `missing` flag).
_flag_bound_unknown(x) = x === missing || isnan(x)
_flag_overlap(lo, hi, pinned::Bool, a, b) =
    pinned || _flag_bound_unknown(lo) || _flag_bound_unknown(hi) ? missing : (hi >= a && lo <= b)
