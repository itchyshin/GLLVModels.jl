# Boundary inference for variance components at the boundary of the parameter space
# (a variance → 0, on the edge of [0, ∞)). Adapted from the DRM-team toolkit
# (cross-pollination) — the honest answer to "is this RE / latent-factor variance 0?",
# where a naive χ²_q reference LRT is conservative because the parameter sits on the
# boundary. Self & Liang (1987) / Stram & Lee (1994): the null LRT follows a χ̄²
# MIXTURE of χ² distributions, not a single χ²_q.
#
# Use cases in GLLVModels.jl: the RE variances (σ_row, the grouped/structured σ_u, the
# random-slope Σ_b diagonals, the Poisson OLRE ψ_t) — all are variances tested against 0.
# Methods-sharing with DRM (MIT/GPL clean).
#
# NOT a valid reference for choosing the number of latent dimensions (K vs K+1). There the
# extra loading column is zero under the null and its direction is unidentified, so the
# problem is not one variance on a boundary: Wilks' theorem fails and the null LRT is not a
# χ̄² mixture (Hayashi, Bentler & Yuan 2007, Struct. Equ. Modeling 14:505–526; Drton 2009,
# Ann. Statist. 37:979–1012 — the limit involves Wishart eigenvalues). Use a parametric
# bootstrap for that test, or an information criterion via `select_lv`. (Corrected
# 2026-09-27, lane auto-d; the line previously listed K-selection as a use case.)

"""
    chibar2_pvalue(LRT, q) -> Float64

p-value of a likelihood-ratio statistic `LRT = 2(ℓ_full − ℓ_reduced)` for the boundary
null that `q` INDEPENDENT variance components are simultaneously 0. The reference is the
χ̄² mixture `Σ_{j=0}^{q} C(q,j) 2^{-q} χ²_j` (Self–Liang case for q independent variances
on the boundary), so

    p = Σ_{j=1}^{q} C(q,j) 2^{-q} · P(χ²_j ≥ LRT)          (the j=0 atom contributes 0 for LRT>0).

For `q = 1` this is the familiar `½ P(χ²_1 ≥ LRT)` (half the naive χ²_1 p-value). Returns
1.0 for `LRT ≤ 0` (no evidence). NOTE: the independent-variances mixture; a variance that
is also correlated with others (a ρ → ±1 boundary) needs the cone-geometry weights — a
documented follow-on.
"""
function chibar2_pvalue(LRT::Real, q::Integer)
    q ≥ 1 || throw(ArgumentError("q (number of boundary variances) must be ≥ 1; got $q"))
    LRT > 0 || return 1.0
    p = 0.0
    @inbounds for j in 1:q
        p += binomial(q, j) * 2.0^(-q) * ccdf(Chisq(j), float(LRT))
    end
    return p
end

"""
    variance_lrt(ℓ_full, ℓ_reduced; n_boundary=1) -> NamedTuple

Boundary likelihood-ratio test that `n_boundary` variance components are 0, from the two
maximised log-likelihoods (full model vs the reduced model with those variances fixed at
0). Returns `(LRT, pvalue, n_boundary)` with the χ̄² mixture p-value
([`chibar2_pvalue`](@ref)). The naive `2·ΔlogLik ~ χ²` p-value would OVER-state the
evidence's conservatism (be too large) — the boundary correction halves (q=1) it.
"""
function variance_lrt(ℓ_full::Real, ℓ_reduced::Real; n_boundary::Integer = 1)
    LRT = 2 * (float(ℓ_full) - float(ℓ_reduced))
    return (LRT = LRT, pvalue = chibar2_pvalue(LRT, n_boundary), n_boundary = n_boundary)
end

# Bisection for a root of g on [a,b] assuming g(a), g(b) bracket it (opposite signs).
function _bisect_root(g, a::Real, b::Real, tol::Real, maxit::Integer)
    fa = g(a)
    for _ in 1:maxit
        m = (a + b) / 2
        fm = g(m)
        (abs(fm) < tol || (b - a) < tol) && return m
        if sign(fm) == sign(fa)
            a = m; fa = fm
        else
            b = m
        end
    end
    return (a + b) / 2
end

"""
    profile_ci_variance(refit_at, v̂, ℓ_max; level=0.95, lower=0.0, …) -> NamedTuple

Profile-likelihood CI for a variance component at the boundary of `[0, ∞)`. `refit_at(v)`
returns the maximised profile log-likelihood with the variance FIXED at `v` (the other
parameters re-optimised); `v̂` is the MLE, `ℓ_max` the full maximised log-likelihood. Inverts
`2(ℓ_max − ℓ_profile(v)) = χ²₁(level)` by bracket-then-bisect, **clamping the lower edge at
`lower` (0)** — when the profile at the boundary is still within the threshold (a poorly-
identified / near-0 variance), the CI lower bound IS the boundary and `at_boundary=true` (the
honest "this variance is consistent with 0" result, paired with [`chibar2_pvalue`](@ref)).
Returns `(lower, upper, level, at_boundary)`.
"""
function profile_ci_variance(refit_at::Function, v̂::Real, ℓ_max::Real;
        level::Real = 0.95, lower::Real = 0.0, upper_factor::Real = 50.0,
        tol::Real = 1e-4, maxit::Integer = 80)
    thr = quantile(Chisq(1), level) / 2
    target = ℓ_max - thr
    g = v -> refit_at(v) - target                 # > 0 inside the CI, < 0 outside
    lo = g(lower) ≥ 0 ? float(lower) : _bisect_root(g, lower, v̂, tol, maxit)
    hi_hi = v̂ > 0 ? v̂ * upper_factor : float(upper_factor)
    while g(hi_hi) > 0 && hi_hi < 1e8
        hi_hi *= 2
    end
    hi = _bisect_root(g, v̂, hi_hi, tol, maxit)
    return (lower = lo, upper = hi, level = level, at_boundary = (lo == float(lower)))
end
