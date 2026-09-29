# Exact Gaussian marginal likelihood of the temporal source (slices 1 and 2).
#
#   y ~ N(X beta, Z (K_blockdiag ⊗ Sigma_T) Z' + J_unit ∘ Sigma_B
#                 + J_unit_obs ∘ Sigma_W + sigma_re^2 J_group + sigma_eps^2 I)
#
# K is block diagonal over series: phi^|t - u| (AR1, integer power) or
# exp(-kappa |t - u|) (OU); Sigma_T is the trait block of the mode. The
# ordinary unit (B), unit_obs (W) and `(1 | g)` blocks are present only when
# composed (slice 2). Every ingredient is linear Gaussian, so this equals
# gllvmTMB's Laplace objective (spec section 1.5). Evaluated as a sum of dense
# Gaussian terms over independent row blocks.

"""
    TemporalLayout

Coordinate layout of the temporal parameter vector, in the order of
gllvmTMB's `opt\$par` as measured on P1 fits: `b_fix`, `log_sigma_eps`,
`theta_rr_B`, `theta_temporal_time`, `theta_temporal_rr`,
`theta_temporal_diag`, `theta_diag_B`, `theta_rr_W`, `theta_diag_W`,
`log_sigma_re_int`. Unused blocks are absent, as R maps them off; the
`_B` / `_W` / `re_int` blocks exist only when ordinary unit / unit_obs terms
are composed with the temporal source (slice 2), and `log_sigma` is `0` when
gllvmTMB's per-row suppression rule fixes `sigma_eps`.
"""
struct TemporalLayout
    q::Int
    p::Int
    rank::Int
    unique::Bool
    beta::UnitRange{Int}
    log_sigma::Int
    time::Int
    rr::UnitRange{Int}
    diag::UnitRange{Int}
    total::Int
    rank_B::Int
    rank_W::Int
    rr_B::UnitRange{Int}
    diag_B::UnitRange{Int}
    rr_W::UnitRange{Int}
    diag_W::UnitRange{Int}
    re_int::UnitRange{Int}
end

function TemporalLayout(q::Integer, p::Integer, rank::Integer, unique::Bool,
        comp::TemporalComposition=_TEMPORAL_NO_COMPOSITION)
    cursor = 0
    take(n) = (r = (cursor + 1):(cursor + n); cursor += n; r)
    tier_rank(t) = t === nothing ? 0 : t.rank
    tier_rr(t) = tier_rank(t) > 0 ? rr_theta_len(p, t.rank) : 0
    tier_diag(t) = t !== nothing && t.diag ? p : 0
    beta = take(q)
    log_sigma = comp.sigma_fixed === nothing ? first(take(1)) : 0
    rr_B = take(tier_rr(comp.B))
    time = first(take(1))
    rr = take(rank > 0 ? rr_theta_len(p, rank) : 0)
    diag = take(unique ? p : 0)
    diag_B = take(tier_diag(comp.B))
    rr_W = take(tier_rr(comp.W))
    diag_W = take(tier_diag(comp.W))
    re_int = take(comp.re_int === nothing ? 0 : 1)
    return TemporalLayout(q, p, rank, unique, beta, log_sigma, time, rr, diag, cursor,
        tier_rank(comp.B), tier_rank(comp.W), rr_B, diag_B, rr_W, diag_W, re_int)
end

TemporalLayout(q::Integer, spec::TemporalSpec) =
    TemporalLayout(q, length(spec.traits), spec.rank, spec.unique, spec.composition)

"""R's `names(opt\$par)` for a layout: block names repeated per coordinate."""
function _temporal_parameter_names(L::TemporalLayout)
    names = String[]
    append!(names, fill("b_fix", L.q))
    L.log_sigma > 0 && push!(names, "log_sigma_eps")
    append!(names, fill("theta_rr_B", length(L.rr_B)))
    push!(names, "theta_temporal_time")
    append!(names, fill("theta_temporal_rr", length(L.rr)))
    append!(names, fill("theta_temporal_diag", length(L.diag)))
    append!(names, fill("theta_diag_B", length(L.diag_B)))
    append!(names, fill("theta_rr_W", length(L.rr_W)))
    append!(names, fill("theta_diag_W", length(L.diag_W)))
    append!(names, fill("log_sigma_re_int", length(L.re_int)))
    return names
end

# Residual variance: free (exp(2 log_sigma_eps)) or fixed by the suppression rule.
_temporal_sigma2(L::TemporalLayout, spec::TemporalSpec, theta) = L.log_sigma > 0 ?
    exp(2 * theta[L.log_sigma]) : oftype(one(eltype(theta)), spec.composition.sigma_fixed^2)

# Trait covariance of an ordinary tier: Lambda Lambda' (rank > 0) plus
# diag(exp(2 theta_diag)), gllvmTMB's theta_rr_* / theta_diag_* blocks.
function _temporal_tier_block(theta::AbstractVector, p::Integer, rank::Integer,
        rr::UnitRange{Int}, dg::UnitRange{Int})
    S = zeros(eltype(theta), p, p)
    rank > 0 && (L = unpack_lambda(view(theta, rr), p, rank); S .+= L * L')
    for (k, j) in enumerate(dg)
        S[k, k] += exp(2 * theta[j])
    end
    return S
end

_temporal_phi(theta) = (1 - 1e-6) * tanh(theta)
_temporal_kappa(theta) = exp(theta)
_temporal_time_value(structure::Symbol, theta) =
    structure === :ar1 ? _temporal_phi(theta) : _temporal_kappa(theta)

"""
    _temporal_trait_block(L, theta) -> (Sigma_T, Lambda_or_nothing, psi_or_nothing)

Trait covariance at one state: `diag(psi)` (indep), `L L'` (dep),
`lambda lambda'` (+ `diag(psi)` when unique) for latent. `psi_j =
exp(2 theta_diag_j)`; loadings unpack with gllvmTMB's convention
([`unpack_lambda`](@ref)).
"""
function _temporal_trait_block(L::TemporalLayout, theta::AbstractVector)
    T = eltype(theta)
    p = L.p
    Sigma = zeros(T, p, p)
    Lambda = nothing
    psi = nothing
    if L.rank > 0
        Lambda = unpack_lambda(view(theta, L.rr), p, L.rank)
        Sigma .+= Lambda * Lambda'
    end
    if L.unique
        psi = exp.(2 .* theta[L.diag])
        for j in 1:p
            Sigma[j, j] += psi[j]
        end
    end
    return Sigma, Lambda, psi
end

# Temporal correlation between two occasions of the same series.
@inline function _temporal_corr(structure::Symbol, a, t1::Float64, t2::Float64)
    if structure === :ar1
        # Integer exponent: stays on the real branch for negative phi and for
        # every AD number type (spec section 3.4; gllvmTMB.cpp:62-75).
        k = Int(abs(round(t1 - t2)))
        return k == 0 ? one(a) : a^k
    else
        return exp(-a * abs(t1 - t2))
    end
end

# Independent row blocks: rows linked by a shared series, and (when composed)
# a shared unit, unit_obs or random-intercept level. Without ordinary terms
# these are the series, in pair-table series order.
function _temporal_series_rows(spec::TemporalSpec)
    c = spec.composition
    if !_temporal_composed(c)
        series = unique(spec.pair_table.series)
        return [findall(==(s), spec.row_series) for s in series]
    end
    n = length(spec.row_series)
    parent = collect(1:n)
    root(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    function link(keys)
        first_row = Dict{Any,Int}()
        for (o, k) in enumerate(keys)
            r = get!(first_row, k, o)
            a, b = root(o), root(r)
            a == b || (parent[max(a, b)] = min(a, b))
        end
    end
    link(spec.row_series)
    c.B === nothing || link(c.unit_id)
    c.W === nothing || link(c.unit_obs_id)
    c.re_int === nothing || link(c.re_int_id)
    roots = [root(o) for o in 1:n]
    return [findall(==(r), roots) for r in unique(roots)]
end

# The pieces of V at theta: temporal trait block and time value, ordinary tier
# blocks, random-intercept variance, residual variance.
function _temporal_cov_pieces(theta::AbstractVector, spec::TemporalSpec, L::TemporalLayout)
    c = spec.composition
    Sigma, _, _ = _temporal_trait_block(L, theta)
    a = _temporal_time_value(spec.structure, theta[L.time])
    SB = c.B === nothing ? nothing : _temporal_tier_block(theta, L.p, L.rank_B, L.rr_B, L.diag_B)
    SW = c.W === nothing ? nothing : _temporal_tier_block(theta, L.p, L.rank_W, L.rr_W, L.diag_W)
    re2 = c.re_int === nothing ? nothing : exp(2 * theta[first(L.re_int)])
    return (Sigma=Sigma, a=a, SB=SB, SW=SW, re2=re2, sigma2=_temporal_sigma2(L, spec, theta))
end

# Marginal covariance between rows oi and oj (without the residual).
@inline function _temporal_cov_entry(spec::TemporalSpec, P, oi::Int, oj::Int)
    c = spec.composition
    ti, tj = spec.trait_id[oi], spec.trait_id[oj]
    v = zero(P.a)
    if spec.row_series[oi] == spec.row_series[oj]
        v += _temporal_corr(spec.structure, P.a, spec.row_time[oi], spec.row_time[oj]) * P.Sigma[ti, tj]
    end
    P.SB === nothing || c.unit_id[oi] != c.unit_id[oj] || (v += P.SB[ti, tj])
    P.SW === nothing || c.unit_obs_id[oi] != c.unit_obs_id[oj] || (v += P.SW[ti, tj])
    P.re2 === nothing || c.re_int_id[oi] != c.re_int_id[oj] || (v += P.re2)
    return v
end

"""
    temporal_marginal_nll(theta, y, X, spec; series_rows = nothing)

Exact negative marginal log-likelihood of the temporal Gaussian model at the
coordinate vector `theta` (layout [`TemporalLayout`](@ref)), including any
ordinary unit / unit_obs / random-intercept blocks composed with the temporal
source. Evaluated as a sum over independent row blocks. Returns `Inf` when a
block covariance is not positive definite or a value is not finite.
"""
function temporal_marginal_nll(theta::AbstractVector, y::AbstractVector,
        X::AbstractMatrix, spec::TemporalSpec; series_rows=nothing)
    L = TemporalLayout(size(X, 2), spec)
    length(theta) == L.total ||
        throw(DimensionMismatch("temporal parameter vector has $(length(theta)) coordinates; expected $(L.total)"))
    T = eltype(theta)
    all(isfinite, theta) || return T(Inf)
    blocks = series_rows === nothing ? _temporal_series_rows(spec) : series_rows
    P = _temporal_cov_pieces(theta, spec, L)
    residual = y .- X * theta[L.beta]
    nll = zero(T)
    for rows in blocks
        m = length(rows)
        V = Matrix{T}(undef, m, m)
        for jb in 1:m, ib in jb:m
            v = _temporal_cov_entry(spec, P, rows[ib], rows[jb])
            ib == jb && (v += P.sigma2)
            V[ib, jb] = v
            V[jb, ib] = v
        end
        all(isfinite, V) || return T(Inf)
        F = cholesky(Symmetric(V, :L); check=false)
        issuccess(F) || return T(Inf)
        z = F.L \ residual[rows]
        nll += (m * log(2pi) + logdet(F) + dot(z, z)) / 2
    end
    return nll
end

"""Positive twin of [`temporal_marginal_nll`](@ref)."""
temporal_marginal_loglik(theta, y, X, spec; kwargs...) =
    -temporal_marginal_nll(theta, y, X, spec; kwargs...)

"""
    _temporal_covariance(theta, spec, q) -> Matrix

Dense marginal covariance of all response rows (row order of the data) at
`theta`: the temporal block `same_series * K(t, u) * Sigma_T[trait, trait']`,
plus the ordinary unit, unit_obs and random-intercept blocks when composed,
plus `sigma_eps^2 I`. Used by the helpers and by tests; the likelihood itself
factors over independent row blocks.
"""
function _temporal_covariance(theta::AbstractVector, spec::TemporalSpec, q::Integer)
    L = TemporalLayout(q, spec)
    P = _temporal_cov_pieces(theta, spec, L)
    n = length(spec.row_time)
    V = zeros(eltype(theta), n, n)
    for rows in _temporal_series_rows(spec), j in rows, i in rows
        V[i, j] = _temporal_cov_entry(spec, P, i, j)
    end
    for i in 1:n
        V[i, i] += P.sigma2
    end
    return V
end
