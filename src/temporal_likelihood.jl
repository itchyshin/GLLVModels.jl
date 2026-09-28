# Exact Gaussian marginal likelihood of the temporal source (slice 1).
#
#   y ~ N(X beta, Z (K_blockdiag ⊗ Sigma_T) Z' + sigma_eps^2 I)
#
# K is block diagonal over series: phi^|t - u| (AR1, integer power) or
# exp(-kappa |t - u|) (OU); Sigma_T is the trait block of the mode. Every
# ingredient is linear Gaussian, so this equals gllvmTMB's Laplace objective
# (spec section 1.5). Evaluated as a sum of dense per-series Gaussian terms.

"""
    TemporalLayout

Coordinate layout of the temporal parameter vector, in the order of
gllvmTMB's `opt\$par` (TMB orders fixed parameters by their template
declaration: `b_fix`, `log_sigma_eps`, `theta_temporal_time`,
`theta_temporal_rr`, `theta_temporal_diag`). Unused blocks are absent, as
R maps them off.
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
end

function TemporalLayout(q::Integer, p::Integer, rank::Integer, unique::Bool)
    nrr = rank > 0 ? rr_theta_len(p, rank) : 0
    ndiag = unique ? p : 0
    beta = 1:q
    log_sigma = q + 1
    time = q + 2
    rr = (q + 3):(q + 2 + nrr)
    diag = (q + 3 + nrr):(q + 2 + nrr + ndiag)
    return TemporalLayout(q, p, rank, unique, beta, log_sigma, time, rr, diag,
        q + 2 + nrr + ndiag)
end

TemporalLayout(q::Integer, spec::TemporalSpec) =
    TemporalLayout(q, length(spec.traits), spec.rank, spec.unique)

"""R's `names(opt\$par)` for a layout: block names repeated per coordinate."""
function _temporal_parameter_names(L::TemporalLayout)
    names = String[]
    append!(names, fill("b_fix", L.q))
    push!(names, "log_sigma_eps", "theta_temporal_time")
    append!(names, fill("theta_temporal_rr", length(L.rr)))
    append!(names, fill("theta_temporal_diag", length(L.diag)))
    return names
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

# Row indices grouped by series, in pair-table series order.
function _temporal_series_rows(spec::TemporalSpec)
    series = unique(spec.pair_table.series)
    return [findall(==(s), spec.row_series) for s in series]
end

"""
    temporal_marginal_nll(theta, y, X, spec; series_rows = nothing)

Exact negative marginal log-likelihood of the temporal Gaussian model at the
coordinate vector `theta` (layout [`TemporalLayout`](@ref)). Returns `Inf`
when a per-series covariance is not positive definite or a value is not finite.
"""
function temporal_marginal_nll(theta::AbstractVector, y::AbstractVector,
        X::AbstractMatrix, spec::TemporalSpec; series_rows=nothing)
    L = TemporalLayout(size(X, 2), spec)
    length(theta) == L.total ||
        throw(DimensionMismatch("temporal parameter vector has $(length(theta)) coordinates; expected $(L.total)"))
    T = eltype(theta)
    all(isfinite, theta) || return T(Inf)
    rows_by_series = series_rows === nothing ? _temporal_series_rows(spec) : series_rows
    Sigma, _, _ = _temporal_trait_block(L, theta)
    a = _temporal_time_value(spec.structure, theta[L.time])
    sigma2 = exp(2 * theta[L.log_sigma])
    residual = y .- X * theta[L.beta]
    nll = zero(T)
    for rows in rows_by_series
        m = length(rows)
        V = Matrix{T}(undef, m, m)
        for jb in 1:m, ib in jb:m
            oi, oj = rows[ib], rows[jb]
            v = _temporal_corr(spec.structure, a, spec.row_time[oi], spec.row_time[oj]) *
                Sigma[spec.trait_id[oi], spec.trait_id[oj]]
            ib == jb && (v += sigma2)
            V[ib, jb] = v
            V[jb, ib] = v
        end
        all(isfinite, V) || return T(Inf)
        F = cholesky(Symmetric(V, :L); check=false)
        issuccess(F) || return T(Inf)
        r = residual[rows]
        z = F.L \ r
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
`theta`: `same_series * K(t, u) * Sigma_T[trait, trait'] + sigma_eps^2 I`.
Used by the helpers and by tests; the likelihood itself factors per series.
"""
function _temporal_covariance(theta::AbstractVector, spec::TemporalSpec, q::Integer)
    L = TemporalLayout(q, spec)
    Sigma, _, _ = _temporal_trait_block(L, theta)
    a = _temporal_time_value(spec.structure, theta[L.time])
    sigma2 = exp(2 * theta[L.log_sigma])
    n = length(spec.row_time)
    V = zeros(eltype(theta), n, n)
    for j in 1:n, i in 1:n
        spec.row_series[i] == spec.row_series[j] || continue
        V[i, j] = _temporal_corr(spec.structure, a, spec.row_time[i], spec.row_time[j]) *
            Sigma[spec.trait_id[i], spec.trait_id[j]]
    end
    for i in 1:n
        V[i, i] += sigma2
    end
    return V
end
