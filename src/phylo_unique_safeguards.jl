# Collapsed-at-origin guard for the phylo_unique block (per-trait σ_phy).

const _SIGMA_PHY_COLLAPSE_ABS = 1e-6
const _PHY_UNIQUE_NOPHYLO_LLTOL = 1e-6

"""
    _scaled_sigma_phy_init(Σ_phy) -> Float64

Σ_phy-scaled default start for signed σ_phy (issue #1179): `0.5 / sqrt(mean(diag(Σ_phy)))`.
Falls back to `0.1` when the tree covariance has nonpositive mean diagonal.
"""
function _scaled_sigma_phy_init(Σ_phy::AbstractMatrix)
    d = mean(diag(Symmetric(Σ_phy)))
    (isfinite(d) && d > 0) || return 0.1
    return 0.5 / sqrt(d)
end

_sigma_phy_collapsed(σ_phy::AbstractVector) =
    isempty(σ_phy) ? false : maximum(abs.(σ_phy)) < _SIGMA_PHY_COLLAPSE_ABS

function _phy_unique_loglik_without_block(y::AbstractMatrix, Λ::AbstractMatrix, σ_eps::Real)
    return gaussian_marginal_loglik(y, Λ, σ_eps)
end

"""
    _phy_unique_stalled_at_origin(σ_phy, logLik, y, Λ_B, σ_eps) -> Bool

`true` when every `|σ_phy[t]|` is below `_SIGMA_PHY_COLLAPSE_ABS` and the marginal
log-likelihood matches the model with no phylogenetic block (same Λ_B, σ_eps).
"""
function _phy_unique_stalled_at_origin(σ_phy::AbstractVector, logLik::Real,
                                       y::AbstractMatrix, Λ_B::AbstractMatrix, σ_eps::Real)
    _sigma_phy_collapsed(σ_phy) || return false
    ll0 = _phy_unique_loglik_without_block(y, Λ_B, σ_eps)
    return abs(logLik - ll0) ≤ max(_PHY_UNIQUE_NOPHYLO_LLTOL, 1e-8 * max(1.0, abs(ll0)))
end

function _resolve_sigma_phy_init(σ_phy_init, Σ_phy::AbstractMatrix, p::Integer)
    if σ_phy_init === nothing
        return fill(_scaled_sigma_phy_init(Σ_phy), p)
    elseif σ_phy_init isa Number
        return fill(Float64(σ_phy_init), p)
    else
        v = Vector{Float64}(σ_phy_init)
        length(v) == p ||
            throw(ArgumentError("σ_phy_init length must be p ($p); got $(length(v))"))
        return v
    end
end

function _resolve_sigma_phy_abs_init(σ_phy_init, Σ_phy::AbstractMatrix)
    σ_phy_init === nothing && return _scaled_sigma_phy_init(Σ_phy)
    return Float64(σ_phy_init)
end