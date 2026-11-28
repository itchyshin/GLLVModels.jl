# In-sample predict / fitted for grouped-dispersion covariate fits where getLV
# already exists: BetaGroupedCovFit, GammaGroupedCovFit, OrdinalPerTraitCovFit
# (issue #1529). Mirrors NBGroupedCovFit (nb_grouped_postfit.jl); new-site
# predict(fit, X) is tracked separately (#1192).

function _grouped_cov_check_predict_type(type::Symbol)
    type in (:response, :mean, :link) ||
        throw(ArgumentError("type must be :response, :mean, or :link; got :$type"))
    return nothing
end

function _grouped_cov_check_ordinal_predict_type(type::Symbol)
    type in (:link, :prob, :class, :response) ||
        throw(ArgumentError("type must be :link, :prob, :class, or :response; got :$type"))
    return nothing
end

# ---------------------------------------------------------------------------
# BetaGroupedCovFit
# ---------------------------------------------------------------------------

"""
    predict(fit::BetaGroupedCovFit, Y, X; type=:response, mask=nothing) -> p×n matrix

In-sample fitted values at the Laplace mode with covariate offset: `type=:link` returns
`η = β + Xγ + Λẑ`; `type=:response` (= `:mean`) the mean `μ = linkinv(link, η)`.
"""
function predict(fit::BetaGroupedCovFit, Y::AbstractMatrix{<:Real},
                 X::AbstractArray{<:Real, 3};
                 type::Symbol = :response, mask = nothing)
    _grouped_cov_check_predict_type(type)
    Z = getLV(fit, Y, X; rotate = false, mask = mask)
    O = _build_offset(X, fit.γ)
    η = fit.β .+ O .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

fitted(fit::BetaGroupedCovFit, Y::AbstractMatrix{<:Real}, X::AbstractArray{<:Real, 3};
       mask = nothing) = predict(fit, Y, X; type = :response, mask = mask)

# ---------------------------------------------------------------------------
# GammaGroupedCovFit
# ---------------------------------------------------------------------------

"""
    predict(fit::GammaGroupedCovFit, Y, X; type=:response, mask=nothing) -> p×n matrix

In-sample fitted values at the Laplace mode with covariate offset: `type=:link` returns
`η = β + Xγ + Λẑ`; `type=:response` (= `:mean`) the mean `μ = linkinv(link, η)`.
"""
function predict(fit::GammaGroupedCovFit, Y::AbstractMatrix{<:Real},
                 X::AbstractArray{<:Real, 3};
                 type::Symbol = :response, mask = nothing)
    _grouped_cov_check_predict_type(type)
    Z = getLV(fit, Y, X; rotate = false, mask = mask)
    O = _build_offset(X, fit.γ)
    η = fit.β .+ O .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

fitted(fit::GammaGroupedCovFit, Y::AbstractMatrix{<:Real}, X::AbstractArray{<:Real, 3};
       mask = nothing) = predict(fit, Y, X; type = :response, mask = mask)

# ---------------------------------------------------------------------------
# OrdinalPerTraitCovFit
# ---------------------------------------------------------------------------

"""
    predict(fit::OrdinalPerTraitCovFit, Y, X; type=:class, mask=nothing) -> matrix

In-sample predictions at `η = β + Xγ + Λẑ`. `type=:link` returns `η` (p×n);
`type=:prob` category probabilities (p×n×Cₘₐₓ); `type=:class` / `:response` the
modal category (p×n integers).
"""
function predict(fit::OrdinalPerTraitCovFit, Y::AbstractMatrix{<:Integer},
                 X::AbstractArray{<:Real, 3};
                 type::Symbol = :class, mask = nothing)
    _grouped_cov_check_ordinal_predict_type(type)
    p, n = size(Y)
    Cmax = maximum(fit.C)
    Z = getLV(fit, Y, X; rotate = false, mask = mask)
    O = _build_offset(X, fit.γ)
    η = fit.β .+ O .+ fit.Λ * Z'
    type === :link && return η
    if type === :prob
        P = zeros(Float64, p, n, Cmax)
        @inbounds for s in 1:n, t in 1:p
            mask !== nothing && !mask[t, s] && continue
            τt = _trait_cutpoints(fit.τ, fit.C, t)
            for c in 1:fit.C[t]
                P[t, s, c] = _ord_prob(c, η[t, s], τt, fit.link)
            end
        end
        return P
    end
    M = Matrix{Int}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        mask !== nothing && !mask[t, s] && continue
        τt = _trait_cutpoints(fit.τ, fit.C, t)
        best = 1
        bestp = -1.0
        for c in 1:fit.C[t]
            pc = _ord_prob(c, η[t, s], τt, fit.link)
            pc > bestp && (bestp = pc; best = c)
        end
        M[t, s] = best
    end
    return M
end

fitted(fit::OrdinalPerTraitCovFit, Y::AbstractMatrix{<:Integer},
       X::AbstractArray{<:Real, 3}; mask = nothing) =
    predict(fit, Y, X; type = :response, mask = mask)