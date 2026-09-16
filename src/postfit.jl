# Post-fit ordination extraction for fitted GLLVMs.
#
# Loadings come from the fit; the canonical rotation is the right-singular-
# vector matrix V of Λ (SVD), sign-fixed so each rotated loading column's
# largest-magnitude entry is non-negative and columns are ordered by
# decreasing singular value. Rotating loadings (Λ → Λ V) and scores
# (Z → Z V) by the same V leaves Λ Zᵀ — hence Σ_y — unchanged.

const AnyGllvmFit = Union{
    GllvmFit, GllvmCovFit, GllvmSpeciesCovFit, FourthCornerFit, RRRFit,
    ConstrainedOrdinationFit, RowEffectFit, RowRandomFit, PoissonRandomSlopeFit,
    GaussianRandomSlopeFit, TwoLevelFit, GaussianREMLFit, PhyloGaussianFit,
    GaussianPerVarFit, SPDEGaussianFit, SPDELatentFit, PhyloGLMFit,
    CoevolutionGLMFit, EMPhyloFit, BranchREFit, RelaxedClockFit, PoissonFit,
    BinomialFit, NBFit, BetaFit, GammaFit, OrdinalFit, OrdinalPerTraitFit,
    OrdinalPerTraitCovFit, TweedieFit, StudentTFit, ExponentialFit, LognormalFit,
    MultinomialFit, TruncatedPoissonFit, TruncatedNegBin2Fit, TruncatedNegBin2PerTraitFit,
    CensoredPoissonFit, GP1Fit, NB1Fit, COMPoissonFit, BetaBinomialFit,
    BetaBinomialGroupedFit, BetaBinomialGroupedCovFit, BetaHurdleFit,
    DeltaLogNormalFit, HurdlePoissonFit, HurdleNBFit, DeltaGammaFit,
    ZIPFit, ZIPCovFit, ZINBFit, ZINBCovFit, ZIBFit, ZIBCovFit,
    NBGroupedFit, NBGroupedCovFit, BetaGroupedFit, BetaGroupedCovFit,
    GammaGroupedFit, GammaGroupedCovFit, NB1GroupedFit, NB1GroupedCovFit,
    TweedieGroupedFit, TweediePerTraitPowerFit
}

# Loadings accessor — dispatches over the fitted types.
_loadings(fit::GllvmFit)    = fit.pars.Λ
_loadings(fit::BinomialFit) = fit.Λ

function _loadings(fit)
    if hasfield(typeof(fit), :Λ)
        return fit.Λ
    elseif hasfield(typeof(fit), :Λc)
        return fit.Λc
    elseif hasfield(typeof(fit), :pars) && haskey(fit.pars, :Λ)
        return fit.pars.Λ
    else
        throw(ArgumentError("_loadings not supported for $(typeof(fit))"))
    end
end

# Canonical sign-fixed right-singular-vector rotation of Λ (p×K) -> K×K.
function _svd_rotation(Λ::AbstractMatrix)
    F = svd(Λ)                      # Λ = U S Vᵀ ; columns of V order by S↓
    V = Matrix(F.V)                 # K×K
    ΛV = Λ * V
    @inbounds for k in 1:size(V, 2)
        idx = argmax(abs.(@view ΛV[:, k]))
        if ΛV[idx, k] < 0
            @views V[:, k] .= .-V[:, k]
        end
    end
    return V
end

"""
    rotation(fit) -> K×K orthogonal matrix

Canonical rotation `R` of the latent space (sign-fixed SVD of the loadings):
`getLoadings(fit; rotate=true) == getLoadings(fit; rotate=false) * R` and
`getLV(fit, y; rotate=true) == getLV(fit, y; rotate=false) * R`. `R'R == I`.
"""
rotation(fit) = _svd_rotation(_loadings(fit))

"""
    getLoadings(fit; rotate=true) -> p×K matrix

Species loadings. `rotate=true` (default) returns them in the canonical
ordination orientation (`Λ R`, columns ordered by decreasing variance, signs
fixed); `rotate=false` returns the raw fitted `Λ`. Rotation leaves `Λ Λᵀ` (and
`Σ_y`) unchanged.
"""
function getLoadings(fit; rotate::Bool = true)
    Λ = _loadings(fit)
    return rotate ? Λ * _svd_rotation(Λ) : copy(Λ)
end

# Fitted mean μ (p×n): X·β when fixed effects are present, else zeros.
function _fitted_mean(fit::GllvmFit, y::AbstractMatrix,
                      X::Union{Nothing, AbstractArray{<:Real, 3}})
    p, n = size(y)
    β = fit.pars.β
    if X === nothing || β === nothing || length(β) == 0
        return zeros(Float64, p, n)
    end
    μ = zeros(Float64, p, n)
    q = size(X, 3)
    @inbounds for s in 1:n, t in 1:p, k in 1:q
        μ[t, s] += X[t, s, k] * β[k]
    end
    return μ
end

_has_lv_predictor(fit::GllvmFit) =
    haskey(fit.pars, :alpha_lv) && fit.pars.alpha_lv !== nothing
_has_lv_predictor(fit::BinomialFit) = fit.alpha_lv !== nothing
_has_lv_predictor(fit::PoissonFit) = fit.alpha_lv !== nothing
_has_lv_predictor(fit::NBFit) = fit.alpha_lv !== nothing
_has_lv_predictor(fit::GammaFit) = fit.alpha_lv !== nothing
_has_lv_predictor(fit::BetaFit) = fit.alpha_lv !== nothing
_has_lv_predictor(fit::OrdinalFit) = fit.alpha_lv !== nothing

function _lv_score_mean_for_fit(fit::GllvmFit, y::AbstractMatrix,
                                X_lv::Union{Nothing, AbstractMatrix})
    _, n = size(y)
    if !_has_lv_predictor(fit)
        return zeros(Float64, n, fit.model.K)
    end
    X_lv === nothing && throw(ArgumentError(
        "this fit used X_lv; provide the same X_lv to getLV, predict, fitted, or residuals"))
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
    size(X_lv, 2) == size(fit.pars.alpha_lv, 1) ||
        throw(ArgumentError(
            "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.pars.alpha_lv, 1)))"))
    return _lv_score_mean(X_lv, fit.pars.alpha_lv)
end

function _lv_score_mean_for_fit(fit::BinomialFit, y::AbstractMatrix,
                                X_lv::Union{Nothing, AbstractMatrix})
    _, n = size(y)
    K = size(fit.Λ, 2)
    if !_has_lv_predictor(fit)
        return zeros(Float64, n, K)
    end
    X_lv === nothing && throw(ArgumentError(
        "this fit used X_lv; provide the same X_lv to getLV, predict, fitted, or residuals"))
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
    size(X_lv, 2) == size(fit.alpha_lv, 1) ||
        throw(ArgumentError(
            "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.alpha_lv, 1)))"))
    return _lv_score_mean(X_lv, fit.alpha_lv)
end

function _lv_score_mean_for_fit(fit::Union{PoissonFit, NBFit, GammaFit, BetaFit, OrdinalFit},
                                y::AbstractMatrix,
                                X_lv::Union{Nothing, AbstractMatrix})
    _, n = size(y)
    K = size(fit.Λ, 2)
    if !_has_lv_predictor(fit)
        return zeros(Float64, n, K)
    end
    X_lv === nothing && throw(ArgumentError(
        "this fit used X_lv; provide the same X_lv to getLV, predict, fitted, or residuals"))
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
    size(X_lv, 2) == size(fit.alpha_lv, 1) ||
        throw(ArgumentError(
            "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.alpha_lv, 1)))"))
    return _lv_score_mean(X_lv, fit.alpha_lv)
end

"""
    getLV(fit::GllvmFit, y; X=nothing, X_lv=nothing,
          component=:total, rotate=true) -> n×K matrix

Conditional latent-variable scores (site ordination): the Gaussian posterior
mean `mₛ = (I + Λᵀ Ψ⁻¹ Λ)⁻¹ Λᵀ Ψ⁻¹ (yₛ − μₛ)`, with residual covariance
`Ψ = Σ_y − ΛΛᵀ` and `μ` the fitted mean (`X·β`, or 0 when there are no fixed
effects). `y` (and `X`, when the fit used fixed effects) must match what was
passed to `fit_gaussian_gllvm` — the fit does not store the data.

For fits with `X_lv`, `component` chooses which latent-score layer to return:
`:mean` is `X_lv * alpha_lv`, `:innovation` is the zero-mean posterior latent
score, and `:total` is their sum. `rotate=true` applies the canonical
[`rotation`](@ref) to whichever component is returned.
"""
function getLV(fit::GllvmFit, y::AbstractMatrix;
               X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
               X_lv::Union{Nothing, AbstractMatrix} = nothing,
               component::Symbol = :total,
               rotate::Bool = true,mask=nothing,offset=nothing)
    component in (:total, :innovation, :mean) ||
        throw(ArgumentError("component must be :total, :innovation, or :mean; got :$component"))
    if _has_gaussian_record(fit)
        X_lv===nothing || throw(ArgumentError("ordinary Gaussian integration does not use X_lv"))
        return _gaussian_record_predict(fit,y;X=X,mask=mask,offset=offset,component=component,rotate=rotate,scores=true)
    end
    (mask===nothing && offset===nothing) || throw(ArgumentError("mask/offset postfit requires retained Gaussian integration data"))
    Λ = fit.pars.Λ
    K = size(Λ, 2)
    Σ = sigma_y_site(fit)
    Ψ = Σ - Λ * Λ'
    Zmean = _lv_score_mean_for_fit(fit, y, X_lv)
    if component === :mean
        return rotate ? Zmean * _svd_rotation(Λ) : Zmean
    end
    R = y .- _fitted_mean(fit, y, X) .- Λ * Zmean'
    ΨiΛ = Ψ \ Λ
    M = Symmetric(I + Λ' * ΨiΛ)
    Z = M \ (ΨiΛ' * R)                  # K×n
    Zt = permutedims(Z)                 # n×K
    Zout = component === :innovation ? Zt : Zmean .+ Zt
    return rotate ? Zout * _svd_rotation(Λ) : Zout
end

"""
    getLV(fit::BinomialFit, Y; N=nothing, X_lv=nothing,
          component=:total, rotate=true) -> n×K matrix

Conditional latent-variable scores: the per-site Laplace mode `ẑₛ` (the inner
Fisher-scoring solve of the marginal). `Y` is the p×n integer response matrix;
`N` the trial counts (default all-ones, i.e. Bernoulli).

For fits with `X_lv`, `component` chooses which latent-score layer to return:
`:mean` is `X_lv * alpha_lv`, `:innovation` is the zero-mean Laplace mode, and
`:total` is their sum. `rotate=true` applies the canonical [`rotation`](@ref)
to whichever component is returned.
"""
function getLV(fit::BinomialFit, Y::AbstractMatrix;
               N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
               X_lv::Union{Nothing, AbstractMatrix} = nothing,
               component::Symbol = :total,
               rotate::Bool = true, mask = nothing,offset=nothing)
    component in (:total, :innovation, :mean) ||
        throw(ArgumentError("component must be :total, :innovation, or :mean; got :$component"))
    if _is_binomial_aghq(fit)
        X_lv===nothing || throw(ArgumentError("AGHQ loadings-only prediction does not use X_lv"))
        if component===:mean
            _binomial_aghq_problem(fit,Y;N=N,mask=mask,offset=offset)
            return zeros(size(Y,2),size(fit.Λ,2))
        end
        return _binomial_aghq_scores(fit,Y;N=N,rotate=rotate,mask=mask,offset=offset)
    end
    offset===nothing || throw(ArgumentError("explicit offset currently requires AGHQ metadata"))
    eltype(Y)<:Integer || throw(ArgumentError("Laplace binomial getLV requires integer responses"))
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    K = size(fit.Λ, 2)
    Zmean = _lv_score_mean_for_fit(fit, Y, X_lv)
    if component === :mean
        return rotate ? Zmean * _svd_rotation(fit.Λ) : Zmean
    end
    lv_offset = _has_lv_predictor(fit) ? fit.Λ * Zmean' : nothing
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = lv_offset === nothing ? nothing : view(lv_offset, :, s)
        Z[:, s] = _laplace_mode(view(Y, :, s), view(Nm, :, s), fit.Λ, fit.β, fit.link;
                                mask = mi, offset = oi)
    end
    Zt = permutedims(Z)                 # n×K
    Zout = component === :innovation ? Zt : Zmean .+ Zt
    return rotate ? Zout * _svd_rotation(fit.Λ) : Zout
end

"""
    predict(fit::GllvmFit, y; type=:response, X=nothing, X_lv=nothing) -> p×n matrix

In-sample fitted values at the conditional latent scores `ẑ` (see [`getLV`](@ref)):
`type=:link` returns the linear predictor `η = μ + Λ ẑ` (`μ` the fixed-effect
mean, `0` without `X`); `type=:response` applies the inverse link (identity for
the Gaussian family, so both types coincide). No `newdata` — `y` (and `X`) must
match the fit.
"""
function predict(fit::GllvmFit, y::AbstractMatrix;
                 type::Symbol = :response,
                 X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                 X_lv::Union{Nothing, AbstractMatrix} = nothing,mask=nothing,offset=nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    if _has_gaussian_record(fit)
        X_lv===nothing || throw(ArgumentError("ordinary Gaussian integration does not use X_lv"))
        return _gaussian_record_predict(fit,y;X=X,mask=mask,offset=offset)
    end
    (mask===nothing && offset===nothing) || throw(ArgumentError("mask/offset postfit requires retained Gaussian integration data"))
    Z = getLV(fit, y; X = X, X_lv = X_lv, component = :total, rotate = false) # n×K
    η = _fitted_mean(fit, y, X) .+ fit.pars.Λ * Z'   # p×n
    return η                                          # identity link
end

"""
    predict(fit::BinomialFit, Y; type=:response, N=nothing, X_lv=nothing) -> p×n matrix

In-sample fitted values at the Laplace conditional mode `ẑ` (see [`getLV`](@ref)):
`type=:link` returns `η = β + Λ ẑ`; `type=:response` returns the inverse-link
fitted probabilities `linkinv(link, η)`.
"""
function predict(fit::BinomialFit, Y::AbstractMatrix;
                 type::Symbol = :response,
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                 X_lv::Union{Nothing, AbstractMatrix} = nothing,mask=nothing,offset=nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    if _is_binomial_aghq(fit)
        X_lv===nothing || throw(ArgumentError("AGHQ loadings-only prediction does not use X_lv"))
        z=_binomial_aghq_scores(fit,Y;N=N,rotate=false,mask=mask,offset=offset)
        eta=fit.β .+ fit.Λ*z' .+ _aghq_prediction_offset(fit,Y,offset)
        return type===:link ? eta : _binomial_aghq_probability.(eta,Ref(fit.link))
    end
    offset===nothing || throw(ArgumentError("explicit offset currently requires AGHQ metadata"))
    Z = getLV(fit, Y; N = N, X_lv = X_lv, component = :total,
              rotate = false,mask=mask)                       # n×K
    η = fit.β .+ fit.Λ * Z'                           # p×n
    type === :link && return η
    return linkinv.(Ref(fit.link), η)
end

"""
    fitted(fit, data; kwargs...) -> p×n matrix

Response-scale in-sample fitted values — `predict(fit, data; type=:response, kwargs...)`.
"""
StatsAPI.fitted(fit::AnyGllvmFit, data; kwargs...) = predict(fit, data; type = :response, kwargs...)

"""
    residuals(fit::GllvmFit, y; type=:dunnsmyth, X=nothing, X_lv=nothing) -> p×n matrix

Conditional residuals at the predicted latent scores. For the Gaussian family the
Dunn–Smyth randomized quantile residual reduces (continuous CDF) to the
standardized residual `(y − μ) / σ_eps`, which also equals the `:pearson`
residual. `μ` is the conditional fitted mean (see [`predict`](@ref)).
"""
function residuals(fit::GllvmFit, y::AbstractMatrix;
                   type::Symbol = :dunnsmyth,
                   X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                   X_lv::Union{Nothing, AbstractMatrix} = nothing,mask=nothing,offset=nothing)
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    if _has_gaussian_record(fit)
        q,_=_gaussian_record_problem(fit,y;X=X,mask=mask,offset=offset)
        mu=predict(fit,y;X=X,X_lv=X_lv,mask=mask,offset=offset)
        return ifelse.(q.data.mask,(q.data.responses.-mu)./fit.pars.σ_eps,NaN)
    end
    (mask===nothing && offset===nothing) || throw(ArgumentError("mask/offset postfit requires retained Gaussian integration data"))
    μ = predict(fit, y; type = :response, X = X, X_lv = X_lv)
    return (y .- μ) ./ fit.pars.σ_eps
end

"""
    extract_lv_effects(fit; type=:trait_effect)

Extract predictor-informed latent-score effects from a
`fit_gaussian_gllvm(...; X_lv=...)` or
`fit_binomial_gllvm(...; X_lv=...)` fit.

- `type=:trait_effect` returns the rotation-stable `p × q_lv` matrix
  `B_lv = Λ * alpha_lv'`, the effect of each `X_lv` predictor on each trait's
  linear predictor.
- `type=:axis_effect` returns the raw `q_lv × K` `alpha_lv` matrix. These
  coefficients are the familiar constrained-ordination / CLV-style view, but
  they are latent-axis and rotation dependent unless a loading constraint or
  rotation convention is declared.

Uncertainty from `confint_lv_effects` targets only the induced trait-scale
product `B_lv = Λ * alpha_lv'`. Axis-effect SEs are not currently returned;
they need a declared rotation/constraint convention before they can be
interpreted honestly. Broader non-Gaussian / structured-source extensions remain
separate validation gates.
"""
function extract_lv_effects(fit::GllvmFit; type::Symbol = :trait_effect)
    _has_lv_predictor(fit) || throw(ArgumentError(
        "extract_lv_effects requires a fit from fit_gaussian_gllvm(...; X_lv=...)"))
    type in (:trait_effect, :axis_effect) ||
        throw(ArgumentError("type must be :trait_effect or :axis_effect; got :$type"))
    type === :axis_effect && return copy(fit.pars.alpha_lv)
    return fit.pars.Λ * fit.pars.alpha_lv'
end

lv_effects(fit::GllvmFit; kwargs...) = extract_lv_effects(fit; kwargs...)

function extract_lv_effects(fit::BinomialFit; type::Symbol = :trait_effect)
    _has_lv_predictor(fit) || throw(ArgumentError(
        "extract_lv_effects requires a fit from fit_binomial_gllvm(...; X_lv=...)"))
    type in (:trait_effect, :axis_effect) ||
        throw(ArgumentError("type must be :trait_effect or :axis_effect; got :$type"))
    type === :axis_effect && return copy(fit.alpha_lv)
    return fit.Λ * fit.alpha_lv'
end

lv_effects(fit::BinomialFit; kwargs...) = extract_lv_effects(fit; kwargs...)

function extract_lv_effects(fit::PoissonFit; type::Symbol = :trait_effect)
    _has_lv_predictor(fit) || throw(ArgumentError(
        "extract_lv_effects requires a fit from fit_poisson_gllvm(...; X_lv=...)"))
    type in (:trait_effect, :axis_effect) ||
        throw(ArgumentError("type must be :trait_effect or :axis_effect; got :$type"))
    type === :axis_effect && return copy(fit.alpha_lv)
    return fit.Λ * fit.alpha_lv'
end

lv_effects(fit::PoissonFit; kwargs...) = extract_lv_effects(fit; kwargs...)

function extract_lv_effects(fit::NBFit; type::Symbol = :trait_effect)
    _has_lv_predictor(fit) || throw(ArgumentError(
        "extract_lv_effects requires a fit from fit_nb_gllvm(...; X_lv=...)"))
    type in (:trait_effect, :axis_effect) ||
        throw(ArgumentError("type must be :trait_effect or :axis_effect; got :$type"))
    type === :axis_effect && return copy(fit.alpha_lv)
    return fit.Λ * fit.alpha_lv'
end

lv_effects(fit::NBFit; kwargs...) = extract_lv_effects(fit; kwargs...)

function extract_lv_effects(fit::GammaFit; type::Symbol = :trait_effect)
    _has_lv_predictor(fit) || throw(ArgumentError(
        "extract_lv_effects requires a fit from fit_gamma_gllvm(...; X_lv=...)"))
    type in (:trait_effect, :axis_effect) ||
        throw(ArgumentError("type must be :trait_effect or :axis_effect; got :$type"))
    type === :axis_effect && return copy(fit.alpha_lv)
    return fit.Λ * fit.alpha_lv'
end

lv_effects(fit::GammaFit; kwargs...) = extract_lv_effects(fit; kwargs...)

function extract_lv_effects(fit::BetaFit; type::Symbol = :trait_effect)
    _has_lv_predictor(fit) || throw(ArgumentError(
        "extract_lv_effects requires a fit from fit_beta_gllvm(...; X_lv=...)"))
    type in (:trait_effect, :axis_effect) ||
        throw(ArgumentError("type must be :trait_effect or :axis_effect; got :$type"))
    type === :axis_effect && return copy(fit.alpha_lv)
    return fit.Λ * fit.alpha_lv'
end

lv_effects(fit::BetaFit; kwargs...) = extract_lv_effects(fit; kwargs...)

function extract_lv_effects(fit::OrdinalFit; type::Symbol = :trait_effect)
    _has_lv_predictor(fit) || throw(ArgumentError(
        "extract_lv_effects requires a fit from fit_ordinal_gllvm(...; X_lv=...)"))
    type in (:trait_effect, :axis_effect) ||
        throw(ArgumentError("type must be :trait_effect or :axis_effect; got :$type"))
    type === :axis_effect && return copy(fit.alpha_lv)
    return fit.Λ * fit.alpha_lv'
end

lv_effects(fit::OrdinalFit; kwargs...) = extract_lv_effects(fit; kwargs...)

"""
    residuals(fit::BinomialFit, Y; type=:dunnsmyth, N=nothing, rng=Random.default_rng())
        -> p×n matrix

Conditional residuals at the predicted latent mode. `:dunnsmyth` returns Dunn–
Smyth randomized quantile residuals — `Φ⁻¹(u)`, `u` uniform on `[F(y−1), F(y)]`
under `Binomial(N, μ)` — ≈ N(0,1) under a correct model (pass a fixed `rng` for
reproducibility). `:pearson` returns `(Y − Nμ) / √(Nμ(1−μ))`.
"""
function residuals(fit::BinomialFit, Y::AbstractMatrix;
                   type::Symbol = :dunnsmyth,
                   N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                   X_lv::Union{Nothing, AbstractMatrix} = nothing,
                   rng::AbstractRNG = Random.default_rng(),mask=nothing,offset=nothing)
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    if _is_binomial_aghq(fit)
        q,_=_binomial_aghq_problem(fit,Y;N=N,mask=mask,offset=offset)
        mu=predict(fit,Y;N=N,X_lv=X_lv,mask=mask,offset=offset)
        result=fill(NaN,size(Y))
        for j in eachindex(result)
            q.data.mask[j] || continue
            nt=q.data.trials[j];y=q.data.responses[j];prob=mu[j]
            if type===:pearson
                variance=nt*prob*(1-prob)
                result[j]=variance>0 ? (y-nt*prob)/sqrt(variance) : NaN
            else
                d=Binomial(Int(nt),prob);lo=cdf(d,y-1);hi=cdf(d,y)
                result[j]=quantile(Normal(),clamp(lo+(hi-lo)*rand(rng),1e-12,1-1e-12))
            end
        end
        return result
    end
    offset===nothing || throw(ArgumentError("explicit offset currently requires AGHQ metadata"))
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    μ = predict(fit, Y; type = :response, N = N, X_lv = X_lv,mask=mask)
    if type === :pearson
        return (Y .- Nm .* μ) ./ sqrt.(Nm .* μ .* (1 .- μ))
    end
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        d = Binomial(Int(Nm[t, s]), μ[t, s])
        Flo = cdf(d, Y[t, s] - 1)
        Fhi = cdf(d, Y[t, s])
        u = Flo + (Fhi - Flo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

# ---------------------------------------------------------------------------
# Model-selection criteria + display + StatsAPI extractors.
# ---------------------------------------------------------------------------

_loglik(fit::GllvmFit)         = fit.logLik
_loglik(fit::GaussianREMLFit)  = fit.reml_loglik
_loglik(fit::PhyloGaussianFit) = -fit.negll
_loglik(fit::BinomialFit)      = fit.loglik

function _loglik(fit)
    if hasfield(typeof(fit), :loglik)
        return fit.loglik
    elseif hasfield(typeof(fit), :logLik)
        return fit.logLik
    elseif hasfield(typeof(fit), :reml_loglik)
        return fit.reml_loglik
    elseif hasfield(typeof(fit), :negll)
        return -fit.negll
    else
        throw(ArgumentError("_loglik not defined for $(typeof(fit))"))
    end
end

# Free-parameter count k (loadings counted modulo the K(K−1)/2 rotational df).
function _nparams(fit::GllvmFit)
    m = fit.model
    p = m.p
    q = if fit.pars.β === nothing
        0
    elseif haskey(fit.pars, :β_fixed)
        count(!, fit.pars.β_fixed)
    else
        length(fit.pars.β)
    end
    k = q + 1                                          # fixed effects + σ_eps
    _has_lv_predictor(fit) && (k += length(fit.pars.alpha_lv))
    k += p * m.K - div(m.K * (m.K - 1), 2)            # Λ_B
    m.K_W > 0        && (k += p * m.K_W - div(m.K_W * (m.K_W - 1), 2))
    m.has_diag       && (k += 2p)                      # σ²_B, σ²_W
    m.K_phy > 0      && (k += p * m.K_phy - div(m.K_phy * (m.K_phy - 1), 2))
    m.has_phy_unique && (k += p)                       # σ_phy
    return k
end

function _nparams(fit::BinomialFit)
    p, K = size(fit.Λ)
    k = p + (p * K - div(K * (K - 1), 2))              # β intercepts + Λ
    _has_lv_predictor(fit) && (k += length(fit.alpha_lv))
    return k
end

_nparams(f::CensoredPoissonFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); p + (p * K - div(K * (K - 1), 2)))
_nparams(f::CoevolutionGLMFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); length(f.β) + (p * K - div(K * (K - 1), 2)) + 1 + length(f.dispersion))
_nparams(f::GaussianREMLFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); (f.β === nothing ? 0 : length(f.β)) + 1 + (p * K - div(K * (K - 1), 2)))
_nparams(f::GaussianRandomSlopeFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); (p * K - div(K * (K - 1), 2)) + 1 + div(f.q * (f.q + 1), 2))
_nparams(f::LognormalFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); (p * K - div(K * (K - 1), 2)) + length(f.β) + length(f.σ))
_nparams(f::MultinomialFit) = length(f.β) + (f.γ === nothing ? 0 : length(f.γ))
_nparams(f::PhyloGLMFit) = length(f.β) + 1 + length(f.dispersion)
_nparams(f::PhyloGaussianFit) = 3
_nparams(f::PoissonRandomSlopeFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); length(f.β) + (p * K - div(K * (K - 1), 2)) + div(f.q * (f.q + 1), 2))
_nparams(f::SPDEGaussianFit) = 4
_nparams(f::StudentTFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); length(f.β) + (p * K - div(K * (K - 1), 2)) + length(f.σ) + (f.estimated_nu ? length(f.ν) : 0))
_nparams(f::TruncatedNegBin2Fit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); length(f.β) + (p * K - div(K * (K - 1), 2)) + length(f.r))
_nparams(f::TruncatedNegBin2PerTraitFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); length(f.β) + (p * K - div(K * (K - 1), 2)) + length(f.r))
_nparams(f::TruncatedPoissonFit) = (p = size(f.Λ, 1); K = size(f.Λ, 2); length(f.β) + (p * K - div(K * (K - 1), 2)))
_nparams(f::TwoLevelFit) = (p = size(f.Λ_B, 1); K_B = size(f.Λ_B, 2); K_W = size(f.Λ_W, 2); p * K_B - div(K_B * (K_B - 1), 2) + p * K_W - div(K_W * (K_W - 1), 2) + length(f.σ²_B) + length(f.σ²_W))

function _nparams(fit)
    k = 0
    if hasfield(typeof(fit), :β) && fit.β !== nothing
        k += length(fit.β)
    elseif hasfield(typeof(fit), :βc) && hasfield(typeof(fit), :βz)
        k += length(fit.βc) + length(fit.βz)
    end
    if hasfield(typeof(fit), :γ) && fit.γ !== nothing
        k += length(fit.γ)
    end
    if hasfield(typeof(fit), :Λ) && fit.Λ !== nothing
        p, K = size(fit.Λ)
        k += p * K - div(K * (K - 1), 2)
    elseif hasfield(typeof(fit), :Λc) && fit.Λc !== nothing
        p, K = size(fit.Λc)
        k += p * K - div(K * (K - 1), 2)
    end
    return max(k, 1)
end

"""
    dof(fit) -> Integer

Return the degrees of freedom (number of free estimated parameters) of `fit`.
"""
StatsAPI.dof(fit::AnyGllvmFit) = _nparams(fit)

"""
    loglikelihood(fit) -> Float64

Return the maximized marginal log-likelihood of `fit`.
"""
StatsAPI.loglikelihood(fit::AnyGllvmFit) = _loglik(fit)

"""
    aic(fit) -> Float64

Akaike information criterion `2k − 2ℓ`: `k` the free-parameter count (`dof(fit)`),
`ℓ` the maximised marginal log-likelihood (`loglikelihood(fit)`).
"""
StatsAPI.aic(fit::AnyGllvmFit) = 2 * StatsAPI.dof(fit) - 2 * StatsAPI.loglikelihood(fit)

"""
    bic(fit, n) -> Float64
    bic(fit, Y; mask = nothing) -> Float64

Bayesian information criterion `k·log(n) − 2ℓ`. `n` is a directly supplied
observation count (pass [`nobs`](@ref) explicitly if a non-default count is
wanted); the `bic(fit, Y)` form infers it as `nobs(fit, Y; mask)` — R's p·n
cell-count convention, not the number of sites.
"""
StatsAPI.bic(fit::AnyGllvmFit, n::Integer) = StatsAPI.dof(fit) * log(n) - 2 * StatsAPI.loglikelihood(fit)
StatsAPI.bic(fit::AnyGllvmFit, Y::AbstractMatrix; mask = nothing) =
    StatsAPI.bic(fit, StatsAPI.nobs(fit, Y; mask = mask))

"""
    nobs(fit, [Y]; mask = nothing) -> Integer

Number of observed response cells: R's `p·n` cell-count convention, i.e.
`length(Y)` minus any missing/masked cells — matching `nobs.gllvmTMB_multi`,
which counts `is_y_observed == 1` cells rather than the number of sites.

If `Y` is provided, returns `count(mask)` when `mask` is given, otherwise
`count(!ismissing, Y)` when `Y`'s eltype allows `missing`, otherwise
`length(Y)` (== `p * n`, complete data). `mask` is `p×n` with `true` meaning
*observed* (the same convention as `fit_*_gllvm(...; mask)`); a fit struct
does not retain the mask it was fitted with, so pass it explicitly here for a
masked fit.

For fit structs that store observation/level counts instead of a response
matrix (e.g. two-level, random-slope, SPDE), `nobs(fit)` returns the
cell-count analogue (`p` times the stored unit count) so the convention holds
without requiring `Y`.
"""
function StatsAPI.nobs(fit::AnyGllvmFit, Y::AbstractMatrix; mask = nothing)
    mask !== nothing && return count(mask)
    Missing <: eltype(Y) && return count(!ismissing, Y)
    return length(Y)
end
StatsAPI.nobs(fit::TwoLevelFit) = size(fit.Λ_B, 1) * fit.nindiv
StatsAPI.nobs(fit::GaussianRandomSlopeFit) = size(fit.Λ, 1) * fit.nlevels
StatsAPI.nobs(fit::PoissonRandomSlopeFit) = size(fit.Λ, 1) * fit.nlevels
StatsAPI.nobs(fit::RowEffectFit) = size(fit.Λ, 1) * length(fit.ρ)
StatsAPI.nobs(fit::SPDEGaussianFit) = size(fit.nodes, 1)   # univariate spatial GP (p = 1); unchanged
StatsAPI.nobs(fit::SPDELatentFit) = size(fit.Λ, 1) * size(fit.nodes, 1)
function StatsAPI.nobs(fit::AnyGllvmFit)
    if hasfield(typeof(fit), :nindiv)
        return fit.nindiv
    elseif hasfield(typeof(fit), :nlevels)
        return fit.nlevels
    elseif hasfield(typeof(fit), :nodes)
        return size(fit.nodes, 1)
    elseif hasfield(typeof(fit), :ρ)
        return length(fit.ρ)
    else
        throw(ArgumentError("nobs for $(typeof(fit)) requires the response matrix Y: nobs(fit, Y)"))
    end
end

"""
    coef(fit) -> Vector{Float64}

Return the estimated regression coefficients (fixed effects or primary parameter vector) of `fit`.
"""
StatsAPI.coef(fit::GllvmFit) = fit.pars.β === nothing ? Float64[] : copy(fit.pars.β)
StatsAPI.coef(fit::GllvmCovFit) = copy(fit.γ)
StatsAPI.coef(fit::GllvmSpeciesCovFit) = copy(fit.B)
StatsAPI.coef(fit::FourthCornerFit) = copy(fit.C)
StatsAPI.coef(fit::RRRFit) = copy(fit.B)
StatsAPI.coef(fit::ConstrainedOrdinationFit) = copy(fit.B)
StatsAPI.coef(fit::GammaGroupedCovFit) = copy(fit.γ)
StatsAPI.coef(fit::NBGroupedCovFit) = copy(fit.γ)
StatsAPI.coef(fit::BetaGroupedCovFit) = copy(fit.γ)
StatsAPI.coef(fit::NB1GroupedCovFit) = copy(fit.γ)
StatsAPI.coef(fit::BetaBinomialGroupedCovFit) = copy(fit.γ)
StatsAPI.coef(fit::OrdinalPerTraitCovFit) = copy(fit.γ)
StatsAPI.coef(fit::ZIPCovFit) = vcat(fit.γz, fit.γc)
StatsAPI.coef(fit::ZINBCovFit) = vcat(fit.γz, fit.γc)
StatsAPI.coef(fit::ZIBCovFit) = vcat(fit.γz, fit.γc)
StatsAPI.coef(fit::DeltaLogNormalFit) = vcat(fit.βz, fit.βc)
StatsAPI.coef(fit::DeltaGammaFit) = vcat(fit.βz, fit.βc)
StatsAPI.coef(fit::HurdlePoissonFit) = vcat(fit.βz, fit.βc)
StatsAPI.coef(fit::HurdleNBFit) = vcat(fit.βz, fit.βc)
StatsAPI.coef(fit::BetaHurdleFit) = vcat(fit.βz, fit.βc)
StatsAPI.coef(fit::ZIPFit) = vcat(fit.βz, fit.βc)
StatsAPI.coef(fit::ZINBFit) = vcat(fit.βz, fit.βc)
StatsAPI.coef(fit::ZIBFit) = vcat(fit.βz, fit.βc)
StatsAPI.coef(fit::OrdinalFit) = copy(fit.τ)
StatsAPI.coef(fit::PhyloGaussianFit) = [fit.μ]
StatsAPI.coef(fit::SPDEGaussianFit) = [fit.μ]
StatsAPI.coef(fit::TwoLevelFit) = Float64[]
StatsAPI.coef(fit::GaussianRandomSlopeFit) = Float64[]
function StatsAPI.coef(fit::AnyGllvmFit)
    if hasfield(typeof(fit), :γ) && fit.γ !== nothing
        return copy(fit.γ)
    elseif hasfield(typeof(fit), :β) && fit.β !== nothing
        return copy(fit.β)
    elseif hasfield(typeof(fit), :μ)
        return [fit.μ]
    else
        return Float64[]
    end
end

"""
    vcov(fit, [Y]; kwargs...) -> AbstractMatrix

Return the asymptotic variance-covariance matrix of estimated parameters for `fit`
via observed information.
"""
function StatsAPI.vcov(fit::GllvmFit; y = nothing, kwargs...)
    y_mat = y !== nothing ? y : (hasproperty(fit, :y) ? fit.y : nothing)
    y_mat === nothing && throw(ArgumentError("`y` matrix must be supplied to compute vcov for GllvmFit: vcov(fit, y)"))
    _has_gaussian_record(fit) && return _gaussian_record_vcov(fit,y_mat;kwargs...)
    ci = confint(fit; y = y_mat, kwargs...)
    return Diagonal(ci.se .^ 2)
end

function StatsAPI.vcov(fit::GllvmFit, Y::AbstractMatrix; kwargs...)
    _has_gaussian_record(fit) && return _gaussian_record_vcov(fit,Y;kwargs...)
    ci = confint(fit; y = Y, kwargs...)
    return Diagonal(ci.se .^ 2)
end

function StatsAPI.vcov(fit::AnyGllvmFit, Y::AbstractMatrix; kwargs...)
    ci = confint(fit, Y; method = :wald, kwargs...)
    return Diagonal(ci.se .^ 2)
end

"""
    stderror(fit, [Y]; kwargs...) -> Vector{Float64}

Return the standard errors of estimated parameters for `fit`.
"""
function StatsAPI.stderror(fit::GllvmFit; y = nothing, kwargs...)
    y_mat = y !== nothing ? y : (hasproperty(fit, :y) ? fit.y : nothing)
    y_mat === nothing && throw(ArgumentError("`y` matrix must be supplied to compute stderror for GllvmFit: stderror(fit, y)"))
    ci = confint(fit; y = y_mat, kwargs...)
    return copy(ci.se)
end

function StatsAPI.stderror(fit::GllvmFit, Y::AbstractMatrix; kwargs...)
    ci = confint(fit; y = Y, kwargs...)
    return copy(ci.se)
end

function StatsAPI.stderror(fit::AnyGllvmFit, Y::AbstractMatrix; kwargs...)
    ci = confint(fit, Y; method = :wald, kwargs...)
    return copy(ci.se)
end

"""
    coeftable(fit, Y; kwargs...) -> GllvmCoefTable

Return a tidy coefficient table for `fit` at response matrix `Y`.
"""
StatsAPI.coeftable(fit::AnyGllvmFit, Y::AbstractMatrix; kwargs...) = coef_table(fit, Y; kwargs...)

"""
    summary(fit) -> String

Return a concise string summary of a fitted GLLVM model.
"""
Base.summary(fit::GllvmFit) = (fit.integration===nothing ? "" : fit.integration.actual===:aghq ? "AGHQ(k=$(fit.integration.k)) " : "exact Gaussian/Laplace ($(fit.integration.reason)) ") * "Gaussian GLLVM fit (p=$(fit.model.p), K=$(fit.model.K), logLik=$(round(fit.logLik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::BinomialFit) = (fit.integration===nothing ? "" : fit.integration.actual===:aghq ? "AGHQ(k=$(fit.integration.k)) " : "Laplace ($(fit.integration.reason)) ") * "Binomial GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::PoissonFit) = (fit.integration===nothing ? "" : fit.integration.actual===:aghq ? "AGHQ(k=$(fit.integration.k)) " : "Laplace ($(fit.integration.reason)) ") * "Poisson GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::NBFit) = "NegativeBinomial GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::NB1Fit) = "NB1 GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::GP1Fit) = "GP1 GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::BetaFit) = "Beta GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::GammaFit) = "Gamma GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::OrdinalFit) = "Ordinal GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::TweedieFit) = "Tweedie GLLVM fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::GaussianREMLFit) = "Gaussian REML fit (p=$(size(fit.Λ, 1)), K=$(size(fit.Λ, 2)), reml_loglik=$(round(fit.reml_loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::PhyloGaussianFit) = "PhyloGaussian fit (logLik=$(round(-fit.negll; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::GllvmCovFit) = "GLLVM Covariates fit (logLik=$(round(fit.loglik; sigdigits=5)), dof=$(_nparams(fit)), AIC=$(round(aic(fit); sigdigits=5)))"
Base.summary(fit::TwoLevelFit) = "TwoLevel GLLVM fit (p=$(size(fit.Λ_B, 1)), K_B=$(size(fit.Λ_B, 2)), K_W=$(size(fit.Λ_W, 2)), logLik=$(round(fit.loglik; sigdigits=5)), AIC=$(round(aic(fit); sigdigits=5)))"


# Rich REPL display (the idiomatic "summary").
function Base.show(io::IO, ::MIME"text/plain", fit::GllvmFit)
    println(io, "Gaussian GLLVM fit")
    if fit.integration!==nothing
        i=fit.integration
        println(io,"  integration = ",i.actual===:aghq ? "AGHQ(k=$(i.k), observed, unpenalized)" : "exact Gaussian/Laplace ($(i.reason))")
    end
    println(io, "  responses p = ", fit.model.p, ", latent factors K = ", fit.model.K)
    println(io, "  logLik = ", round(fit.logLik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.n_iter, " iterations)")
end

function Base.show(io::IO, ::MIME"text/plain", fit::BinomialFit)
    p, K = size(fit.Λ)
    println(io, summary(fit))
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

Base.show(io::IO, fit::GllvmFit) =
    print(io, "GllvmFit(p=", fit.model.p, ", K=", fit.model.K,
          ", logLik=", round(fit.logLik; sigdigits = 6),
          fit.converged ? "" : ", NOT CONVERGED", ")")

# ---------------------------------------------------------------------------
# Poisson post-fit methods (parallel to Binomial; counts via the log link).
# ---------------------------------------------------------------------------

_loadings(fit::PoissonFit) = fit.Λ
_loglik(fit::PoissonFit)   = fit.loglik

function _nparams(fit::PoissonFit)
    p, K = size(fit.Λ)
    k = p + (p * K - div(K * (K - 1), 2))              # β intercepts + Λ
    _has_lv_predictor(fit) && (k += length(fit.alpha_lv))
    return k
end

"""
    getLV(fit::PoissonFit, Y; N=nothing, X_lv=nothing,
          component=:total, rotate=true) -> n×K matrix

Conditional latent-variable scores for a Poisson fit: the per-site Laplace mode
`ẑₛ`. `Y` is the p×n integer count matrix; `rotate=true` applies the canonical
[`rotation`](@ref). (`N` is accepted for signature symmetry and ignored.)

For fits with `X_lv`, `component` chooses which latent-score layer to return:
`:mean` is `X_lv * alpha_lv`, `:innovation` is the zero-mean Laplace mode, and
`:total` is their sum.
"""
function getLV(fit::PoissonFit, Y::AbstractMatrix;
               N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
               X_lv::Union{Nothing, AbstractMatrix} = nothing,
               component::Symbol = :total,
               rotate::Bool = true, mask = nothing, offset=nothing)
    component in (:total, :innovation, :mean) ||
        throw(ArgumentError("component must be :total, :innovation, or :mean; got :$component"))
    if _is_poisson_aghq(fit)
        X_lv===nothing || throw(ArgumentError("AGHQ loadings-only prediction does not use X_lv"))
        if component===:mean
            _poisson_aghq_problem(fit,Y;mask=mask,offset=offset)
            return zeros(size(Y,2),size(fit.Λ,2))
        end
        return _poisson_aghq_scores(fit,Y;rotate=rotate,mask=mask,offset=offset)
    end
    offset===nothing || throw(ArgumentError("explicit offset in getLV is currently carried by AGHQ fits only"))
    eltype(Y)<:Integer || throw(ArgumentError("Laplace Poisson getLV currently requires integer responses"))
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    K = size(fit.Λ, 2)
    Zmean = _lv_score_mean_for_fit(fit, Y, X_lv)
    if component === :mean
        return rotate ? Zmean * _svd_rotation(fit.Λ) : Zmean
    end
    lv_offset = _has_lv_predictor(fit) ? fit.Λ * Zmean' : nothing
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = lv_offset === nothing ? nothing : view(lv_offset, :, s)
        Z[:, s] = _laplace_mode(Poisson(), view(Y, :, s), view(Nm, :, s), fit.Λ,
                                fit.β, fit.link; mask = mi, offset = oi)
    end
    Zt = permutedims(Z)
    Zout = component === :innovation ? Zt : Zmean .+ Zt
    return rotate ? Zout * _svd_rotation(fit.Λ) : Zout
end

"""
    predict(fit::PoissonFit, Y; type=:response, N=nothing, X_lv=nothing) -> p×n matrix

In-sample fitted values at the Laplace mode: `type=:link` returns `η = β + Λ ẑ`;
`type=:response` the inverse-link fitted rates `linkinv(link, η) = exp(η)`. For
fits that used `X_lv`, pass the same predictor matrix.
"""
function predict(fit::PoissonFit, Y::AbstractMatrix;
                 type::Symbol = :response,
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                 X_lv::Union{Nothing, AbstractMatrix} = nothing, mask=nothing,offset=nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    if _is_poisson_aghq(fit)
        X_lv===nothing || throw(ArgumentError("AGHQ loadings-only prediction does not use X_lv"))
        q,_=_poisson_aghq_problem(fit,Y;mask=mask,offset=offset)
        z=_poisson_aghq_scores(fit,Y;rotate=false,mask=mask,offset=offset)
        eta=fit.β .+ fit.Λ*z' .+ _aghq_prediction_offset(fit,Y,offset)
        return type===:link ? eta : exp.(eta)
    end
    offset===nothing || throw(ArgumentError("explicit prediction offset currently requires AGHQ metadata"))
    Z = getLV(fit, Y; N = N, X_lv = X_lv, component = :total, rotate = false,mask=mask)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), η)
end

"""
    residuals(fit::PoissonFit, Y; type=:dunnsmyth, rng=Random.default_rng()) -> p×n matrix

Conditional residuals for a Poisson fit. `:dunnsmyth` returns Dunn–Smyth
randomized quantile residuals — `Φ⁻¹(u)`, `u` uniform on `[F(y−1), F(y)]` under
`Poisson(μ)` — ≈ N(0,1) under a correct model (pass a fixed `rng` to reproduce).
`:pearson` returns `(Y − μ) / √μ`.
"""
function residuals(fit::PoissonFit, Y::AbstractMatrix;
                   type::Symbol = :dunnsmyth,
                   X_lv::Union{Nothing, AbstractMatrix} = nothing,
                   rng::AbstractRNG = Random.default_rng(),mask=nothing,offset=nothing)
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    p, n = size(Y)
    μ = predict(fit,Y;type=:response,X_lv=X_lv,mask=mask,offset=offset)
    if _is_poisson_aghq(fit)
        q,_=_poisson_aghq_problem(fit,Y;mask=mask,offset=offset)
        result=fill(NaN,size(Y))
        for s in 1:n,t in 1:p
            q.data.mask[t,s] || continue
            y=q.data.responses[t,s];mu=μ[t,s]
            if type===:pearson
                result[t,s]=(y-mu)/sqrt(mu)
            else
                d=Poisson(mu);lo=cdf(d,y-1);hi=cdf(d,y)
                result[t,s]=quantile(Normal(),clamp(lo+(hi-lo)*rand(rng),1e-12,1-1e-12))
            end
        end
        return result
    end
    if type === :pearson
        return (Y .- μ) ./ sqrt.(μ)
    end
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        d = Poisson(μ[t, s])
        Flo = cdf(d, Y[t, s] - 1)
        Fhi = cdf(d, Y[t, s])
        u = Flo + (Fhi - Flo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::PoissonFit)
    p, K = size(fit.Λ)
    println(io, "Poisson GLLVM fit")
    if fit.integration!==nothing
        i=fit.integration
        println(io,"  integration = ",i.actual===:aghq ? "AGHQ(k=$(i.k), observed, unpenalized)" : "Laplace ($(i.reason))")
    end
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Negative-binomial post-fit methods (parallel to Poisson; counts with
# dispersion r — Var = μ + μ²/r — via the log link).
# ---------------------------------------------------------------------------

_loadings(fit::NBFit) = fit.Λ
_loglik(fit::NBFit)   = fit.loglik

function _nparams(fit::NBFit)
    p, K = size(fit.Λ)
    k = p + (p * K - div(K * (K - 1), 2)) + 1          # β + Λ + dispersion r
    _has_lv_predictor(fit) && (k += length(fit.alpha_lv))
    return k
end

"""
    getLV(fit::NBFit, Y; N=nothing, X_lv=nothing,
          component=:total, rotate=true) -> n×K matrix

Conditional latent-variable scores for a negative-binomial fit: the per-site
Laplace mode `ẑₛ` (computed at the fitted dispersion `r`). `rotate=true` applies
the canonical [`rotation`](@ref). For fits with `X_lv`, `component` chooses the
layer: `:mean` is `X_lv * alpha_lv`, `:innovation` the zero-mean Laplace mode,
`:total` their sum.
"""
function getLV(fit::NBFit, Y::AbstractMatrix{<:Integer};
               N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
               X_lv::Union{Nothing, AbstractMatrix} = nothing,
               component::Symbol = :total,
               rotate::Bool = true, mask = nothing)
    component in (:total, :innovation, :mean) ||
        throw(ArgumentError("component must be :total, :innovation, or :mean; got :$component"))
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    K = size(fit.Λ, 2)
    fam = NegativeBinomial(fit.r, 0.5)
    Zmean = _lv_score_mean_for_fit(fit, Y, X_lv)
    if component === :mean
        return rotate ? Zmean * _svd_rotation(fit.Λ) : Zmean
    end
    lv_offset = _has_lv_predictor(fit) ? fit.Λ * Zmean' : nothing
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = lv_offset === nothing ? nothing : view(lv_offset, :, s)
        Z[:, s] = _laplace_mode(fam, view(Y, :, s), view(Nm, :, s), fit.Λ,
                                fit.β, fit.link; mask = mi, offset = oi)
    end
    Zt = permutedims(Z)
    Zout = component === :innovation ? Zt : Zmean .+ Zt
    return rotate ? Zout * _svd_rotation(fit.Λ) : Zout
end

"""
    predict(fit::NBFit, Y; type=:response, N=nothing) -> p×n matrix

In-sample fitted values at the Laplace mode: `type=:link` returns `η = β + Λ ẑ`;
`type=:response` the inverse-link fitted means `linkinv(link, η) = exp(η)`.
"""
function predict(fit::NBFit, Y::AbstractMatrix{<:Integer};
                 type::Symbol = :response,
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                 X_lv::Union{Nothing, AbstractMatrix} = nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; N = N, X_lv = X_lv, component = :total, rotate = false)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), η)
end

"""
    residuals(fit::NBFit, Y; type=:dunnsmyth, rng=Random.default_rng()) -> p×n matrix

Conditional residuals for a negative-binomial fit. `:dunnsmyth` returns Dunn–Smyth
randomized quantile residuals — `Φ⁻¹(u)`, `u` uniform on `[F(y−1), F(y)]` under
`NegativeBinomial(r, r/(r+μ))` — ≈ N(0,1) under a correct model (pass a fixed
`rng` to reproduce). `:pearson` returns `(Y − μ) / √(μ + μ²/r)`.
"""
function residuals(fit::NBFit, Y::AbstractMatrix{<:Integer};
                   type::Symbol = :dunnsmyth,
                   X_lv::Union{Nothing, AbstractMatrix} = nothing,
                   rng::AbstractRNG = Random.default_rng())
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    p, n = size(Y)
    r = fit.r
    μ = predict(fit, Y; type = :response, X_lv = X_lv)
    if type === :pearson
        return (Y .- μ) ./ sqrt.(μ .+ μ .^ 2 ./ r)
    end
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        d = NegativeBinomial(r, r / (r + μ[t, s]))
        Flo = cdf(d, Y[t, s] - 1)
        Fhi = cdf(d, Y[t, s])
        u = Flo + (Fhi - Flo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::NBFit)
    p, K = size(fit.Λ)
    println(io, "Negative-binomial GLLVM fit")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)), ", dispersion r = ", round(fit.r; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# NB1 (negative binomial type-1, linear variance Var = μ(1+φ)) post-fit methods —
# a mirror of NBFit with the mean-dependent size r = μ/φ, constant prob 1/(1+φ).
# ---------------------------------------------------------------------------

_loadings(fit::NB1Fit) = fit.Λ
_loglik(fit::NB1Fit)   = fit.loglik

function _nparams(fit::NB1Fit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + 1       # β + Λ + dispersion φ
end

function getLV(fit::NB1Fit, Y::AbstractMatrix{<:Integer};
               N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
               rotate::Bool = true, mask = nothing)
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    K = size(fit.Λ, 2)
    fam = NB1(fit.φ)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        Z[:, s] = _laplace_mode(fam, view(Y, :, s), view(Nm, :, s), fit.Λ,
                                fit.β, fit.link; mask = mi)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

function predict(fit::NB1Fit, Y::AbstractMatrix{<:Integer};
                 type::Symbol = :response,
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; N = N, rotate = false)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), η)
end

"""
    residuals(fit::NB1Fit, Y; type=:dunnsmyth, rng=Random.default_rng()) -> p×n matrix

Conditional residuals for an NB1 fit. `:dunnsmyth` returns Dunn–Smyth randomized
quantile residuals under `NegativeBinomial(μ/φ, 1/(1+φ))`; `:pearson` returns
`(Y − μ) / √(μ(1+φ))`.
"""
function residuals(fit::NB1Fit, Y::AbstractMatrix{<:Integer};
                   type::Symbol = :dunnsmyth,
                   rng::AbstractRNG = Random.default_rng())
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    p, n = size(Y)
    φ = fit.φ
    μ = predict(fit, Y; type = :response)
    if type === :pearson
        return (Y .- μ) ./ sqrt.(μ .* (1 + φ))
    end
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        d = NegativeBinomial(μ[t, s] / φ, 1 / (1 + φ))
        Flo = cdf(d, Y[t, s] - 1)
        Fhi = cdf(d, Y[t, s])
        u = Flo + (Fhi - Flo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::NB1Fit)
    p, K = size(fit.Λ)
    println(io, "Negative-binomial type-1 (NB1) GLLVM fit")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)), ", dispersion φ = ", round(fit.φ; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# GP-1 (generalized Poisson type-1, Var = μ(1+α μ)², signed dispersion α) post-fit
# methods — mirror the NB1 block; Dunn–Smyth uses the summed GP-1 CDF (_gp1_cdf).
# ---------------------------------------------------------------------------

_loadings(fit::GP1Fit) = fit.Λ
_loglik(fit::GP1Fit)   = fit.loglik

function _nparams(fit::GP1Fit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + 1       # β + Λ + dispersion α
end

function getLV(fit::GP1Fit, Y::AbstractMatrix{<:Integer};
               N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
               rotate::Bool = true, mask = nothing)
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    K = size(fit.Λ, 2)
    fam = GeneralizedPoisson1(fit.α)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        Z[:, s] = _laplace_mode(fam, view(Y, :, s), view(Nm, :, s), fit.Λ,
                                fit.β, fit.link; mask = mi)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

function predict(fit::GP1Fit, Y::AbstractMatrix{<:Integer};
                 type::Symbol = :response,
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; N = N, rotate = false)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), η)
end

"""
    residuals(fit::GP1Fit, Y; type=:dunnsmyth, rng=Random.default_rng()) -> p×n matrix

Conditional residuals for a GP-1 fit. `:dunnsmyth` returns Dunn–Smyth randomized
quantile residuals under the fitted GP-1 pmf (CDF summed from the family log-pmf);
`:pearson` returns `(Y − μ) / √(μ(1+α μ)²)`.
"""
function residuals(fit::GP1Fit, Y::AbstractMatrix{<:Integer};
                   type::Symbol = :dunnsmyth,
                   rng::AbstractRNG = Random.default_rng())
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    p, n = size(Y)
    fam = GeneralizedPoisson1(fit.α)
    μ = predict(fit, Y; type = :response)
    if type === :pearson
        return (Y .- μ) ./ sqrt.(μ .* (1 .+ fit.α .* μ) .^ 2)
    end
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        Flo = _gp1_cdf(fam, μ[t, s], Y[t, s] - 1)
        Fhi = _gp1_cdf(fam, μ[t, s], Y[t, s])
        u = Flo + (Fhi - Flo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::GP1Fit)
    p, K = size(fit.Λ)
    println(io, "Generalized-Poisson type-1 (GP-1) GLLVM fit")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)), ", dispersion α = ", round(fit.α; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Beta post-fit methods (proportions in (0,1); mean μ = logistic(η), precision
# φ — Var = μ(1−μ)/(1+φ) — via the logit link). Responses are continuous, so the
# Dunn–Smyth residual reduces to the (deterministic) PIT, as in the Gaussian case.
# ---------------------------------------------------------------------------

_loadings(fit::BetaFit) = fit.Λ
_loglik(fit::BetaFit)   = fit.loglik

function _nparams(fit::BetaFit)
    p, K = size(fit.Λ)
    k = p + (p * K - div(K * (K - 1), 2)) + 1          # β + Λ + precision φ
    _has_lv_predictor(fit) && (k += length(fit.alpha_lv))
    return k
end

"""
    getLV(fit::BetaFit, Y; rotate=true) -> n×K matrix

Conditional latent-variable scores for a Beta fit: the per-site Laplace mode `ẑₛ`
(computed at the fitted precision `φ`). `Y` is the p×n matrix of proportions in
(0,1); `rotate=true` applies the canonical [`rotation`](@ref).
"""
function getLV(fit::BetaFit, Y::AbstractMatrix{<:Real};
               X_lv::Union{Nothing, AbstractMatrix} = nothing,
               component::Symbol = :total,
               rotate::Bool = true, mask = nothing)
    component in (:total, :innovation, :mean) ||
        throw(ArgumentError("component must be :total, :innovation, or :mean; got :$component"))
    p, n = size(Y)
    K = size(fit.Λ, 2)
    fam = Beta(fit.φ, 1.0)
    ones_p = ones(Int, p)
    Zmean = _lv_score_mean_for_fit(fit, Y, X_lv)
    if component === :mean
        return rotate ? Zmean * _svd_rotation(fit.Λ) : Zmean
    end
    lv_offset = _has_lv_predictor(fit) ? fit.Λ * Zmean' : nothing
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = lv_offset === nothing ? nothing : view(lv_offset, :, s)
        Z[:, s] = _laplace_mode(fam, view(Y, :, s), ones_p, fit.Λ, fit.β, fit.link;
                                mask = mi, offset = oi)
    end
    Zt = permutedims(Z)
    Zout = component === :innovation ? Zt : Zmean .+ Zt
    return rotate ? Zout * _svd_rotation(fit.Λ) : Zout
end

"""
    predict(fit::BetaFit, Y; type=:response) -> p×n matrix

In-sample fitted values at the Laplace mode: `type=:link` returns `η = β + Λ ẑ`;
`type=:response` the inverse-link fitted means `linkinv(link, η) = logistic(η)`
(proportions in (0,1)).
"""
function predict(fit::BetaFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response,
                 X_lv::Union{Nothing, AbstractMatrix} = nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; X_lv = X_lv, component = :total, rotate = false)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), η)
end

"""
    residuals(fit::BetaFit, Y; type=:dunnsmyth) -> p×n matrix

Conditional residuals for a Beta fit. The Beta CDF is continuous, so the
`:dunnsmyth` randomized quantile residual reduces to the deterministic PIT
`Φ⁻¹(F(y))` under `Beta(μφ, (1−μ)φ)` — ≈ N(0,1) under a correct model — exactly as
in the Gaussian case. `:pearson` returns `(Y − μ) / √(μ(1−μ)/(1+φ))`.
"""
function residuals(fit::BetaFit, Y::AbstractMatrix{<:Real}; type::Symbol = :dunnsmyth,
                   X_lv::Union{Nothing, AbstractMatrix} = nothing)
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    p, n = size(Y)
    φ = fit.φ
    μ = predict(fit, Y; type = :response, X_lv = X_lv)
    if type === :pearson
        return (Y .- μ) ./ sqrt.(μ .* (1 .- μ) ./ (1 + φ))
    end
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        d = Beta(μ[t, s] * φ, (1 - μ[t, s]) * φ)
        u = cdf(d, clamp(float(Y[t, s]), 1e-12, 1 - 1e-12))
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::BetaFit)
    p, K = size(fit.Λ)
    println(io, "Beta GLLVM fit")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)), ", precision φ = ", round(fit.φ; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Ordinal post-fit methods (ordered categories 1:C; cumulative logit, common
# ordered cutpoints τ; latent η = (Λz)_t, no intercept). The "fitted value" is
# the modal category; residuals are Dunn–Smyth randomized quantile (discrete CDF).
# ---------------------------------------------------------------------------

_loadings(fit::OrdinalFit) = fit.Λ
_loglik(fit::OrdinalFit)   = fit.loglik
_loadings(fit::OrdinalPerTraitFit) = fit.Λ
_loglik(fit::OrdinalPerTraitFit)   = fit.loglik
_loadings(fit::OrdinalPerTraitCovFit) = fit.Λ
_loglik(fit::OrdinalPerTraitCovFit)   = fit.loglik

function _nparams(fit::OrdinalFit)
    p, K = size(fit.Λ)
    k = (p * K - div(K * (K - 1), 2)) + (fit.C - 1)   # Λ + cutpoints, no β
    _has_lv_predictor(fit) && (k += length(fit.alpha_lv))
    return k
end
function _nparams(fit::OrdinalPerTraitFit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + sum(fit.C .- 2)
end
function _nparams(fit::OrdinalPerTraitCovFit)
    p, K = size(fit.Λ)
    return p + count(!, fit.γ_fixed) + (p * K - div(K * (K - 1), 2)) + sum(fit.C .- 2)
end

"""
    getLV(fit::OrdinalFit, Y; rotate=true) -> n×K matrix

Conditional latent-variable scores for an ordinal fit: the per-site Laplace mode
`ẑₛ` (at the fitted cutpoints). `Y` is the p×n matrix of ordinal responses (`1:C`);
`rotate=true` applies the canonical [`rotation`](@ref).
"""
function getLV(fit::OrdinalFit, Y::AbstractMatrix{<:Integer};
               X_lv::Union{Nothing, AbstractMatrix} = nothing,
               component::Symbol = :total,
               rotate::Bool = true, mask = nothing)
    component in (:total, :innovation, :mean) ||
        throw(ArgumentError("component must be :total, :innovation, or :mean; got :$component"))
    p, n = size(Y)
    K = size(fit.Λ, 2)
    Zmean = _lv_score_mean_for_fit(fit, Y, X_lv)
    if component === :mean
        return rotate ? Zmean * _svd_rotation(fit.Λ) : Zmean
    end
    lv_offset = _has_lv_predictor(fit) ? fit.Λ * Zmean' : nothing
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = lv_offset === nothing ? nothing : view(lv_offset, :, s)
        Z[:, s] = _ordinal_laplace_mode(view(Y, :, s), fit.Λ, fit.τ, fit.link;
                                        mask = mi, offset = oi)
    end
    Zt = permutedims(Z)
    Zout = component === :innovation ? Zt : Zmean .+ Zt
    return rotate ? Zout * _svd_rotation(fit.Λ) : Zout
end
function getLV(fit::OrdinalPerTraitFit, Y::AbstractMatrix{<:Integer};
               rotate::Bool = true, mask = nothing)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        Z[:, s] = _ordinal_laplace_mode_pertrait(view(Y, :, s), fit.Λ, fit.β,
                                                 fit.τ, fit.C, fit.link; mask = mi)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    getLV(fit::OrdinalPerTraitCovFit, Y, X; rotate=true, mask=nothing) -> n×K matrix

Conditional latent scores at `η = β + Xγ + Λz` with per-trait ordinal cutpoints.
"""
function getLV(fit::OrdinalPerTraitCovFit, Y::AbstractMatrix{<:Integer},
               X::AbstractArray{<:Real, 3};
               rotate::Bool = true, mask = nothing)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    O = _build_offset(X, fit.γ)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = view(O, :, s)
        Z[:, s] = _ordinal_laplace_mode_pertrait(view(Y, :, s), fit.Λ, fit.β,
                                                 fit.τ, fit.C, fit.link;
                                                 mask = mi, offset = oi)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::OrdinalFit, Y; type=:class) -> matrix or p×n×C array

In-sample predictions at the Laplace mode `ẑ` (η = Λẑ). `type=:link` returns the
linear predictor `η` (p×n); `type=:prob` the category probabilities (p×n×C array,
summing to 1 over the last axis); `type=:class` / `:response` the modal category
(p×n integer matrix).
"""
function predict(fit::OrdinalFit, Y::AbstractMatrix{<:Integer};
                 X_lv::Union{Nothing, AbstractMatrix} = nothing,
                 type::Symbol = :class)
    type in (:link, :prob, :class, :response) ||
        throw(ArgumentError("type must be :link, :prob, :class, or :response; got :$type"))
    p, n = size(Y); C = fit.C
    Z = getLV(fit, Y; X_lv = X_lv, rotate = false)
    η = fit.Λ * Z'                                   # p×n
    type === :link && return η
    if type === :prob
        P = Array{Float64, 3}(undef, p, n, C)
        @inbounds for s in 1:n, t in 1:p, c in 1:C
            P[t, s, c] = _ord_prob(c, η[t, s], fit.τ, fit.link)
        end
        return P
    end
    M = Matrix{Int}(undef, p, n)                     # modal category
    @inbounds for s in 1:n, t in 1:p
        best = 1; bestp = -1.0
        for c in 1:C
            pc = _ord_prob(c, η[t, s], fit.τ, fit.link)
            pc > bestp && (bestp = pc; best = c)
        end
        M[t, s] = best
    end
    return M
end
function predict(fit::OrdinalPerTraitFit, Y::AbstractMatrix{<:Integer};
                 type::Symbol = :class)
    type in (:link, :prob, :class, :response) ||
        throw(ArgumentError("type must be :link, :prob, :class, or :response; got :$type"))
    p, n = size(Y)
    Cmax = maximum(fit.C)
    Z = getLV(fit, Y; rotate = false)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    if type === :prob
        P = zeros(Float64, p, n, Cmax)
        @inbounds for s in 1:n, t in 1:p
            τt = _trait_cutpoints(fit.τ, fit.C, t)
            for c in 1:fit.C[t]
                P[t, s, c] = _ord_prob(c, η[t, s], τt, fit.link)
            end
        end
        return P
    end
    M = Matrix{Int}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
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

"""
    residuals(fit::OrdinalFit, Y; type=:dunnsmyth, rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals for an ordinal fit — `Φ⁻¹(u)`, `u` uniform
on `[P(Y≤c−1), P(Y≤c)]` under the fitted cumulative-logit model at the Laplace mode
— ≈ N(0,1) under a correct model (pass a fixed `rng` to reproduce). Only
`:dunnsmyth` is defined for ordered categories.
"""
function residuals(fit::OrdinalFit, Y::AbstractMatrix{<:Integer};
                   type::Symbol = :dunnsmyth, rng::AbstractRNG = Random.default_rng())
    type === :dunnsmyth ||
        throw(ArgumentError("ordinal residuals support type=:dunnsmyth only; got :$type"))
    p, n = size(Y); C = fit.C
    Z = getLV(fit, Y; rotate = false)
    η = fit.Λ * Z'
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        c = Int(Y[t, s])
        Fhi = c >= C ? 1.0 : _ord_F(fit.τ[c] - η[t, s], fit.link)
        Flo = c <= 1 ? 0.0 : _ord_F(fit.τ[c - 1] - η[t, s], fit.link)
        u = Flo + (Fhi - Flo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end
function residuals(fit::OrdinalPerTraitFit, Y::AbstractMatrix{<:Integer};
                   type::Symbol = :dunnsmyth, rng::AbstractRNG = Random.default_rng())
    type === :dunnsmyth ||
        throw(ArgumentError("ordinal residuals support type=:dunnsmyth only; got :$type"))
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    η = fit.β .+ fit.Λ * Z'
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        τt = _trait_cutpoints(fit.τ, fit.C, t)
        c = Int(Y[t, s])
        Fhi = c >= fit.C[t] ? 1.0 : _ord_F(τt[c] - η[t, s], fit.link)
        Flo = c <= 1 ? 0.0 : _ord_F(τt[c - 1] - η[t, s], fit.link)
        u = Flo + (Fhi - Flo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::OrdinalFit)
    p, K = size(fit.Λ)
    println(io, "Ordinal GLLVM fit (cumulative logit)")
    println(io, "  responses p = ", p, ", latent factors K = ", K, ", categories C = ", fit.C)
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end
function Base.show(io::IO, ::MIME"text/plain", fit::OrdinalPerTraitFit)
    p, K = size(fit.Λ)
    println(io, "Ordinal GLLVM fit (per-trait cutpoints)")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", categories C = ", fit.C)
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Gamma post-fit methods (positive continuous; mean μ = exp(η), shape α —
# Var = μ²/α — via the log link). Responses are continuous, so the Dunn–Smyth
# residual reduces to the deterministic PIT, as in the Gaussian and Beta cases.
# ---------------------------------------------------------------------------

_loadings(fit::GammaFit) = fit.Λ
_loglik(fit::GammaFit)   = fit.loglik

function _nparams(fit::GammaFit)
    p, K = size(fit.Λ)
    k = p + (p * K - div(K * (K - 1), 2)) + 1          # β + Λ + shape α
    _has_lv_predictor(fit) && (k += length(fit.alpha_lv))
    return k
end

"""
    getLV(fit::GammaFit, Y; rotate=true) -> n×K matrix

Conditional latent-variable scores for a Gamma fit: the per-site Laplace mode `ẑₛ`
(computed at the fitted shape `α`). `Y` is the p×n matrix of positive reals;
`rotate=true` applies the canonical [`rotation`](@ref).
"""
function getLV(fit::GammaFit, Y::AbstractMatrix{<:Real};
               X_lv::Union{Nothing, AbstractMatrix} = nothing,
               component::Symbol = :total,
               rotate::Bool = true, mask = nothing)
    component in (:total, :innovation, :mean) ||
        throw(ArgumentError("component must be :total, :innovation, or :mean; got :$component"))
    p, n = size(Y)
    K = size(fit.Λ, 2)
    fam = Gamma(fit.α, 1.0)
    ones_p = ones(Int, p)
    Zmean = _lv_score_mean_for_fit(fit, Y, X_lv)
    if component === :mean
        return rotate ? Zmean * _svd_rotation(fit.Λ) : Zmean
    end
    lv_offset = _has_lv_predictor(fit) ? fit.Λ * Zmean' : nothing
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = lv_offset === nothing ? nothing : view(lv_offset, :, s)
        Z[:, s] = _laplace_mode(fam, view(Y, :, s), ones_p, fit.Λ, fit.β, fit.link;
                                mask = mi, offset = oi)
    end
    Zt = permutedims(Z)
    Zout = component === :innovation ? Zt : Zmean .+ Zt
    return rotate ? Zout * _svd_rotation(fit.Λ) : Zout
end

"""
    predict(fit::GammaFit, Y; type=:response) -> p×n matrix

In-sample fitted values at the Laplace mode: `type=:link` returns `η = β + Λ ẑ`;
`type=:response` the inverse-link fitted means `linkinv(link, η) = exp(η)` (positive reals).
"""
function predict(fit::GammaFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response,
                 X_lv::Union{Nothing, AbstractMatrix} = nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; X_lv = X_lv, component = :total, rotate = false)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), η)
end

"""
    residuals(fit::GammaFit, Y; type=:dunnsmyth) -> p×n matrix

Conditional residuals for a Gamma fit. The Gamma CDF is continuous, so the
`:dunnsmyth` randomized quantile residual reduces to the deterministic PIT
`Φ⁻¹(F(y))` under `Gamma(α, μ/α)` — ≈ N(0,1) under a correct model — exactly as
in the Gaussian and Beta cases. `:pearson` returns `(Y − μ) / √(μ²/α)`.
"""
function residuals(fit::GammaFit, Y::AbstractMatrix{<:Real}; type::Symbol = :dunnsmyth,
                   X_lv::Union{Nothing, AbstractMatrix} = nothing)
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    p, n = size(Y)
    α = fit.α
    μ = predict(fit, Y; type = :response, X_lv = X_lv)
    if type === :pearson
        return (Y .- μ) ./ sqrt.(μ .^ 2 ./ α)
    end
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        d = Gamma(α, μ[t, s] / α)
        u = cdf(d, max(float(Y[t, s]), 1e-300))
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

# --- Exponential post-fit (positive continuous, Var = μ², no dispersion) ---
_loadings(fit::ExponentialFit) = fit.Λ
_loglik(fit::ExponentialFit)   = fit.loglik
_nparams(fit::ExponentialFit)  = (p = size(fit.Λ, 1); K = size(fit.Λ, 2); p + (p * K - div(K * (K - 1), 2)))

function getLV(fit::ExponentialFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λ, 2)
    fam = Exponential(1.0); ones_p = ones(Int, p)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _laplace_mode(fam, view(Y, :, s), ones_p, fit.Λ, fit.β, fit.link)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

function predict(fit::ExponentialFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; rotate = false)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    # clamp η before the (exp) inverse link, matching the inner mode solver
    # (_clamp_eta) and the other predict methods: an extreme conditional mode
    # must not over/underflow μ (Exponential(0) is invalid; Inf corrupts residuals).
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

"""
    residuals(fit::ExponentialFit, Y; type=:dunnsmyth) -> p×n matrix

`:dunnsmyth` randomized-quantile (here deterministic PIT, the Exponential CDF being
continuous) `Φ⁻¹(F(y))` under `Exponential(μ)`; `:pearson` returns `(Y − μ)/μ`.
"""
function residuals(fit::ExponentialFit, Y::AbstractMatrix{<:Real}; type::Symbol = :dunnsmyth)
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    p, n = size(Y)
    μ = predict(fit, Y; type = :response)
    type === :pearson && return (Y .- μ) ./ μ
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        u = cdf(Exponential(μ[t, s]), max(float(Y[t, s]), 1e-300))
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::ExponentialFit)
    p, K = size(fit.Λ)
    println(io, "Exponential GLLVM fit")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

function Base.show(io::IO, ::MIME"text/plain", fit::GammaFit)
    p, K = size(fit.Λ)
    println(io, "Gamma GLLVM fit")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)), ", shape α = ", round(fit.α; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Delta-lognormal post-fit methods (two-part: occurrence Bernoulli × positive
# lognormal; shared latent z drives the positive part, Λ_z = 0).
# ---------------------------------------------------------------------------

_loadings(fit::DeltaLogNormalFit) = fit.Λc
_loglik(fit::DeltaLogNormalFit)   = fit.loglik

function _nparams(fit::DeltaLogNormalFit)
    p, K = size(fit.Λc)
    ndisp = fit.disp_group === :shared ? 1 : p
    return 2p + (p * K - div(K * (K - 1), 2)) + ndisp   # βz + βc + Λc + σ
end

"""
    getLV(fit::DeltaLogNormalFit, Y; rotate=true) -> n×K matrix

Conditional latent scores for a Delta-lognormal fit: the per-site two-part Laplace
mode `ẑₛ` (occurrence intercept-only, so only the positive part loads on `z`).
"""
function getLV(fit::DeltaLogNormalFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y)
    K = size(fit.Λc, 2)
    fam = DeltaLogNormal(fit.σ)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(fam, view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    predict(fit::DeltaLogNormalFit, Y; type=:response) -> p×n matrix

In-sample predictions at the Laplace mode. `type=:link` is the positive-part linear
predictor `η^c = β^c + Λ_c ẑ`; `:occurrence` the presence probability `π = logistic(β^z)`;
`:positive` the conditional positive mean `exp(η^c + σ²/2)`; `:response` the
unconditional mean `π · exp(η^c + σ²/2)`.
"""
function predict(fit::DeltaLogNormalFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :occurrence, :positive, :link) ||
        throw(ArgumentError("type must be :response, :occurrence, :positive, or :link; got :$type"))
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'                       # p×n
    type === :link && return ηc
    π = inv.(1 .+ exp.(-fit.βz))                     # length p
    type === :occurrence && return repeat(π, 1, n)
    posmean = exp.(ηc .+ fit.σ^2 / 2)
    type === :positive && return posmean
    return π .* posmean
end

"""
    residuals(fit::DeltaLogNormalFit, Y; rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals for the two-part fit: `Φ⁻¹(u)` with
`u = (1−π) + π·G(y)` for `y>0` (`G` the lognormal CDF) and `u` uniform on `[0, 1−π]`
for `y=0` — ≈ N(0,1) under a correct model (pass a fixed `rng` to reproduce).
"""
function residuals(fit::DeltaLogNormalFit, Y::AbstractMatrix{<:Real};
                   rng::AbstractRNG = Random.default_rng())
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    π = inv.(1 .+ exp.(-fit.βz))
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        πt = π[t]
        if Y[t, s] > 0
            u = (1 - πt) + πt * cdf(LogNormal(ηc[t, s], fit.σ), Y[t, s])
        else
            u = (1 - πt) * rand(rng)
        end
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::DeltaLogNormalFit)
    p, K = size(fit.Λc)
    println(io, "Delta-lognormal GLLVM fit (two-part)")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", log-SD σ = ", round(fit.σ; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Hurdle-Poisson post-fit (occurrence Bernoulli × zero-truncated Poisson count).
# ---------------------------------------------------------------------------

_loadings(fit::HurdlePoissonFit) = fit.Λc
_loglik(fit::HurdlePoissonFit)   = fit.loglik

function _nparams(fit::HurdlePoissonFit)
    p, K = size(fit.Λc)
    return 2p + (p * K - div(K * (K - 1), 2))   # βz + βc + Λc (no dispersion)
end

function getLV(fit::HurdlePoissonFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y)
    K = size(fit.Λc, 2)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(HurdlePoisson(), view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    predict(fit::HurdlePoissonFit, Y; type=:response) -> p×n matrix

`:link` = count log-mean predictor `η^c`; `:occurrence` = `π = logistic(β^z)`;
`:positive` = the zero-truncated count mean `μ/(1−e^{−μ})`; `:response` =
unconditional mean `π · μ/(1−e^{−μ})`.
"""
function predict(fit::HurdlePoissonFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :occurrence, :positive, :link) ||
        throw(ArgumentError("type must be :response, :occurrence, :positive, or :link; got :$type"))
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    type === :link && return ηc
    π = inv.(1 .+ exp.(-fit.βz))
    type === :occurrence && return repeat(π, 1, n)
    μ = exp.(ηc)
    μtr = μ ./ (1 .- exp.(-μ))
    type === :positive && return μtr
    return π .* μtr
end

"""
    residuals(fit::HurdlePoissonFit, Y; rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals for the discrete two-part fit: `Φ⁻¹(u)`
with `u` uniform on `[F(y−1), F(y)]` under the hurdle CDF
`F(k) = (1−π) + π·F_trunc(k)` (`F_trunc` the zero-truncated Poisson CDF).
"""
function residuals(fit::HurdlePoissonFit, Y::AbstractMatrix{<:Real};
                   rng::AbstractRNG = Random.default_rng())
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    π = inv.(1 .+ exp.(-fit.βz))
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        πt = π[t]; y = Int(Y[t, s])
        if y == 0
            lo = 0.0; hi = 1 - πt
        else
            μ = exp(ηc[t, s]); p0 = exp(-μ)
            Flo = y == 1 ? 0.0 : (cdf(Poisson(μ), y - 1) - p0) / (1 - p0)
            Fhi = (cdf(Poisson(μ), y) - p0) / (1 - p0)
            lo = (1 - πt) + πt * Flo
            hi = (1 - πt) + πt * Fhi
        end
        u = lo + (hi - lo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::HurdlePoissonFit)
    p, K = size(fit.Λc)
    println(io, "Hurdle-Poisson GLLVM fit (two-part)")
    println(io, "  responses p = ", p, ", latent factors K = ", K)
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Hurdle-NB post-fit (occurrence Bernoulli × zero-truncated NB2 count).
# ---------------------------------------------------------------------------

_loadings(fit::HurdleNBFit) = fit.Λc
_loglik(fit::HurdleNBFit)   = fit.loglik

function _nparams(fit::HurdleNBFit)
    p, K = size(fit.Λc)
    return 2p + (p * K - div(K * (K - 1), 2)) + 1   # βz + βc + Λc + r
end

function getLV(fit::HurdleNBFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(HurdleNB(fit.r), view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    predict(fit::HurdleNBFit, Y; type=:response) -> p×n matrix

`:link` = `η^c`; `:occurrence` = `π`; `:positive` = zero-truncated NB mean
`μ/(1−p₀)` (`p₀=(r/(r+μ))^r`); `:response` = `π · μ/(1−p₀)`.
"""
function predict(fit::HurdleNBFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :occurrence, :positive, :link) ||
        throw(ArgumentError("type must be :response, :occurrence, :positive, or :link; got :$type"))
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    type === :link && return ηc
    π = inv.(1 .+ exp.(-fit.βz))
    type === :occurrence && return repeat(π, 1, n)
    μ = exp.(ηc); r = fit.r
    μtr = μ ./ (1 .- (r ./ (r .+ μ)) .^ r)
    type === :positive && return μtr
    return π .* μtr
end

"""
    residuals(fit::HurdleNBFit, Y; rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals for the discrete two-part fit, using the
hurdle CDF `F(k) = (1−π) + π·F_trunc(k)` (`F_trunc` the zero-truncated NB2 CDF).
"""
function residuals(fit::HurdleNBFit, Y::AbstractMatrix{<:Real};
                   rng::AbstractRNG = Random.default_rng())
    p, n = size(Y); r = fit.r
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    π = inv.(1 .+ exp.(-fit.βz))
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        πt = π[t]; y = Int(Y[t, s])
        if y == 0
            lo = 0.0; hi = 1 - πt
        else
            μ = exp(ηc[t, s]); p0 = (r / (r + μ))^r
            nb = NegativeBinomial(r, r / (r + μ))
            Flo = y == 1 ? 0.0 : (cdf(nb, y - 1) - p0) / (1 - p0)
            Fhi = (cdf(nb, y) - p0) / (1 - p0)
            lo = (1 - πt) + πt * Flo
            hi = (1 - πt) + πt * Fhi
        end
        u = lo + (hi - lo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::HurdleNBFit)
    p, K = size(fit.Λc)
    println(io, "Hurdle-NB GLLVM fit (two-part)")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", dispersion r = ", round(fit.r; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Delta-Gamma post-fit (occurrence Bernoulli × positive Gamma, log-link mean).
# ---------------------------------------------------------------------------

_loadings(fit::DeltaGammaFit) = fit.Λc
_loglik(fit::DeltaGammaFit)   = fit.loglik

function _nparams(fit::DeltaGammaFit)
    p, K = size(fit.Λc)
    ndisp = fit.disp_group === :shared ? 1 : p
    return 2p + (p * K - div(K * (K - 1), 2)) + ndisp   # βz + βc + Λc + α
end

"""
    getLV(fit::DeltaGammaFit, Y; rotate=true) -> n×K matrix

Conditional latent scores for a Delta-Gamma fit: the per-site two-part Laplace mode
`ẑₛ` (occurrence intercept-only, so only the positive part loads on `z`).
"""
function getLV(fit::DeltaGammaFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    fam = DeltaGamma(fit.α)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(fam, view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    predict(fit::DeltaGammaFit, Y; type=:response) -> p×n matrix

`:link` = positive-part log-mean predictor `η^c`; `:occurrence` = presence
probability `π = logistic(β^z)`; `:positive` = conditional positive mean `μ = exp(η^c)`
(the Gamma mean); `:response` = unconditional mean `π · μ`.
"""
function predict(fit::DeltaGammaFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :occurrence, :positive, :link) ||
        throw(ArgumentError("type must be :response, :occurrence, :positive, or :link; got :$type"))
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'                       # p×n
    type === :link && return ηc
    π = inv.(1 .+ exp.(-fit.βz))                     # length p
    type === :occurrence && return repeat(π, 1, n)
    μ = exp.(ηc)
    type === :positive && return μ
    return π .* μ
end

"""
    residuals(fit::DeltaGammaFit, Y; rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals for the two-part fit: `Φ⁻¹(u)` with
`u = (1−π) + π·G(y)` for `y>0` (`G` the Gamma CDF) and `u` uniform on `[0, 1−π]`
for `y=0` — ≈ N(0,1) under a correct model (pass a fixed `rng` to reproduce).
"""
function residuals(fit::DeltaGammaFit, Y::AbstractMatrix{<:Real};
                   rng::AbstractRNG = Random.default_rng())
    p, n = size(Y); α = fit.α
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    π = inv.(1 .+ exp.(-fit.βz))
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        πt = π[t]
        if Y[t, s] > 0
            μ = exp(ηc[t, s])
            u = (1 - πt) + πt * cdf(Gamma(α, μ / α), Y[t, s])
        else
            u = (1 - πt) * rand(rng)
        end
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::DeltaGammaFit)
    p, K = size(fit.Λc)
    println(io, "Delta-Gamma GLLVM fit (two-part)")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", shape α = ", round(fit.α; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Zero-inflated post-fit (ZIP / ZINB: structural zero × Poisson / NB2 count).
# Unconditional mean is (1−π)·μ (structural zeros contribute 0).
# ---------------------------------------------------------------------------

_loadings(fit::ZIPFit) = fit.Λc
_loglik(fit::ZIPFit)   = fit.loglik
_loadings(fit::ZINBFit) = fit.Λc
_loglik(fit::ZINBFit)   = fit.loglik
_loadings(fit::ZIPCovFit) = fit.Λc
_loglik(fit::ZIPCovFit)   = fit.loglik
_loadings(fit::ZINBCovFit) = fit.Λc
_loglik(fit::ZINBCovFit)   = fit.loglik

function _nparams(fit::ZIPFit)
    p, K = size(fit.Λc)
    return 2p + (p * K - div(K * (K - 1), 2))        # βz + βc + Λc
end
function _nparams(fit::ZINBFit)
    p, K = size(fit.Λc)
    return 2p + (p * K - div(K * (K - 1), 2)) + 1     # βz + βc + Λc + r
end
function _nparams(fit::ZIPCovFit)
    p, K = size(fit.Λc)
    q_free = count(!, fit.γ_fixed)
    return 2p + 2q_free + (p * K - div(K * (K - 1), 2))  # βz + γz + βc + γc + Λc
end
function _nparams(fit::ZINBCovFit)
    p, K = size(fit.Λc)
    q_free = count(!, fit.γ_fixed)
    return 2p + 2q_free + (p * K - div(K * (K - 1), 2)) + 1  # + shared log r
end

function getLV(fit::ZIPFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(ZIPoisson(), view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    getLV(fit::ZIPCovFit, Y, X; rotate=true) -> n×K matrix

Conditional latent modes under dual offsets `Oz = Xγz`, `Oc = Xγc` (`Λ_z = 0`).
"""
function getLV(fit::ZIPCovFit, Y::AbstractMatrix{<:Real}, X::AbstractArray{<:Real, 3};
               rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q); got $(size(X))"))
    Oz = _build_offset(X, fit.γz)
    Oc = _build_offset(X, fit.γc)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(ZIPoisson(), view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc;
                                offsetz = view(Oz, :, s), offsetc = view(Oc, :, s))
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

function getLV(fit::ZINBFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(ZINB(fit.r), view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    getLV(fit::ZINBCovFit, Y, X; rotate=true) -> n×K matrix

Conditional latent modes under dual offsets `Oz = Xγz`, `Oc = Xγc` (`Λ_z = 0`)
and the shared scalar `r`.
"""
function getLV(fit::ZINBCovFit, Y::AbstractMatrix{<:Real}, X::AbstractArray{<:Real, 3};
               rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q); got $(size(X))"))
    Oz = _build_offset(X, fit.γz)
    Oc = _build_offset(X, fit.γc)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(ZINB(fit.r), view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc;
                                offsetz = view(Oz, :, s), offsetc = view(Oc, :, s))
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    predict(fit::ZIPFit, Y; type=:response) -> p×n matrix

`:link` = count log-mean predictor `η^c`; `:zeroinfl` = structural-zero
probability `π = logistic(β^z)`; `:mean` = the count mean `μ = exp(η^c)`;
`:response` = unconditional mean `(1−π)·μ`.
"""
function predict(fit::ZIPFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :zeroinfl, :mean, :link) ||
        throw(ArgumentError("type must be :response, :zeroinfl, :mean, or :link; got :$type"))
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    type === :link && return ηc
    π = inv.(1 .+ exp.(-fit.βz))
    type === :zeroinfl && return repeat(π, 1, n)
    μ = exp.(ηc)
    type === :mean && return μ
    return (1 .- π) .* μ
end

"""
    predict(fit::ZINBFit, Y; type=:response) -> p×n matrix

As [`predict(::ZIPFit, …)`](@ref); `:mean` is the NB2 count mean `μ`, `:response`
the unconditional mean `(1−π)·μ`.
"""
function predict(fit::ZINBFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :zeroinfl, :mean, :link) ||
        throw(ArgumentError("type must be :response, :zeroinfl, :mean, or :link; got :$type"))
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    type === :link && return ηc
    π = inv.(1 .+ exp.(-fit.βz))
    type === :zeroinfl && return repeat(π, 1, n)
    μ = exp.(ηc)
    type === :mean && return μ
    return (1 .- π) .* μ
end

# Dunn–Smyth residuals for the zero-inflated CDF F(k) = π + (1−π)·F_count(k).
function _zi_residuals(π::AbstractVector, ηc::AbstractMatrix, Y::AbstractMatrix,
                       countdist, rng::AbstractRNG)
    p, n = size(Y)
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        πt = π[t]; y = Int(Y[t, s])
        d = countdist(exp(ηc[t, s]))
        if y == 0
            lo = 0.0
            hi = πt + (1 - πt) * cdf(d, 0)
        else
            lo = πt + (1 - πt) * cdf(d, y - 1)
            hi = πt + (1 - πt) * cdf(d, y)
        end
        u = lo + (hi - lo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

"""
    residuals(fit::ZIPFit, Y; rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals under the zero-inflated CDF
`F(k) = π + (1−π)·F_Poisson(k)`.
"""
function residuals(fit::ZIPFit, Y::AbstractMatrix{<:Real};
                   rng::AbstractRNG = Random.default_rng())
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    π = inv.(1 .+ exp.(-fit.βz))
    return _zi_residuals(π, ηc, Y, μ -> Poisson(μ), rng)
end

"""
    residuals(fit::ZINBFit, Y; rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals under `F(k) = π + (1−π)·F_NB2(k)`.
"""
function residuals(fit::ZINBFit, Y::AbstractMatrix{<:Real};
                   rng::AbstractRNG = Random.default_rng())
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    π = inv.(1 .+ exp.(-fit.βz)); r = fit.r
    return _zi_residuals(π, ηc, Y, μ -> NegativeBinomial(r, r / (r + μ)), rng)
end

function Base.show(io::IO, ::MIME"text/plain", fit::ZIPFit)
    p, K = size(fit.Λc)
    println(io, "Zero-inflated Poisson GLLVM fit (two-part)")
    println(io, "  responses p = ", p, ", latent factors K = ", K)
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

function Base.show(io::IO, ::MIME"text/plain", fit::ZINBFit)
    p, K = size(fit.Λc)
    println(io, "Zero-inflated NB GLLVM fit (two-part)")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", dispersion r = ", round(fit.r; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Zero-inflated binomial post-fit (ZIB: structural zero × Binomial(N, μ) count,
# μ = logistic(η^c), N trials fixed — no dispersion). Mirrors ZINB, swapping the
# NB2 count for Binomial(N, μ). Unconditional mean is (1−π)·N·μ.
# ---------------------------------------------------------------------------

_loadings(fit::ZIBFit) = fit.Λc
_loglik(fit::ZIBFit)   = fit.loglik
_loadings(fit::ZIBCovFit) = fit.Λc
_loglik(fit::ZIBCovFit)   = fit.loglik

function _nparams(fit::ZIBFit)
    p, K = size(fit.Λc)
    return 2p + (p * K - div(K * (K - 1), 2))        # βz + βc + Λc (N fixed, no dispersion)
end
function _nparams(fit::ZIBCovFit)
    p, K = size(fit.Λc)
    q_free = count(!, fit.γ_fixed)
    return 2p + 2q_free + (p * K - div(K * (K - 1), 2))  # βz + γz + βc + γc + Λc (N fixed)
end

function getLV(fit::ZIBFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(ZIB(fit.N), view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    getLV(fit::ZIBCovFit, Y, X; rotate=true) -> n×K matrix

Conditional latent modes under dual offsets `Oz = Xγz`, `Oc = Xγc` (`Λ_z = 0`)
with shared scalar trials `N`.
"""
function getLV(fit::ZIBCovFit, Y::AbstractMatrix{<:Real}, X::AbstractArray{<:Real, 3};
               rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q); got $(size(X))"))
    Oz = _build_offset(X, fit.γz)
    Oc = _build_offset(X, fit.γc)
    Λz = zeros(p, K)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(ZIB(fit.N), view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc;
                                offsetz = view(Oz, :, s), offsetc = view(Oc, :, s))
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    predict(fit::ZIBFit, Y; type=:response) -> p×n matrix

`:link` = count success-logit predictor `η^c`; `:zeroinfl` = structural-zero
probability `π = logistic(β^z)`; `:mean` = the binomial mean `N·μ`
(`μ = logistic(η^c)`); `:response` = unconditional mean `(1−π)·N·μ`.
"""
function predict(fit::ZIBFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :zeroinfl, :mean, :link) ||
        throw(ArgumentError("type must be :response, :zeroinfl, :mean, or :link; got :$type"))
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    type === :link && return ηc
    π = inv.(1 .+ exp.(-fit.βz))
    type === :zeroinfl && return repeat(π, 1, n)
    μ = inv.(1 .+ exp.(-ηc))                          # logit link for the count part
    type === :mean && return fit.N .* μ
    return (1 .- π) .* (fit.N .* μ)
end

"""
    residuals(fit::ZIBFit, Y; rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals under the zero-inflated CDF
`F(k) = π + (1−π)·F_Binomial(N,μ)(k)`.
"""
function residuals(fit::ZIBFit, Y::AbstractMatrix{<:Real};
                   rng::AbstractRNG = Random.default_rng())
    p, n = size(Y)
    Z = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    π = inv.(1 .+ exp.(-fit.βz))
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        πt = π[t]; y = Int(Y[t, s])
        μ = inv(1 + exp(-ηc[t, s]))
        d = Binomial(fit.N, μ)
        if y == 0
            lo = 0.0
            hi = πt + (1 - πt) * cdf(d, 0)
        else
            lo = πt + (1 - πt) * cdf(d, y - 1)
            hi = πt + (1 - πt) * cdf(d, y)
        end
        u = lo + (hi - lo) * rand(rng)
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::ZIBFit)
    p, K = size(fit.Λc)
    println(io, "Zero-inflated binomial GLLVM fit (two-part)")
    println(io, "  responses p = ", p, ", latent factors K = ", K, ", trials N = ", fit.N)
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Tweedie post-fit (compound Poisson–Gamma, power 1 < p < 2; mean μ = exp(η),
# dispersion φ, Var = φ μ^p; point mass at 0 plus a positive continuous part).
# Scalar-μ family, mirroring Gamma; the Tweedie CDF is mixed (atom at 0 + density
# for y>0), so the Dunn–Smyth residual randomises the jump at 0 and is the
# deterministic PIT on the positive part.
# ---------------------------------------------------------------------------

_loadings(fit::TweedieFit) = fit.Λ
_loglik(fit::TweedieFit)   = fit.loglik

function _nparams(fit::TweedieFit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + 2       # β + Λ + dispersion φ + power p
end

"""
    getLV(fit::TweedieFit, Y; rotate=true) -> n×K matrix

Conditional latent-variable scores for a Tweedie fit: the per-site Laplace mode
`ẑₛ` (computed at the fitted dispersion `φ` and power `p`). `Y` is the p×n matrix
of non-negative reals; `rotate=true` applies the canonical [`rotation`](@ref).
"""
function getLV(fit::TweedieFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    fam = TweedieED(fit.φ, fit.p)
    ones_p = ones(Int, p)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _laplace_mode(fam, view(Y, :, s), ones_p, fit.Λ, fit.β, fit.link)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::TweedieFit, Y; type=:response) -> p×n matrix

In-sample fitted values at the Laplace mode: `type=:link` returns `η = β + Λ ẑ`;
`type=:response` the inverse-link fitted means `linkinv(link, η) = exp(η)`
(non-negative reals).
"""
function predict(fit::TweedieFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; rotate = false)
    η = fit.β .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

"""
    residuals(fit::TweedieFit, Y; type=:dunnsmyth, rng=Random.default_rng()) -> p×n matrix

Conditional residuals for a Tweedie fit. The Tweedie CDF has an atom at `0` plus a
continuous positive part, so the `:dunnsmyth` randomized quantile residual draws
`u` uniform on `[0, F(0)]` at `y=0` and is the deterministic PIT `Φ⁻¹(F(y))` for
`y>0` — ≈ N(0,1) under a correct model (pass a fixed `rng` to reproduce). `:pearson`
returns `(Y − μ) / √(φ μ^p)`.
"""
function residuals(fit::TweedieFit, Y::AbstractMatrix{<:Real};
                   type::Symbol = :dunnsmyth, rng::AbstractRNG = Random.default_rng())
    type in (:dunnsmyth, :pearson) ||
        throw(ArgumentError("type must be :dunnsmyth or :pearson; got :$type"))
    p, n = size(Y)
    φ = fit.φ; pw = fit.p
    μ = predict(fit, Y; type = :response)
    if type === :pearson
        return (Y .- μ) ./ sqrt.(φ .* μ .^ pw)
    end
    R = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        if Y[t, s] <= 0
            F0 = exp(tweedie_logpdf(0.0, μ[t, s], φ, pw))   # P(Y = 0) (the atom)
            u = F0 * rand(rng)
        else
            u = tweedie_cdf(float(Y[t, s]), μ[t, s], φ, pw) # atom + positive-part CDF
        end
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::TweedieFit)
    p, K = size(fit.Λ)
    println(io, "Tweedie GLLVM fit")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", link = ", nameof(typeof(fit.link)),
            ", φ = ", round(fit.φ; sigdigits = 4), ", power = ", round(fit.p; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Covariate-fit post-fit (GllvmCovFit: η = β + Xγ + Λẑ). Needs the (p,n,q) design
# `X` (and Binomial trial counts `N`) to rebuild the linear predictor.
# ---------------------------------------------------------------------------

_loadings(fit::GllvmCovFit) = fit.Λ
_loglik(fit::GllvmCovFit)   = fit.loglik

function _nparams(fit::GllvmCovFit)
    p, K = size(fit.Λ); q = count(!, fit.γ_fixed)
    return p + q + (p * K - div(K * (K - 1), 2)) + (isnan(fit.dispersion) ? 0 : 1)
end

"""
    getLV(fit::GllvmCovFit, Y, X; rotate=true, N=nothing) -> n×K matrix

Conditional latent scores for a covariate fit: the per-site offset-aware Laplace
mode `ẑₛ` at `η = β + Xγ + Λz`.
"""
function getLV(fit::GllvmCovFit, Y::AbstractMatrix{<:Real}, X::AbstractArray{<:Real, 3};
               rotate::Bool = true, N::Union{Nothing, AbstractMatrix} = nothing)
    p, n = size(Y); K = size(fit.Λ, 2)
    Nm = N === nothing ? fill(1, p, n) : N
    fam = _cov_family(fit.family, fit.dispersion)
    O = _build_offset(X, fit.γ)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        η0 = fit.β .+ view(O, :, s)
        Z[:, s] = _laplace_mode_off(fam, view(Y, :, s), view(Nm, :, s), fit.Λ, η0, fit.link)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::GllvmCovFit, Y, X; type=:response, N=nothing) -> p×n matrix

`:link` = the linear predictor `η = β + Xγ + Λẑ`; `:response` (= `:mean`) = the
mean `μ = linkinv(link, η)` (a probability for Binomial, a positive mean for the
count/positive families).
"""
function predict(fit::GllvmCovFit, Y::AbstractMatrix{<:Real}, X::AbstractArray{<:Real, 3};
                 type::Symbol = :response, N::Union{Nothing, AbstractMatrix} = nothing)
    type in (:response, :mean, :link) ||
        throw(ArgumentError("type must be :response, :mean, or :link; got :$type"))
    Z = getLV(fit, Y, X; rotate = false, N = N)
    O = _build_offset(X, fit.γ)
    η = fit.β .+ O .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

"""
    fitted(fit::GllvmCovFit, Y, X; N=nothing) -> p×n matrix of fitted means.
"""
fitted(fit::GllvmCovFit, Y::AbstractMatrix{<:Real}, X::AbstractArray{<:Real, 3};
       N::Union{Nothing, AbstractMatrix} = nothing) =
    predict(fit, Y, X; type = :response, N = N)

"""
    predict(fit::GllvmCovFit, X; type=:response) -> p×n matrix

Population-level (new-site) prediction at a covariate design `X` (`(p, n, q)`) with
the latent at its prior mean `z = 0` — the latent is not estimable at unseen sites.
`:link` returns the fixed-effect linear predictor `η = β + Xγ`; `:response`
(= `:mean`) the mean `μ = linkinv(link, η)`. (For in-sample *conditional*
predictions at the fitted sites, use the three-argument `predict(fit, Y, X)`.)
"""
function predict(fit::GllvmCovFit, X::AbstractArray{<:Real, 3}; type::Symbol = :response)
    type in (:response, :mean, :link) ||
        throw(ArgumentError("type must be :response, :mean, or :link; got :$type"))
    O = _build_offset(X, fit.γ)
    η = fit.β .+ O
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

function Base.show(io::IO, ::MIME"text/plain", fit::GllvmCovFit)
    p, K = size(fit.Λ); q = length(fit.γ)
    println(io, "GLLVM fit with covariates (", nameof(typeof(fit.family)), ", Laplace)")
    println(io, "  responses p = ", p, ", covariates q = ", q, ", latent factors K = ", K,
            isnan(fit.dispersion) ? "" : ", dispersion = $(round(fit.dispersion; sigdigits = 4))")
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end
