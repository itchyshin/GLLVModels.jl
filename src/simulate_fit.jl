# simulate(fit, …) — draw a fresh response matrix from a fitted non-Gaussian GLLVM.
# Reuses the per-family samplers (`_cov_sample`, `families/covariates.jl`): draws a
# new per-site latent `z_s ~ N(0, I_K)`, forms `η`, and samples each response. The
# shared `laplace.jl` core is untouched.

using Random: AbstractRNG, default_rng

# (family marker carrying the fitted dispersion, link) for each scalar-μ fit.
_sim_family(fit::PoissonFit)     = (Poisson(), fit.link)
_sim_family(fit::NBFit)          = (NegativeBinomial(fit.r, 0.5), fit.link)
_sim_family(fit::BetaFit)        = (Beta(fit.φ, 1.0), fit.link)
_sim_family(fit::GammaFit)       = (Gamma(fit.α, 1.0), fit.link)
_sim_family(fit::ExponentialFit) = (Exponential(1.0), fit.link)

const _ScalarMuFit = Union{PoissonFit, NBFit, BetaFit, GammaFit, ExponentialFit}

# Element type of a simulated matrix: continuous families → Float64, else Int.
_sim_eltype(fam) = fam isa Union{Beta, Gamma, Exponential} ? Float64 : Int

"""
    simulate(fit, n; rng=Random.default_rng()) -> p×n matrix

Simulate a fresh response matrix (`p` species × `n` sites) from a fitted
non-Gaussian GLLVM (`PoissonFit`, `NBFit`, `BetaFit`, `GammaFit`,
`ExponentialFit`). A new latent `z_s ~ N(0, I_K)` is drawn per site, `η = β + Λ z`,
and each response sampled from the fitted family at `μ = linkinv(link, η)`. Pass a
fixed `rng` to reproduce. (For `BinomialFit` pass `N`; for a covariate
`GllvmCovFit` use the `X` method below.)
"""
function simulate(fit::_ScalarMuFit, n::Integer; rng::AbstractRNG = default_rng())
    fam, link = _sim_family(fit)
    p, K = size(fit.Λ)
    Y = Matrix{_sim_eltype(fam)}(undef, p, n)
    @inbounds for s in 1:n
        η = fit.β .+ fit.Λ * randn(rng, K)
        for t in 1:p
            μ = linkinv(link, _clamp_eta(η[t]))
            Y[t, s] = _cov_sample(fam, μ, 1, rng)
        end
    end
    return Y
end

"""
    simulate(fit::BinomialFit, n; N=nothing, X_lv=nothing,
             rng=Random.default_rng(), offset=nothing) -> p×n matrix

Binomial simulation with trial counts `N` (default all-ones / Bernoulli). For
fits that used `X_lv`, pass the matching n×q_lv predictor matrix; the simulator
draws `z_total = X_lv * alpha_lv + z` with `z ~ N(0, I_K)`.

`offset` is a known additive term in `η` (p×n). Only AGHQ fits accept it; on a
Laplace fit an explicit `offset` throws an `ArgumentError`. On an AGHQ fit with a
non-zero stored offset, simulating a different number of sites than the fit needs an
explicit `offset` (`ArgumentError`), and a non-finite or wrongly shaped `offset` is
refused.
"""
function simulate(fit::BinomialFit, n::Integer;
                  N::Union{Nothing, AbstractMatrix} = nothing,
                  X_lv::Union{Nothing, AbstractMatrix} = nothing,
                  rng::AbstractRNG = default_rng(),offset=nothing)
    if _is_binomial_aghq(fit)
        X_lv===nothing || throw(ArgumentError("AGHQ loadings-only simulation does not use X_lv"))
        return _binomial_aghq_simulate(fit,n;N=N,rng=rng,offset=offset)
    end
    offset===nothing || throw(ArgumentError("explicit offset currently requires AGHQ metadata"))
    p, K = size(fit.Λ)
    Nm = N === nothing ? fill(1, p, n) : N
    Zmean = if fit.alpha_lv === nothing
        zeros(Float64, n, K)
    else
        X_lv === nothing && throw(ArgumentError(
            "this BinomialFit used X_lv; pass the matching X_lv to simulate"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n ($n)"))
        size(X_lv, 2) == size(fit.alpha_lv, 1) ||
            throw(ArgumentError(
                "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.alpha_lv, 1)))"))
        _lv_score_mean(X_lv, fit.alpha_lv)
    end
    Y = Matrix{Int}(undef, p, n)
    @inbounds for s in 1:n
        z_total = Zmean[s, :] .+ randn(rng, K)
        η = fit.β .+ fit.Λ * z_total
        for t in 1:p
            μ = clamp(linkinv(fit.link, _clamp_eta(η[t])), 1e-12, 1 - 1e-12)
            Y[t, s] = rand(rng, Binomial(Nm[t, s], μ))
        end
    end
    return Y
end

"""
    simulate(fit::PoissonFit, n; X_lv=nothing, rng=Random.default_rng(),
             offset=nothing) -> p×n matrix

Poisson simulation. For fits that used `X_lv`, pass the matching n×q_lv predictor
matrix; the simulator draws `z_total = X_lv * alpha_lv + z` with `z ~ N(0, I_K)`.
Without `X_lv` this reduces to the ordinary scalar-μ Poisson draw (identical RNG
stream).

`offset` is a known additive term in `η` (p×n). Only AGHQ fits accept it; on a
Laplace fit an explicit `offset` throws an `ArgumentError`. On an AGHQ fit with a
non-zero stored offset, simulating a different number of sites than the fit needs an
explicit `offset` (`ArgumentError`), and a non-finite or wrongly shaped `offset` is
refused.
"""
function simulate(fit::PoissonFit, n::Integer;
                  X_lv::Union{Nothing, AbstractMatrix} = nothing,
                  rng::AbstractRNG = default_rng(),offset=nothing)
    if _is_poisson_aghq(fit)
        X_lv===nothing || throw(ArgumentError("AGHQ loadings-only simulation does not use X_lv"))
        return _poisson_aghq_simulate(fit,n;rng=rng,offset=offset)
    end
    offset===nothing || throw(ArgumentError("explicit simulation offset currently requires AGHQ metadata"))
    fam, link = _sim_family(fit)
    p, K = size(fit.Λ)
    Zmean = if fit.alpha_lv === nothing
        zeros(Float64, n, K)
    else
        X_lv === nothing && throw(ArgumentError(
            "this PoissonFit used X_lv; pass the matching X_lv to simulate"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n ($n)"))
        size(X_lv, 2) == size(fit.alpha_lv, 1) ||
            throw(ArgumentError(
                "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.alpha_lv, 1)))"))
        _lv_score_mean(X_lv, fit.alpha_lv)
    end
    Y = Matrix{_sim_eltype(fam)}(undef, p, n)
    @inbounds for s in 1:n
        z_total = Zmean[s, :] .+ randn(rng, K)
        η = fit.β .+ fit.Λ * z_total
        for t in 1:p
            μ = linkinv(link, _clamp_eta(η[t]))
            Y[t, s] = _cov_sample(fam, μ, 1, rng)
        end
    end
    return Y
end

"""
    simulate(fit::NBFit, n; X_lv=nothing, rng=Random.default_rng()) -> p×n matrix

Negative-binomial (NB2) simulation. For fits that used `X_lv`, pass the matching
n×q_lv predictor matrix; `z_total = X_lv * alpha_lv + z`. Without `X_lv` this
reduces to the ordinary scalar-μ NB2 draw (identical RNG stream).
"""
function simulate(fit::NBFit, n::Integer;
                  X_lv::Union{Nothing, AbstractMatrix} = nothing,
                  rng::AbstractRNG = default_rng())
    fam, link = _sim_family(fit)
    p, K = size(fit.Λ)
    Zmean = if fit.alpha_lv === nothing
        zeros(Float64, n, K)
    else
        X_lv === nothing && throw(ArgumentError(
            "this NBFit used X_lv; pass the matching X_lv to simulate"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n ($n)"))
        size(X_lv, 2) == size(fit.alpha_lv, 1) ||
            throw(ArgumentError(
                "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.alpha_lv, 1)))"))
        _lv_score_mean(X_lv, fit.alpha_lv)
    end
    Y = Matrix{_sim_eltype(fam)}(undef, p, n)
    @inbounds for s in 1:n
        z_total = Zmean[s, :] .+ randn(rng, K)
        η = fit.β .+ fit.Λ * z_total
        for t in 1:p
            μ = linkinv(link, _clamp_eta(η[t]))
            Y[t, s] = _cov_sample(fam, μ, 1, rng)
        end
    end
    return Y
end

"""
    simulate(fit::GammaFit, n; X_lv=nothing, rng=Random.default_rng()) -> p×n matrix

Gamma simulation (positive continuous). For fits that used `X_lv`, pass the
matching n×q_lv predictor matrix; `z_total = X_lv * alpha_lv + z`. Without `X_lv`
this reduces to the ordinary scalar-μ Gamma draw (identical RNG stream).
"""
function simulate(fit::GammaFit, n::Integer;
                  X_lv::Union{Nothing, AbstractMatrix} = nothing,
                  rng::AbstractRNG = default_rng())
    fam, link = _sim_family(fit)
    p, K = size(fit.Λ)
    Zmean = if fit.alpha_lv === nothing
        zeros(Float64, n, K)
    else
        X_lv === nothing && throw(ArgumentError(
            "this GammaFit used X_lv; pass the matching X_lv to simulate"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n ($n)"))
        size(X_lv, 2) == size(fit.alpha_lv, 1) ||
            throw(ArgumentError(
                "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.alpha_lv, 1)))"))
        _lv_score_mean(X_lv, fit.alpha_lv)
    end
    Y = Matrix{_sim_eltype(fam)}(undef, p, n)
    @inbounds for s in 1:n
        z_total = Zmean[s, :] .+ randn(rng, K)
        η = fit.β .+ fit.Λ * z_total
        for t in 1:p
            μ = linkinv(link, _clamp_eta(η[t]))
            Y[t, s] = _cov_sample(fam, μ, 1, rng)
        end
    end
    return Y
end

"""
    simulate(fit::BetaFit, n; X_lv=nothing, rng=Random.default_rng()) -> p×n matrix

Beta simulation (proportions in (0,1)). For fits that used `X_lv`, pass the
matching n×q_lv predictor matrix; `z_total = X_lv * alpha_lv + z`. Without `X_lv`
this reduces to the ordinary scalar-μ Beta draw (identical RNG stream).
"""
function simulate(fit::BetaFit, n::Integer;
                  X_lv::Union{Nothing, AbstractMatrix} = nothing,
                  rng::AbstractRNG = default_rng())
    fam, link = _sim_family(fit)
    p, K = size(fit.Λ)
    Zmean = if fit.alpha_lv === nothing
        zeros(Float64, n, K)
    else
        X_lv === nothing && throw(ArgumentError(
            "this BetaFit used X_lv; pass the matching X_lv to simulate"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n ($n)"))
        size(X_lv, 2) == size(fit.alpha_lv, 1) ||
            throw(ArgumentError(
                "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.alpha_lv, 1)))"))
        _lv_score_mean(X_lv, fit.alpha_lv)
    end
    Y = Matrix{_sim_eltype(fam)}(undef, p, n)
    @inbounds for s in 1:n
        z_total = Zmean[s, :] .+ randn(rng, K)
        η = fit.β .+ fit.Λ * z_total
        for t in 1:p
            μ = linkinv(link, _clamp_eta(η[t]))
            Y[t, s] = _cov_sample(fam, μ, 1, rng)
        end
    end
    return Y
end

"""
    simulate(fit::OrdinalFit, n; X_lv=nothing, rng=Random.default_rng()) -> p×n matrix

Ordinal simulation from a shared-cutpoint cumulative-link fit. For fits that
used `X_lv`, pass the matching n×q_lv predictor matrix; the simulator draws
`z_total = X_lv * alpha_lv + z` with `z ~ N(0, I_K)`.
"""
function simulate(fit::OrdinalFit, n::Integer;
                  X_lv::Union{Nothing, AbstractMatrix} = nothing,
                  rng::AbstractRNG = default_rng())
    p, K = size(fit.Λ)
    Zmean = if fit.alpha_lv === nothing
        zeros(Float64, n, K)
    else
        X_lv === nothing && throw(ArgumentError(
            "this OrdinalFit used X_lv; pass the matching X_lv to simulate"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n ($n)"))
        size(X_lv, 2) == size(fit.alpha_lv, 1) ||
            throw(ArgumentError(
                "X_lv second dim ($(size(X_lv, 2))) must equal fitted alpha_lv rows ($(size(fit.alpha_lv, 1)))"))
        _lv_score_mean(X_lv, fit.alpha_lv)
    end
    Y = Matrix{Int}(undef, p, n)
    @inbounds for s in 1:n
        z_total = Zmean[s, :] .+ randn(rng, K)
        η = fit.Λ * z_total
        for t in 1:p
            probs = [_ord_prob(c, η[t], fit.τ, fit.link) for c in 1:fit.C]
            Y[t, s] = rand(rng, Categorical(probs ./ sum(probs)))
        end
    end
    return Y
end

"""
    simulate(fit::StudentTFit, n; rng=Random.default_rng()) -> p×n matrix

Simulate from a fitted Student-t GLLVM (heavy-tailed continuous, fixed `ν`,
identity link): a new latent `z_s ~ N(0, I_K)` per site, `η = β + Λ z`, location
`μ = η` (identity link), and each response drawn as `μ + σ · t`, `t ~ TDist(ν)` —
the exact sampling inverse of the location–scale t density. Pass a fixed `rng` to
reproduce.
"""
function simulate(fit::StudentTFit, n::Integer; rng::AbstractRNG = default_rng())
    p, K = size(fit.Λ)
    Y = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n
        η = fit.β .+ fit.Λ * randn(rng, K)
        for t in 1:p
            μ = linkinv(fit.link, _clamp_eta(η[t]))
            σ_t = fit.σ isa Real ? fit.σ : fit.σ[t]
            ν_t = fit.ν isa Real ? fit.ν : fit.ν[t]
            Y[t, s] = μ + σ_t * rand(rng, TDist(ν_t))
        end
    end
    return Y
end

# One Tweedie (compound Poisson–Gamma, 1 < p < 2) draw at mean μ, dispersion φ,
# power p: N ~ Poisson(λ), λ = μ^{2−p}/(φ(2−p)); y = 0 if N = 0, else
# y ~ Gamma(N·α, scale = φ(p−1)μ^{p−1}), α = (2−p)/(p−1).
function _tweedie_sample(μ::Real, φ::Real, p::Real, rng::AbstractRNG)
    μ = max(float(μ), 1e-12); φ = float(φ); p = float(p)
    λ = μ^(2.0 - p) / (φ * (2.0 - p))
    N = rand(rng, Poisson(λ))
    N == 0 && return 0.0
    α = (2.0 - p) / (p - 1.0)
    scale = φ * (p - 1.0) * μ^(p - 1.0)
    return rand(rng, Gamma(N * α, scale))
end

"""
    simulate(fit::TweedieFit, n; rng=Random.default_rng()) -> p×n matrix

Simulate a fresh response matrix from a fitted Tweedie GLLVM (compound Poisson–
Gamma, power `1 < p < 2`): a new latent `z_s ~ N(0, I_K)` per site, `η = β + Λ z`,
`μ = linkinv(link, η)`, and each response drawn from the compound Poisson–Gamma at
`(μ, φ, p)` — a point mass at `0` plus a positive continuous part. Pass a fixed
`rng` to reproduce.
"""
function simulate(fit::TweedieFit, n::Integer; rng::AbstractRNG = default_rng())
    p, K = size(fit.Λ)
    Y = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n
        η = fit.β .+ fit.Λ * randn(rng, K)
        for t in 1:p
            μ = linkinv(fit.link, _clamp_eta(η[t]))
            Y[t, s] = _tweedie_sample(μ, fit.φ, fit.p, rng)
        end
    end
    return Y
end

"""
    simulate(fit::GllvmCovFit, X; N=nothing, rng=Random.default_rng()) -> p×n matrix

Simulate from a fitted covariate model at the design `X` (`(p, n, q)`), with the
fixed-effect offset: `η = β + Xγ + Λ z`. `n` is taken from `size(X, 2)`.
"""
function simulate(fit::GllvmCovFit, X::AbstractArray{<:Real, 3};
                  N::Union{Nothing, AbstractMatrix} = nothing, rng::AbstractRNG = default_rng())
    p, K = size(fit.Λ); n = size(X, 2)
    fam = _cov_family(fit.family, fit.dispersion)
    Nm = N === nothing ? fill(1, p, n) : N
    O = _build_offset(X, fit.γ)
    Y = Matrix{_sim_eltype(fit.family)}(undef, p, n)
    @inbounds for s in 1:n
        η = fit.β .+ view(O, :, s) .+ fit.Λ * randn(rng, K)
        for t in 1:p
            μ = linkinv(fit.link, _clamp_eta(η[t]))
            Y[t, s] = _cov_sample(fam, μ, Nm[t, s], rng)
        end
    end
    return Y
end
