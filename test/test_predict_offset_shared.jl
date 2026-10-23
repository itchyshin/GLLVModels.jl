using GLLVModels, Test, Random, Distributions, LinearAlgebra
using SpecialFunctions: loggamma
const GM = GLLVModels

# predict / getLV / residuals on the shared-dispersion fits made with an offset (#788).
#
# The bug this file pins: Gamma, Beta, NB1, GP-1, Exponential, hurdle-NB, ZIP, ZINB, ZIB,
# Delta-lognormal and Delta-Gamma fits accepted `offset` at fit time but did not keep it,
# so `getLV(fit, Y)` searched the latent mode of an offset-free model and `predict(fit, Y)`
# returned β + Λẑ without the offset. The pattern is #787's (test/test_predict_offset.jl)
# and #801's (test/test_predict_offset_grouped.jl): the fit stores `offset`, and
# `_laplace_prediction_offset` applies the training offset to a Y of the training size and
# refuses new units without an explicit offset. On the two-part fits the offset is on the
# count / positive-part predictor η^c, as at fit time.
#
# The references are not produced by the code under test: the mode condition is the
# central finite-difference gradient of the offset-aware log posterior, written out here
# from the Distributions.jl densities (GP-1, which Distributions.jl lacks, from its pmf
# formula); the objective check evaluates each family's public Laplace marginal with the
# stored offset.

_ps_lg(x) = 1 / (1 + exp(-x))

# GP-1 log pmf (Famoye 1993, mean parameterisation): g = 1 + αμ, h = 1 + αy.
_ps_gp1(α, μ, y) = (g = 1 + α * μ; h = 1 + α * y;
                    y * (log(μ) - log(g)) + (y - 1) * log(h) - loggamma(y + 1.0) - μ * h / g)

# log p(y | z) − z'z/2 at one site; `ll(t, y, η)` is trait t's conditional log density at
# linear predictor η (on a two-part fit, η is the count / positive-part predictor).
_ps_logpost(ll, y, β, o, Λ, z) =
    (η = β .+ o .+ Λ * z; sum(ll(t, y[t], η[t]) for t in eachindex(y)) - dot(z, z) / 2)

function _ps_fd_score(ll, y, β, o, Λ, z; h = 1e-5)
    g = similar(z)
    for k in eachindex(z)
        e = zeros(length(z)); e[k] = h
        g[k] = (_ps_logpost(ll, y, β, o, Λ, z .+ e) - _ps_logpost(ll, y, β, o, Λ, z .- e)) / (2h)
    end
    return g
end

# The largest finite-difference score norm over the sites; ≈ 0 when every mode is a zero
# of the offset-aware score.
_ps_worst_score(ll, Y, β, Λ, O, Z) =
    maximum(norm(_ps_fd_score(ll, Y[:, s], β, O[:, s], Λ, Z[s, :])) for s in axes(Y, 2))

function _ps_ztnb(rng, r, μ)
    while true
        y = rand(rng, NegativeBinomial(r, r / (r + μ))); y > 0 && return y
    end
end

# Simulated data with a p×n offset O (seeded per family).
function _ps_sim(name; p = 5, n = 50, K = 1)
    rng = MersenneTwister(788 + length(name))
    O = 0.8 .* randn(rng, p, n); Λ = 0.6 .* randn(rng, p, K); Z = randn(rng, n, K)
    η = (name in ("Beta", "ZIB") ? 0.0 : 0.7) .+ O .+ Λ * Z'
    Y = if name == "Gamma"
        [rand(rng, Gamma(3.0, exp(e) / 3)) for e in η]
    elseif name == "Beta"
        [clamp(rand(rng, Beta(_ps_lg(e) * 8, (1 - _ps_lg(e)) * 8)), 1e-4, 1 - 1e-4) for e in η]
    elseif name in ("NB1", "GP1")
        [rand(rng, NegativeBinomial(exp(e) / 0.5, 1 / 1.5)) for e in η]
    elseif name == "Exponential"
        [rand(rng, Exponential(exp(e))) for e in η]
    elseif name == "HurdleNB"
        [rand(rng) < 0.7 ? _ps_ztnb(rng, 3.0, exp(e)) : 0 for e in η]
    elseif name == "ZIP"
        [rand(rng) < 0.3 ? 0 : rand(rng, Poisson(exp(e))) for e in η]
    elseif name == "ZINB"
        [rand(rng) < 0.3 ? 0 : rand(rng, NegativeBinomial(3.0, 3.0 / (3.0 + exp(e)))) for e in η]
    elseif name == "ZIB"
        [rand(rng) < 0.3 ? 0 : rand(rng, Binomial(5, _ps_lg(e))) for e in η]
    elseif name == "DeltaLogNormal"
        [rand(rng) < 0.7 ? rand(rng, LogNormal(e, 0.5)) : 0.0 for e in η]
    else # DeltaGamma
        [rand(rng) < 0.7 ? rand(rng, Gamma(3.0, exp(e) / 3)) : 0.0 for e in η]
    end
    return Y, O
end

_ps_disp(d, t) = d isa Real ? d : d[t]

# Per family: fitter, conditional log density ll(f) -> (t, y, η), Laplace marginal at given
# offset (keyword `off`), and response-scale mean from the link predictor.
const _PS_CASES = (
    ("Gamma", Y -> (; kw...) -> fit_gamma_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> logpdf(Gamma(f.α, exp(η) / f.α), y),
     (f, Y, off) -> GM.gamma_marginal_loglik_laplace(Y, f.Λ, f.β, f.α; link = f.link,
                                                      offset = off, hessian = f.hessian),
     (f, η) -> exp.(η)),
    ("Beta", Y -> (; kw...) -> fit_beta_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> (μ = _ps_lg(η); logpdf(Beta(μ * f.φ, (1 - μ) * f.φ), y)),
     (f, Y, off) -> GM.beta_marginal_loglik_laplace(Y, f.Λ, f.β, f.φ; link = f.link,
                                                     offset = off, hessian = f.hessian),
     (f, η) -> _ps_lg.(η)),
    ("NB1", Y -> (; kw...) -> fit_nb1_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> logpdf(NegativeBinomial(exp(η) / f.φ, 1 / (1 + f.φ)), y),
     (f, Y, off) -> GM.nb1_marginal_loglik_laplace(Y, f.Λ, f.β, f.φ; link = f.link,
                                                    offset = off, hessian = f.hessian),
     (f, η) -> exp.(η)),
    ("GP1", Y -> (; kw...) -> fit_gp1_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> _ps_gp1(f.α, exp(η), y),
     (f, Y, off) -> GM.marginal_loglik_laplace(GM.GeneralizedPoisson1(f.α), Y, ones(Int, size(Y)),
                                               f.Λ, f.β, f.link; offset = off, hessian = f.hessian),
     (f, η) -> exp.(η)),
    ("Exponential", Y -> (; kw...) -> fit_exponential_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> logpdf(Exponential(exp(η)), y),
     (f, Y, off) -> GM.exponential_marginal_loglik_laplace(Y, f.Λ, f.β; link = f.link,
                                                            offset = off, hessian = f.hessian),
     (f, η) -> exp.(η)),
    ("HurdleNB", Y -> (; kw...) -> fit_hurdle_nb_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> (π = _ps_lg(f.βz[t]); μ = exp(η); d = NegativeBinomial(f.r, f.r / (f.r + μ));
                        y == 0 ? log(1 - π) : log(π) + logpdf(d, y) - log1p(-pdf(d, 0))),
     (f, Y, off) -> GM.hurdle_nb_marginal_loglik_laplace(Y, f.Λc, f.βz, f.βc, f.r; offsetc = off),
     (f, η) -> (μ = exp.(η); _ps_lg.(f.βz) .* μ ./ (1 .- (f.r ./ (f.r .+ μ)) .^ f.r))),
    ("ZIP", Y -> (; kw...) -> fit_zip_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> (π = _ps_lg(f.βz[t]); d = Poisson(exp(η));
                        y == 0 ? log(π + (1 - π) * pdf(d, 0)) : log(1 - π) + logpdf(d, y)),
     (f, Y, off) -> GM.zip_marginal_loglik_laplace(Y, f.Λc, f.βz, f.βc; offsetc = off),
     (f, η) -> (1 .- _ps_lg.(f.βz)) .* exp.(η)),
    ("ZINB", Y -> (; kw...) -> fit_zinb_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> (π = _ps_lg(f.βz[t]); μ = exp(η); d = NegativeBinomial(f.r, f.r / (f.r + μ));
                        y == 0 ? log(π + (1 - π) * pdf(d, 0)) : log(1 - π) + logpdf(d, y)),
     (f, Y, off) -> GM.zinb_marginal_loglik_laplace(Y, f.Λc, f.βz, f.βc, f.r; offsetc = off),
     (f, η) -> (1 .- _ps_lg.(f.βz)) .* exp.(η)),
    ("ZIB", Y -> (; kw...) -> fit_zib_gllvm(Y; K = 1, N = 5, kw...),
     f -> (t, y, η) -> (π = _ps_lg(f.βz[t]); d = Binomial(f.N, _ps_lg(η));
                        y == 0 ? log(π + (1 - π) * pdf(d, 0)) : log(1 - π) + logpdf(d, Int(y))),
     (f, Y, off) -> GM.zib_marginal_loglik_laplace(Y, f.Λc, f.βz, f.βc, f.N; offsetc = off),
     (f, η) -> (1 .- _ps_lg.(f.βz)) .* f.N .* _ps_lg.(η)),
    ("DeltaLogNormal", Y -> (; kw...) -> fit_delta_lognormal_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> (π = _ps_lg(f.βz[t]);
                        y == 0 ? log(1 - π) : log(π) + logpdf(LogNormal(η, _ps_disp(f.σ, t)), y)),
     (f, Y, off) -> GM.delta_lognormal_marginal_loglik_laplace(Y, f.Λc, f.βz, f.βc, f.σ; offsetc = off),
     (f, η) -> _ps_lg.(f.βz) .* exp.(η .+ (f.σ isa Real ? f.σ^2 : f.σ .^ 2) ./ 2)),
    ("DeltaGamma", Y -> (; kw...) -> fit_delta_gamma_gllvm(Y; K = 1, kw...),
     f -> (t, y, η) -> (π = _ps_lg(f.βz[t]); α = _ps_disp(f.α, t);
                        y == 0 ? log(1 - π) : log(π) + logpdf(Gamma(α, exp(η) / α), y)),
     (f, Y, off) -> GM.delta_gamma_marginal_loglik_laplace(Y, f.Λc, f.βz, f.βc, f.α; offsetc = off),
     (f, η) -> _ps_lg.(f.βz) .* exp.(η)),
)

_ps_beta(f) = hasproperty(f, :βc) ? f.βc : f.β
_ps_load(f) = hasproperty(f, :Λc) ? f.Λc : f.Λ
_ps_res(f, Y; kw...) = f isa Union{GammaFit, BetaFit, ExponentialFit} ? residuals(f, Y; kw...) :
                                                                        residuals(f, Y; rng = MersenneTwister(1), kw...)

# The Dunn-Smyth interval [F(y-1), F(y)] of the zero-inflated / hurdle count CDF.
function _ps_ds_interval(name, f, t, y, η)
    μ = exp(η)
    if name == "HurdleNB"
        π = _ps_lg(f.βz[t]); d = NegativeBinomial(f.r, f.r / (f.r + μ)); p0 = pdf(d, 0)
        y == 0 && return (0.0, 1 - π)
        Flo = y == 1 ? 0.0 : (cdf(d, y - 1) - p0) / (1 - p0)
        return ((1 - π) + π * Flo, (1 - π) + π * (cdf(d, y) - p0) / (1 - p0))
    end
    π = _ps_lg(f.βz[t])
    d = name == "ZIP" ? Poisson(μ) : name == "ZINB" ? NegativeBinomial(f.r, f.r / (f.r + μ)) :
        Binomial(f.N, _ps_lg(η))
    y == 0 && return (0.0, π + (1 - π) * cdf(d, 0))
    return (π + (1 - π) * cdf(d, y - 1), π + (1 - π) * cdf(d, y))
end

@testset "shared-dispersion fits use the training offset (#788)" begin
    p, n = 5, 50
    @testset "$name" for (name, mkfit, mkll, marg, mean_of) in _PS_CASES
        Y, O = _ps_sim(name)
        fit = mkfit(Y)
        f = fit(; offset = O)
        @test f.offset == O                                   # stored verbatim
        β, Λ = _ps_beta(f), _ps_load(f)
        # The stored offset is the fit's own objective; the offset-free one is not.
        @test isapprox(marg(f, Y, f.offset), f.loglik; rtol = 1e-6)
        @test !isapprox(marg(f, Y, nothing), f.loglik; rtol = 1e-3)
        ll = mkll(f)
        Z = getLV(f, Y; rotate = false)
        @test _ps_worst_score(ll, Y, β, Λ, O, Z) < 1e-4
        @test getLV(f, Y; rotate = false, offset = O) ≈ Z atol = 1e-12
        ηh = predict(f, Y; type = :link)
        @test ηh ≈ β .+ O .+ Λ * Z' atol = 1e-10
        @test predict(f, Y; type = :link, offset = O) ≈ ηh atol = 1e-12
        @test predict(f, Y) ≈ mean_of(f, ηh) rtol = 1e-10
        # Residuals are evaluated at the offset-aware conditional mean.
        R = _ps_res(f, Y)
        @test R == _ps_res(f, Y; offset = O)
        if f isa Union{GammaFit, BetaFit, NB1Fit, GP1Fit, ExponentialFit}
            μ = mean_of(f, ηh)
            sd = f isa GammaFit ? μ ./ sqrt(f.α) :
                 f isa BetaFit ? sqrt.(μ .* (1 .- μ) ./ (1 + f.φ)) :
                 f isa NB1Fit ? sqrt.(μ .* (1 + f.φ)) :
                 f isa GP1Fit ? sqrt.(μ .* (1 .+ f.α .* μ) .^ 2) : μ
            @test residuals(f, Y; type = :pearson) ≈ (Y .- μ) ./ sd rtol = 1e-10
        elseif name in ("DeltaLogNormal", "DeltaGamma")
            # A positive cell's residual is the deterministic PIT of the two-part CDF.
            for s in 1:n, t in 1:p
                Y[t, s] > 0 || continue
                π = _ps_lg(f.βz[t])
                G = name == "DeltaGamma" ?
                    cdf(Gamma(_ps_disp(f.α, t), exp(ηh[t, s]) / _ps_disp(f.α, t)), Y[t, s]) :
                    cdf(LogNormal(ηh[t, s], _ps_disp(f.σ, t)), Y[t, s])
                @test R[t, s] ≈ quantile(Normal(), clamp((1 - π) + π * G, 1e-12, 1 - 1e-12)) atol = 1e-8
            end
        else
            # A randomised residual's uniform lies in its cell's CDF interval.
            inside = all(begin
                lo, hi = _ps_ds_interval(name, f, t, Int(Y[t, s]), ηh[t, s])
                u = cdf(Normal(), R[t, s])
                lo - 1e-8 <= u <= hi + 1e-8
            end for s in 1:n, t in 1:p)
            @test inside
        end
        # New units: the training offset does not apply; an explicit one is required.
        Ynew = Y[:, 1:20]; Onew = O[:, 1:20] .+ 0.3
        @test_throws ArgumentError getLV(f, Ynew)
        @test_throws ArgumentError predict(f, Ynew)
        @test_throws ArgumentError _ps_res(f, Ynew)
        @test getLV(f, Ynew; rotate = false, offset = O[:, 1:20]) ≈ Z[1:20, :] atol = 1e-8
        Zn = getLV(f, Ynew; rotate = false, offset = Onew)
        @test _ps_worst_score(ll, Ynew, β, Λ, Onew, Zn) < 1e-4
        @test predict(f, Ynew; type = :link, offset = Onew) ≈ β .+ Onew .+ Λ * Zn' atol = 1e-10
        # A scalar or length-p offset means the p×n matrix it broadcasts to.
        @test predict(f, Ynew; type = :link, offset = 0.25) ≈
              predict(f, Ynew; type = :link, offset = fill(0.25, p, 20)) atol = 1e-12
        @test getLV(f, Ynew; offset = fill(0.1, p)) ≈ getLV(f, Ynew; offset = fill(0.1, p, 20)) atol = 1e-12
        # A fit without an offset keeps nothing and predicts new units as before.
        f0 = fit()
        @test f0.offset === nothing
        @test predict(f0, Ynew; type = :link) ≈
              _ps_beta(f0) .+ _ps_load(f0) * getLV(f0, Ynew; rotate = false)'
    end

    @testset "fit_gllvm stores the offset it passes on" begin
        Y, O = _ps_sim("Gamma")
        f = fit_gllvm(Y; family = Gamma(), K = 1, offset = O)
        @test f isa GammaFit && f.offset == O
        @test predict(f, Y; type = :link) ≈ f.β .+ O .+ f.Λ * getLV(f, Y; rotate = false)' atol = 1e-10
        Y, O = _ps_sim("ZIP")
        f = fit_gllvm(Y; family = ZIPoisson(), K = 1, offset = O)
        @test f isa ZIPFit && f.offset == O
    end

    @testset "positional constructors without an offset still work" begin
        β = [0.1, 0.2]; Λ = reshape([0.5, -0.4], 2, 1)
        for f in (GammaFit(β, Λ, 2.0, GM.LogLink(), -1.0, true, 1),
                  GammaFit(β, Λ, 2.0, GM.LogLink(), -1.0, true, 1, nothing, Float64[]),
                  GammaFit(β, Λ, 2.0, GM.LogLink(), -1.0, true, 1, nothing, Float64[], :observed),
                  BetaFit(β, Λ, 5.0, GM.LogitLink(), -1.0, true, 1),
                  BetaFit(β, Λ, 5.0, GM.LogitLink(), -1.0, true, 1, nothing, Float64[], :observed),
                  NB1Fit(β, Λ, 0.5, GM.LogLink(), -1.0, true, 1),
                  NB1Fit(β, Λ, 0.5, GM.LogLink(), -1.0, true, 1, :observed),
                  GP1Fit(β, Λ, 0.1, GM.LogLink(), -1.0, true, 1),
                  GP1Fit(β, Λ, 0.1, GM.LogLink(), -1.0, true, 1, :observed),
                  ExponentialFit(β, Λ, GM.LogLink(), -1.0, true, 1),
                  ExponentialFit(β, Λ, GM.LogLink(), -1.0, true, 1, :observed),
                  HurdleNBFit(β, β, Λ, 2.0, -1.0, true, 1),
                  ZIPFit(β, β, Λ, -1.0, true, 1),
                  ZINBFit(β, β, Λ, 2.0, -1.0, true, 1),
                  ZIBFit(β, β, Λ, 5, -1.0, true, 1),
                  DeltaLogNormalFit(β, β, Λ, 0.5, -1.0, true, 1),
                  DeltaLogNormalFit(β, β, Λ, 0.5, -1.0, true, 1, :separate, :shared),
                  DeltaGammaFit(β, β, Λ, 2.0, -1.0, true, 1),
                  DeltaGammaFit(β, β, Λ, 2.0, -1.0, true, 1, :separate, :shared))
            @test f.offset === nothing
        end
    end
end
