using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

# predict / getLV / residuals on a grouped-dispersion fit made with an offset (#788).
#
# The bug this file pins: the grouped-dispersion fits (NBGroupedFit and its NB1, Beta and
# Gamma siblings) accepted `offset` at fit time but did not keep it, so `getLV(fit, Y)`
# searched the latent mode of an offset-free model and `predict(fit, Y)` returned β + Λẑ
# without the offset. The formula front end routes `NegativeBinomial()` to
# `NBGroupedFit` (`disp_group = :species`, gllvmTMB's default), so this was the default NB
# route. On the p = 5, n = 50 NB2 data below the training-row link predictor missed
# β + O + Λẑ_O by 1.83 before the fix. The pattern is #787's (test/test_predict_offset.jl):
# the fit stores `offset`, and `_laplace_prediction_offset` applies the training offset to
# a Y of the training size and refuses new units without an explicit offset.
#
# The references are not produced by the code under test: the mode condition is the
# central finite-difference gradient of the offset-aware log posterior, written out here
# from the Distributions.jl densities; the objective check evaluates the public grouped
# marginal with the stored offset.

_glin(f, O, Z) = f.β .+ O .+ f.Λ * Z'
# NB2 with size r and mean μ. A group at the Poisson limit (fitted r of order 1e10 here) is
# evaluated as Poisson(μ): the two log densities differ by O(μ²/r), while Distributions'
# NB2 logpdf at such r is too noisy for a finite-difference score.
_nb2(r, μ) = r >= 1e6 * max(μ, 1.0) ? Poisson(μ) : NegativeBinomial(r, r / (r + μ))
_logistic(x) = 1 / (1 + exp(-x))

# log p(y | z) − z'z/2 at one site; `dist(t, μ)` is trait t's conditional distribution.
function _site_logpost(dist, y, β, o, Λ, z, invlink)
    η = β .+ o .+ Λ * z
    return sum(logpdf(dist(t, invlink(η[t])), y[t]) for t in eachindex(y)) - dot(z, z) / 2
end

# Central finite-difference gradient of the site log posterior in z.
function _fd_score(dist, y, β, o, Λ, z, invlink; h = 1e-5)
    g = similar(z)
    for k in eachindex(z)
        e = zeros(length(z)); e[k] = h
        g[k] = (_site_logpost(dist, y, β, o, Λ, z .+ e, invlink) -
                _site_logpost(dist, y, β, o, Λ, z .- e, invlink)) / (2h)
    end
    return g
end

# Every site's mode is a zero of the offset-aware score.
function _modes_are_score_zeros(dist, Y, f, O, Z, invlink; tol = 1e-4)
    worst = 0.0
    for s in axes(Y, 2)
        g = _fd_score(dist, Y[:, s], f.β, O[:, s], f.Λ, Z[s, :], invlink)
        worst = max(worst, norm(g))
    end
    return worst < tol
end

function _sim_grouped(rng, p, n, K)
    O = 0.8 .* randn(rng, p, n)
    Λ = 0.6 .* randn(rng, p, K); Z = randn(rng, n, K)
    return O, Λ, Z
end

@testset "grouped-dispersion fits use the training offset (#788)" begin
    p, n, K = 5, 50, 1

    @testset "NB2 grouped (fit_nb_gllvm_grouped)" begin
        rng = MersenneTwister(788)
        O, Λt, Zt = _sim_grouped(rng, p, n, K)
        η = 0.7 .+ O .+ Λt * Zt'
        rt = [2.0, 4.0, 8.0, 3.0, 6.0]
        Y = [rand(rng, NegativeBinomial(rt[t], rt[t] / (rt[t] + exp(η[t, s])))) for t in 1:p, s in 1:n]
        f = fit_nb_gllvm_grouped(Y; K = K, group = collect(1:p), offset = O)
        @test f.offset == O                                   # stored verbatim
        rvec = f.r_group[f.group]
        # The stored offset is the fit's own objective; the offset-free one is not.
        @test isapprox(GM.nb_grouped_marginal_loglik_laplace(Y, f.Λ, f.β, rvec; link = f.link,
                                                             offset = f.offset, hessian = f.hessian),
                       f.loglik; rtol = 1e-6)
        @test !isapprox(GM.nb_grouped_marginal_loglik_laplace(Y, f.Λ, f.β, rvec; link = f.link,
                                                              hessian = f.hessian),
                        f.loglik; rtol = 1e-3)
        nbdist(t, μ) = _nb2(rvec[t], μ)
        Z = getLV(f, Y; rotate = false)
        @test _modes_are_score_zeros(nbdist, Y, f, O, Z, exp)
        @test getLV(f, Y; rotate = false, offset = O) ≈ Z atol = 1e-12
        ηh = predict(f, Y; type = :link)
        @test ηh ≈ _glin(f, O, Z) atol = 1e-10
        @test predict(f, Y) ≈ exp.(ηh)
        @test predict(f, Y; type = :link, offset = O) ≈ ηh atol = 1e-12
        μ = exp.(ηh)
        @test residuals(f, Y; type = :pearson) ≈ (Y .- μ) ./ sqrt.(μ .+ μ .^ 2 ./ rvec)
        # New units: the training offset does not apply; an explicit one is required.
        Ynew = Y[:, 1:20]; Onew = O[:, 1:20] .+ 0.3
        @test_throws ArgumentError predict(f, Ynew)
        @test_throws ArgumentError getLV(f, Ynew)
        @test_throws ArgumentError residuals(f, Ynew; type = :pearson)
        Zn = getLV(f, Ynew; rotate = false, offset = Onew)
        @test _modes_are_score_zeros(nbdist, Ynew, f, Onew, Zn, exp)
        @test predict(f, Ynew; type = :link, offset = Onew) ≈ _glin(f, Onew, Zn) atol = 1e-10
        # A scalar offset is broadcast and stored as the p×n matrix it means; a length-p
        # vector likewise (one offset per trait).
        @test predict(f, Ynew; type = :link, offset = 0.25) ≈
              predict(f, Ynew; type = :link, offset = fill(0.25, p, 20)) atol = 1e-12
        @test getLV(f, Ynew; offset = fill(0.1, p)) ≈ getLV(f, Ynew; offset = fill(0.1, p, 20)) atol = 1e-12
        # A fit without an offset keeps nothing and predicts new units as before.
        f0 = fit_nb_gllvm_grouped(Y; K = K, group = collect(1:p))
        @test f0.offset === nothing
        @test predict(f0, Ynew; type = :link) ≈ f0.β .+ f0.Λ * getLV(f0, Ynew; rotate = false)'
    end

    @testset "formula front end, family = NegativeBinomial() (default disp_group = :species)" begin
        rng = MersenneTwister(788)
        O, Λt, Zt = _sim_grouped(rng, p, n, K)
        η = 0.7 .+ O .+ Λt * Zt'
        rt = [2.0, 4.0, 8.0, 3.0, 6.0]
        Y = [rand(rng, NegativeBinomial(rt[t], rt[t] / (rt[t] + exp(η[t, s])))) for t in 1:p, s in 1:n]
        f = gllvm(@formula(y ~ 1), Y, (site = 1:n,); family = NegativeBinomial(), K = K, offset = O)
        @test f isa NBGroupedFit
        @test f.offset == O
        Z = getLV(f, Y; rotate = false)
        rvec = f.r_group[f.group]
        @test _modes_are_score_zeros((t, μ) -> _nb2(rvec[t], μ),
                                     Y, f, O, Z, exp)
        @test predict(f, Y; type = :link) ≈ _glin(f, O, Z) atol = 1e-10
        # Same fit as the named fitter (the formula route only adds the per-trait groups).
        fn = fit_nb_gllvm_grouped(Y; K = K, group = collect(1:p), offset = O)
        @test predict(f, Y; type = :link) ≈ predict(fn, Y; type = :link) atol = 1e-8
        # A scalar formula offset is stored as the p×n matrix it means.
        @test gllvm(@formula(y ~ 1), Y, (site = 1:n,); family = NegativeBinomial(), K = K,
                    offset = log(2.0)).offset == fill(log(2.0), p, n)
        @test gllvm(@formula(y ~ 1), Y, (site = 1:n,); family = NegativeBinomial(), K = K).offset === nothing
    end

    # The other grouped fits that take an offset have getLV only (no predict / residuals).
    @testset "$name grouped" for (name, fitter, simY, dist, invlink, marg, disp) in (
        ("NB1", fit_nb1_gllvm_grouped,
         (rng, η) -> [rand(rng, NegativeBinomial(exp(η[t, s]) / 1.5, 1 / 2.5)) for t in axes(η, 1), s in axes(η, 2)],
         (φ) -> (t, μ) -> NegativeBinomial(μ / φ[t], 1 / (1 + φ[t])), exp,
         GM.nb1_grouped_marginal_loglik_laplace, f -> f.φ[f.group]),
        ("Gamma", fit_gamma_gllvm_grouped,
         (rng, η) -> [rand(rng, Gamma(3.0, exp(η[t, s]) / 3.0)) for t in axes(η, 1), s in axes(η, 2)],
         (α) -> (t, μ) -> Gamma(α[t], μ / α[t]), exp,
         GM.gamma_grouped_marginal_loglik_laplace, f -> f.α[f.group]),
        ("Beta", fit_beta_gllvm_grouped,
         (rng, η) -> [clamp(rand(rng, Beta(_logistic(η[t, s]) * 8, (1 - _logistic(η[t, s])) * 8)), 1e-4, 1 - 1e-4)
                      for t in axes(η, 1), s in axes(η, 2)],
         (φ) -> (t, μ) -> Beta(μ * φ[t], (1 - μ) * φ[t]), _logistic,
         GM.beta_grouped_marginal_loglik_laplace, f -> f.φ[f.group]),
    )
        rng = MersenneTwister(7880 + length(name))
        O, Λt, Zt = _sim_grouped(rng, p, n, K)
        η = (name == "Beta" ? 0.0 : 0.7) .+ O .+ Λt * Zt'
        Y = simY(rng, η)
        f = fitter(Y; K = K, group = collect(1:p), offset = O)
        @test f.offset == O
        dv = disp(f)
        @test isapprox(marg(Y, f.Λ, f.β, dv; link = f.link, offset = f.offset, hessian = f.hessian),
                       f.loglik; rtol = 1e-6)
        Z = getLV(f, Y; rotate = false)
        @test _modes_are_score_zeros(dist(dv), Y, f, O, Z, invlink)
        @test getLV(f, Y; rotate = false, offset = O) ≈ Z atol = 1e-12
        @test_throws ArgumentError getLV(f, Y[:, 1:20])
        Zn = getLV(f, Y[:, 1:20]; rotate = false, offset = O[:, 1:20])
        @test Zn ≈ Z[1:20, :] atol = 1e-8                 # sites are independent given θ
        @test fitter(Y; K = K, group = collect(1:p)).offset === nothing
    end

    @testset "positional constructors without an offset still work" begin
        β = [0.1, 0.2]; Λ = reshape([0.5, -0.4], 2, 1); g = [1, 2]
        for f in (NBGroupedFit(β, Λ, [2.0, 3.0], g, GM.LogLink(), -1.0, true, 1),
                  NBGroupedFit(β, Λ, [2.0, 3.0], g, GM.LogLink(), -1.0, true, 1, :fisher),
                  NBGroupedFit(β, Λ, [2.0, 3.0], g, GM.LogLink(), -1.0, true, 1, :observed, [false, false]),
                  NB1GroupedFit(β, Λ, [0.5, 1.0], g, GM.LogLink(), -1.0, true, 1),
                  NB1GroupedFit(β, Λ, [0.5, 1.0], g, GM.LogLink(), -1.0, true, 1, :observed, [false, false]),
                  BetaGroupedFit(β, Λ, [5.0, 6.0], g, GM.LogitLink(), -1.0, true, 1),
                  BetaGroupedFit(β, Λ, [5.0, 6.0], g, GM.LogitLink(), -1.0, true, 1, :observed, [false, false]),
                  GammaGroupedFit(β, Λ, [2.0, 3.0], g, GM.LogLink(), -1.0, true, 1),
                  GammaGroupedFit(β, Λ, [2.0, 3.0], g, GM.LogLink(), -1.0, true, 1, :observed, [false, false]))
            @test f.offset === nothing
        end
    end
end
