using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

# predict / getLV / residuals on a Laplace fit made with an offset
# (η = β + offset + Λz, e.g. log-exposure).
#
# The bug this file pins: the Laplace Poisson, NB2, binomial and hurdle-Poisson fits did
# not keep the offset they were fitted with, so `getLV(fit, Y)` searched the latent mode of
# an offset-free model and `predict(fit, Y)` returned β + Λẑ without the offset (a Poisson
# fit, p = 6, n = 60, offset 0.8 * randn: max |Δη| = 3.17 on the link scale). gllvmTMB's
# `predict` uses the stored training offset (`.gllvmTMB_offset_vec`) on training rows and
# re-evaluates the offset on new rows, refusing new rows that lack it
# (`.gllvmTMB_offset_newdata`).
#
# The references are not produced by the code under test: the mode condition is the
# score of the offset-aware log posterior, written out here from the family densities;
# the objective check evaluates the public marginal with the stored offset.

_lin(f, O, Z) = f.β .+ O .+ f.Λ * Z'

# Score of log p(y | z) − z'z / 2 in z, log link (Poisson) or NB2 with dispersion r.
function _count_score(y, N, η, Λ, z; r = Inf)
    μ = exp.(η)
    w = isinf(r) ? (y .- μ) : (y .- μ) .* (r ./ (r .+ μ))
    return Λ' * w .- z
end
# Binomial-logit score with trials N.
_binom_score(y, N, η, Λ, z) = Λ' * (y .- N ./ (1 .+ exp.(-η))) .- z
# Hurdle-Poisson count part (zero-truncated Poisson on y > 0; zeros carry no z).
function _hurdle_score(y, η, Λ, z)
    g = zeros(length(η))
    for t in eachindex(y)
        y[t] > 0 || continue
        μ = exp(η[t])
        g[t] = y[t] - μ - μ * exp(-μ) / (1 - exp(-μ))
    end
    return Λ' * g .- z
end

function _sim_counts(rng, p, n, K; nb = false)
    O = 0.8 .* randn(rng, p, n)
    Λ = 0.6 .* randn(rng, p, K); β = fill(0.7, p)
    Z = randn(rng, n, K)
    η = β .+ O .+ Λ * Z'
    Y = nb ? [rand(rng, NegativeBinomial(5.0, 5.0 / (5.0 + exp(η[t, s])))) for t in 1:p, s in 1:n] :
             [rand(rng, Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
    return Y, O
end

@testset "predict / getLV use the training offset (Laplace)" begin
    p, n, K = 5, 50, 1

    @testset "Poisson" begin
        Y, O = _sim_counts(MersenneTwister(11), p, n, K)
        f = fit_poisson_gllvm(Y; K = K, offset = O)
        @test f.offset == O                                   # stored verbatim
        # The stored offset is the fit's own objective; the offset-free one is not.
        @test isapprox(GM.poisson_marginal_loglik_laplace(Y, f.Λ, f.β; offset = f.offset),
                       f.loglik; rtol = 1e-6)
        @test !isapprox(GM.poisson_marginal_loglik_laplace(Y, f.Λ, f.β), f.loglik; rtol = 1e-3)
        Z = getLV(f, Y; rotate = false)
        for s in 1:n
            η = f.β .+ O[:, s] .+ f.Λ * Z[s, :]
            @test norm(_count_score(Y[:, s], 1, η, f.Λ, Z[s, :])) < 1e-6
        end
        η = predict(f, Y; type = :link)
        @test η ≈ _lin(f, O, Z) atol = 1e-10
        @test predict(f, Y) ≈ exp.(η)
        @test predict(f, Y; type = :link, offset = O) ≈ η atol = 1e-12
        μ = exp.(η)
        @test residuals(f, Y; type = :pearson) ≈ (Y .- μ) ./ sqrt.(μ)
        # New units: the training offset does not apply; an explicit one is required.
        Ynew, Onew = _sim_counts(MersenneTwister(12), p, 20, K)
        @test_throws ArgumentError predict(f, Ynew)
        @test_throws ArgumentError getLV(f, Ynew)
        Zn = getLV(f, Ynew; rotate = false, offset = Onew)
        for s in 1:20
            ηs = f.β .+ Onew[:, s] .+ f.Λ * Zn[s, :]
            @test norm(_count_score(Ynew[:, s], 1, ηs, f.Λ, Zn[s, :])) < 1e-6
        end
        @test predict(f, Ynew; type = :link, offset = Onew) ≈ _lin(f, Onew, Zn) atol = 1e-10
        # A scalar offset is broadcast and stored as the p×n matrix it means.
        fs = fit_poisson_gllvm(Y; K = K, offset = log(2.0))
        @test fs.offset == fill(log(2.0), p, n)
        @test predict(fs, Y; type = :link) ≈ _lin(fs, fill(log(2.0), p, n), getLV(fs, Y; rotate = false)) atol = 1e-10
        # The unified dispatcher stores it too.
        @test fit_gllvm(Y; family = Poisson(), K = K, offset = O).offset == O
        # A fit without an offset keeps nothing and predicts new units as before.
        f0 = fit_poisson_gllvm(Y; K = K)
        @test f0.offset === nothing
        @test predict(f0, Ynew; type = :link) ≈ f0.β .+ f0.Λ * getLV(f0, Ynew; rotate = false)'
    end

    @testset "NB2" begin
        Y, O = _sim_counts(MersenneTwister(21), p, n, K; nb = true)
        f = fit_nb_gllvm(Y; K = K, offset = O)
        @test f.offset == O
        @test isapprox(GM.nb_marginal_loglik_laplace(Y, f.Λ, f.β, f.r; offset = f.offset,
                                                     hessian = f.hessian),
                       f.loglik; rtol = 1e-6)
        Z = getLV(f, Y; rotate = false)
        for s in 1:n
            η = f.β .+ O[:, s] .+ f.Λ * Z[s, :]
            @test norm(_count_score(Y[:, s], 1, η, f.Λ, Z[s, :]; r = f.r)) < 1e-6
        end
        η = predict(f, Y; type = :link)
        @test η ≈ _lin(f, O, Z) atol = 1e-10
        @test predict(f, Y; type = :link, offset = O) ≈ η atol = 1e-12
        μ = exp.(η)
        @test residuals(f, Y; type = :pearson) ≈ (Y .- μ) ./ sqrt.(μ .+ μ .^ 2 ./ f.r)
        @test_throws ArgumentError predict(f, Y[:, 1:10])
        @test predict(f, Y[:, 1:10]; type = :link, offset = O[:, 1:10]) ≈ η[:, 1:10] atol = 1e-10
    end

    @testset "Binomial" begin
        rng = MersenneTwister(31)
        O = 0.8 .* randn(rng, p, n)
        Λt = 0.6 .* randn(rng, p, K); Zt = randn(rng, n, K)
        Nm = fill(4, p, n)
        Y = [rand(rng, Binomial(4, 1 / (1 + exp(-(O[t, s] + dot(Λt[t, :], Zt[s, :])))))) for t in 1:p, s in 1:n]
        f = fit_binomial_gllvm(Y; K = K, N = Nm, offset = O)
        @test f.offset == O
        @test isapprox(GM.binomial_marginal_loglik_laplace(Y, Nm, f.Λ, f.β, f.link;
                                                           offset = f.offset, hessian = f.hessian),
                       f.loglik; rtol = 1e-6)
        Z = getLV(f, Y; N = Nm, rotate = false)
        for s in 1:n
            η = f.β .+ O[:, s] .+ f.Λ * Z[s, :]
            @test norm(_binom_score(Y[:, s], Nm[:, s], η, f.Λ, Z[s, :])) < 1e-6
        end
        η = predict(f, Y; N = Nm, type = :link)
        @test η ≈ _lin(f, O, Z) atol = 1e-10
        @test predict(f, Y; N = Nm) ≈ 1 ./ (1 .+ exp.(-η))
        @test_throws ArgumentError predict(f, Y[:, 1:10]; N = Nm[:, 1:10])
    end

    @testset "Hurdle-Poisson (count offset)" begin
        Y, O = _sim_counts(MersenneTwister(41), p, n, K)
        f = fit_hurdle_poisson_gllvm(Y; K = K, offset = O)
        @test f.offset == O
        Z = getLV(f, Y; rotate = false)
        for s in 1:n
            η = f.βc .+ O[:, s] .+ f.Λc * Z[s, :]
            @test norm(_hurdle_score(Y[:, s], η, f.Λc, Z[s, :])) < 1e-6
        end
        @test predict(f, Y; type = :link) ≈ f.βc .+ O .+ f.Λc * Z' atol = 1e-10
        @test_throws ArgumentError predict(f, Y[:, 1:10])
    end
end
