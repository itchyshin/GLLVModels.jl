using GLLVModels, Test, Random, Distributions, ForwardDiff, LinearAlgebra

# #1209: per-cell 1e-8 weight floor in the Laplace log-determinant biased logLik
# at the fitted optimum. Raw observed weights in the log-det; floor only in mode
# search; converged = false with reason :boundary when any cell binds.

@testset "Beta-binomial Laplace weight floor (#1209)" begin

    @testset "issue cell: negative raw weight; mode search still floors" begin
        f(η) = GLLVModels.betabinomial_logp(0, η, 10, 0.5)
        Wraw = -ForwardDiff.derivative(e -> ForwardDiff.derivative(f, e), 2.0)
        _, Wmode = GLLVModels._bb_score_weight(0, 2.0, 10, 0.5)
        @test Wraw < GLLVModels.BB_LAPLACE_WEIGHT_FLOOR
        @test Wmode == GLLVModels.BB_LAPLACE_WEIGHT_FLOOR
    end

    @testset "verdict: weight floor at optimum is :boundary" begin
        @test GLLVModels._beta_binomial_verdict(true, 100.0, 12.0; n_weight_floor = 1) ==
              (false, -100.0, :boundary)
        @test GLLVModels._beta_binomial_verdict(true, 100.0, 12.0; n_weight_floor = 0) ==
              (true, -100.0, :ok)
    end

    @testset "seed 118 reproducer: raw-weight Laplace matches package loglik" begin
        lg(x) = 1 / (1 + exp(-x))
        rng = MersenneTwister(118)
        p, n, Ntrials, phit = 15, 150, 20, 6.0
        Lt = 0.8 .* randn(rng, p)
        bt = 0.3 .* randn(rng, p)
        Y = zeros(Int, p, n)
        for i in 1:n
            z = randn(rng)
            for t in 1:p
                μ = lg(bt[t] + Lt[t] * z)
                Y[t, i] = rand(rng, Binomial(Ntrials, rand(rng, Beta(μ * phit, (1 - μ) * phit))))
            end
        end
        Nt = fill(Ntrials, p, n)
        fit = GLLVModels.fit_beta_binomial_gllvm(Y; K = 1, N = Nt)
        raw_manual = 0.0
        nfloor = 0
        for i in 1:n
            y = Y[:, i]
            z = GLLVModels._beta_binomial_mode(y, fill(Ntrials, p), fit.Λ, fit.β, fit.φ)
            η = fit.β .+ fit.Λ * z
            W = [-ForwardDiff.derivative(e -> ForwardDiff.derivative(u -> GLLVModels.betabinomial_logp(y[t], u,
                    Ntrials, fit.φ), e), η[t]) for t in 1:p]
            nfloor += count(<(GLLVModels.BB_LAPLACE_WEIGHT_FLOOR), W)
            ll = sum(GLLVModels.betabinomial_logp(y[t], η[t], Ntrials, fit.φ) for t in 1:p)
            raw_manual += ll - 0.5 * dot(z, z) - 0.5 * logdet(I + fit.Λ' * (W .* fit.Λ))
        end
        pkg = GLLVModels.betabinomial_marginal_loglik_laplace(Y, Nt, fit.Λ, fit.β, fit.φ)
        @test nfloor > 0
        @test fit.loglik ≈ pkg atol = 1e-9
        @test fit.loglik ≈ raw_manual atol = 1e-6
        @test !fit.converged
    end
end