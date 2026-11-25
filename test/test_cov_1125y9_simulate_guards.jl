# Lane cursor-cloud-1125y9: simulate_fit.jl argument guards on binomial/poisson routes.
using Test, GLLVModels, Random, Distributions

const G = GLLVModels

@testset "1125y9 simulate guards" begin
    rng = MersenneTwister(1125093)
    p, n = 3, 10
    Y = [rand(Binomial(1, 0.4)) for _ in 1:p, _ in 1:n]
    fit = fit_binomial_gllvm(Y; K = 1, iterations = 15)

    @testset "Laplace binomial refuses explicit offset" begin
        @test_throws ArgumentError simulate(fit, 5; offset = zeros(p, 5))
    end

    @testset "binomial X_lv requirement" begin
        fit_lv = G.BinomialFit(
            fit.β, fit.Λ, fit.link, fit.loglik, fit.converged, fit.iterations,
            fill(0.1, 2, 1), fit.theta_packed, fit.hessian, fit.saturation,
            fit.integration, fit.loading_ridge, fit.offset,
        )
        @test_throws ArgumentError simulate(fit_lv, 4)
        X_lv = randn(rng, 4, size(fit_lv.alpha_lv, 1))
        Ys = simulate(fit_lv, 4; X_lv = X_lv, rng = rng)
        @test size(Ys) == (p, 4)
        @test all(x -> x >= 0, Ys)
    end

    @testset "poisson scalar-μ simulate shape" begin
        Yp = [rand(Poisson(2)) for _ in 1:p, _ in 1:n]
        fp = fit_poisson_gllvm(Yp; K = 1, iterations = 12)
        S = simulate(fp, 7; rng = MersenneTwister(9))
        @test size(S) == (p, 7)
        @test eltype(S) == Int
    end

    @testset "simulate trials matrix" begin
        N = fill(2, p, 5)
        S = simulate(fit, 5; N = N, rng = MersenneTwister(11))
        @test all(S .<= N)
    end
end