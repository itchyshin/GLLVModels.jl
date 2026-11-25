# Lane cursor-cloud-1125y13 — src/ordination_uncertainty.jl
using GLLVModels, Test, Random, Statistics, Distributions

const G = GLLVModels

@testset "cov 1125y13 ordination_uncertainty" begin
    rng = MersenneTwister(1125138)

    @testset "_procrustes_align preserves row count" begin
        A = randn(rng, 6, 2)
        B = randn(rng, 6, 2)
        Al = G._procrustes_align(A, B)
        @test size(Al) == size(A)
        @test cor(vec(Al), vec(B)) > cor(vec(A), vec(B)) - 0.3
    end

    @testset "ordination_uncertainty Poisson smoke" begin
        p, n, K = 5, 20, 1
        β = log.(fill(3.0, p))
        Y = [rand(Poisson(3)) for _ in 1:p, _ in 1:n]
        fit = fit_poisson_gllvm(Y; K = K, β_init = β, iterations = 80, g_tol = 1e-4)
        u = ordination_uncertainty(fit, Y; n_boot = 12, level = 0.9, rng = rng)
        @test size(u.scores) == (n, K)
        @test size(u.se) == (n, K)
        @test all(isfinite, u.lower)
        @test all(isfinite, u.upper)
        @test mean(u.se) >= 0
        @test u.n_boot >= 2
    end

    @testset "refusals" begin
        fit = fit_poisson_gllvm(rand(1:5, 2, 10); K = 1, iterations = 40)
        @test_throws ArgumentError ordination_uncertainty(fit, rand(1:5, 2, 10); n_boot = 1)
        @test_throws ArgumentError ordination_uncertainty(fit, rand(1:5, 2, 10); level = 1.2)
    end
end