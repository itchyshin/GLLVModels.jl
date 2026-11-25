# Lane cursor-cloud-1125y12 — families/censored_poisson.jl hooks
using GLLVModels, Test, Distributions, Random

const _CP = GLLVModels

@testset "cov 1125y12 censored Poisson Laplace hooks" begin
    fam = CensoredPoisson()

    @testset "_censored_poisson_logS and G" begin
        μ = 2.5
        C = 3
        logS = _CP._censored_poisson_logS(μ, C)
        @test logS ≈ log(1 - cdf(Poisson(μ), C - 1)) rtol = 1e-10
        G = _CP._censored_poisson_G(μ, C)
        @test isfinite(G) && G > 0
        @test_throws ArgumentError _CP._censored_poisson_logS(μ, 0)
    end

    @testset "_glm_logpdf score weight uncensored" begin
        μ = 1.8
        @test _CP._glm_logpdf(fam, μ, 0, 2) ≈ logpdf(Poisson(μ), 2) rtol = 1e-12
        sc = _CP._glm_logpdf(fam, μ, 3, 0) # censored at C=3
        @test isfinite(sc)
        s = _CP._glm_score(fam, μ, 0, μ, 2)
        @test s ≈ (2 - μ) rtol = 1e-12
        w = _CP._glm_weight(fam, μ, 0, μ)
        @test w ≈ μ rtol = 1e-12
    end

    @testset "_glm_score weight censored branch" begin
        μ = 3.0
        C = 2
        s = _CP._glm_score(fam, μ, C, μ, 99)
        G = _CP._censored_poisson_G(μ, C)
        @test s ≈ G rtol = 1e-12
        w = _CP._glm_weight(fam, μ, C, μ)
        @test w >= 0
        @test w ≈ G * (G + μ - C) rtol = 1e-10
    end

    @testset "censored_bounds_to_YN" begin
        lower = [0 2 5; 1 3 4]
        upper = [0 2 typemax(Int); 1 3 typemax(Int)]
        Y, N = _CP.censored_bounds_to_YN(lower, upper)
        @test Y == lower
        @test N == [0 0 5; 0 0 4]
        @test_throws ArgumentError _CP.censored_bounds_to_YN([1 2], [3 4; 5 6])
        @test_throws ArgumentError _CP.censored_bounds_to_YN([-1 0], [0 0])
    end

    @testset "fit_censored_poisson_gllvm tiny" begin
        Y = rand(Poisson(1.5), 3, 10)
        lower = copy(Y)
        upper = copy(Y)
        # one right-censored column (C = 2) alongside uncensored cells
        lower[:, 10] .= 2
        upper[:, 10] .= typemax(Int)
        fit = fit_censored_poisson_gllvm(Y; K = 1, lower = lower, upper = upper,
                                         iterations = 80, g_tol = 1e-4)
        @test fit isa CensoredPoissonFit
        @test isfinite(fit.loglik)
        shown = sprint(show, fit)
        @test occursin("CensoredPoisson", shown)
    end
end