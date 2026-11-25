# Lane cursor-cloud-1125y13 — src/families/gp1.jl
using GLLVModels, Test, Random, Distributions, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y13 gp1" begin
    rng = MersenneTwister(1125134)

    @testset "_gp1_log1pmx and large-y logpdf" begin
        for x in (-0.4, -0.01, 0.0, 0.2)
            @test isfinite(G._gp1_log1pmx(x))
            @test G._gp1_log1pmx(x) ≈ log1p(x) - x rtol = x == 0 ? 0 : 1e-10
        end
        f = GeneralizedPoisson1(0.05)
        μ = 12.0
        y = 50
        direct = G._glm_logpdf(f, μ, 1, y)
        large = G._gp1_logpdf_large_y(f.α, μ, y)
        @test direct ≈ large rtol = 1e-9
        @test isfinite(direct)
    end

    @testset "score/weight Poisson limit" begin
        f0 = GeneralizedPoisson1(0.0)
        μ, y = 3.5, 4
        @test G._glm_score(f0, μ, 1, μ, y) ≈ (y - μ)
        @test G._glm_weight(f0, μ, 1, μ) ≈ μ
    end

    @testset "_gp1_cdf and _rand_gp1" begin
        f = GeneralizedPoisson1(0.02)
        μ = 4.0
        c = G._gp1_cdf(f, μ, 3)
        @test 0 ≤ c ≤ 1
        k = G._rand_gp1(rng, f, μ)
        @test k isa Integer
        @test k >= 0
    end

    @testset "_gp1_verdict" begin
        ok = G._gp1_verdict(true, 100.0)
        @test ok[1]
        @test ok[3] == :ok
        failed = G._gp1_verdict(true, 1.0e12)
        @test failed[1] == false
        @test failed[2] == -Inf
        @test failed[3] == :objective_impossible
    end

    @testset "tiny fit_gp1_gllvm" begin
        p, n = 4, 20
        Y = rand(rng, 0:8, p, n)
        fit = fit_gp1_gllvm(Y; K = 1, iterations = 80, g_tol = 1e-4)
        @test fit isa GP1Fit
        @test isfinite(fit.loglik)
        shown = sprint(show, fit)
        @test occursin("GP1Fit", shown)
    end
end