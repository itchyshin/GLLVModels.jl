# Lane cursor-cloud-1125y12 — twolevel.jl derived extractors and PSD helper
using GLLVModels, Test, LinearAlgebra, Random

const _TL2 = GLLVModels

@testset "cov 1125y12 two-level derived quantities" begin
    rng = MersenneTwister(11251220)
    p, n = 4, 16
    y = randn(rng, p, n)
    indiv = repeat(1:4; inner = 4)
    fit = fit_twolevel_gaussian(y, indiv; K_B = 1, K_W = 1, iterations = 100, g_tol = 1e-4)

    @testset "communality and correlation" begin
        cB = communality_B(fit)
        cW = communality_W(fit)
        @test length(cB) == p && length(cW) == p
        @test all(0.0 .<= cB .<= 1.0)
        @test all(0.0 .<= cW .<= 1.0)
        RB = correlation_B(fit)
        RW = correlation_W(fit)
        @test size(RB) == (p, p)
        @test diag(RB) ≈ ones(p) rtol = 0 atol = 1e-10
        @test issymmetric(RB)
        @test issymmetric(RW)
    end

    @testset "_to_correlation and _psd_sqrt_factor" begin
        Σ = fit.Σ_B
        R = _TL2._to_correlation(Σ)
        @test all(abs.(diag(R) .- 1.0) .< 1e-12)
        S = _TL2._psd_sqrt_factor(Σ)
        @test size(S) == (p, p)
        @test S * S' ≈ Σ rtol = 1e-8
    end

    @testset "repeatability_wald_ci smoke" begin
        rows = repeatability_wald_ci(fit, y, indiv; level = 0.9)
        @test length(rows) == p
        @test all(r -> isfinite(r.estimate) && r.lower <= r.estimate <= r.upper, rows)
        @test all(r -> 0.0 <= r.estimate <= 1.0, rows)
    end
end