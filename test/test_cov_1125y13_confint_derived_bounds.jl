# Lane cursor-cloud-1125y13 — src/confint_derived.jl bound post-processing (not 1125y9 percentile)
using GLLVModels, Test, Random, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y13 confint_derived bounds" begin

    @testset "_derived_check_bounds" begin
        lo, hi = G._derived_check_bounds((0.1, 0.9), 0.5)
        @test lo == 0.1 && hi == 0.9
        @test_throws ArgumentError G._derived_check_bounds((0.9, 0.1), 0.5)
        @test_throws ArgumentError G._derived_check_bounds((0.0, 1.0), 1.5)
    end

    @testset "_derived_bound_side" begin
        D(c) = (c - 0.5)^2
        cutoff = 0.25
        b, at = G._derived_bound_side(D, NaN, 0.0, cutoff, true)
        @test b == 0.0
        @test at
        b2, at2 = G._derived_bound_side(D, 0.35, 0.0, cutoff, true)
        @test b2 == 0.35
        @test !at2
        b3, at3 = G._derived_bound_side(D, NaN, 1.0, cutoff, false)
        @test b3 == 1.0
        @test at3
    end

    @testset "communality and correlation accessors" begin
        rng = MersenneTwister(1125137)
        Y = randn(rng, 4, 18)
        fit = fit_gaussian_gllvm(Y; K = 1, iterations = 25)
        c = G.communality(fit)
        @test length(c) == 4
        @test all(0 .<= c .<= 1)
        ρ = G.correlation(fit)
        @test issymmetric(ρ)
        @test all(diag(ρ) .≈ 1.0)
        σ = G.sigma_y_site(fit)
        @test size(σ) == (4, 4)
        props = G.proportions(fit; component = :shared)
        @test length(props) == 4
        @test all(>(0), props)
    end
end