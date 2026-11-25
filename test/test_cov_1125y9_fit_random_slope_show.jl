# Lane cursor-cloud-1125y9: fit_random_effects.jl fit guards and show (beyond test_random_slopes.jl).
using Test, GLLVModels, Random, LinearAlgebra

const G = GLLVModels

@testset "1125y9 random slope fit guards" begin
    rng = MersenneTwister(1125094)
    p, n = 3, 12
    Y = randn(rng, p, n)
    grouping = repeat(1:4; inner = 3)
    Z = hcat(ones(n), randn(rng, n))

    @testset "dimension and K guards" begin
        @test_throws DimensionMismatch fit_gaussian_random_slope(Y, grouping[1:(n - 1)], Z; K = 1)
        @test_throws ArgumentError fit_gaussian_random_slope(Y, grouping, Z; K = 0)
    end

    @testset "GaussianRandomSlopeFit show" begin
        fit = fit_gaussian_random_slope(Y, grouping, Z; K = 1, iterations = 40)
        s = sprint(show, fit)
        @test occursin("GaussianRandomSlopeFit", s)
        @test occursin("nlevels=4", s)
        @test occursin("q=2", s)
        @test issymmetric(fit.Σ_b)
        @test fit.Σ_b[1, 1] > 0
    end

    @testset "_chol_cov_npar" begin
        @test G._chol_cov_npar(1) == 1
        @test G._chol_cov_npar(3) == 6
        θ = zeros(6)
        θ[1] = log(0.5)
        Σ, L = G._unpack_chol_cov(θ, 3)
        @test issymmetric(Σ)
        @test all(x -> x > 0, eigvals(Symmetric(Σ)))
    end
end