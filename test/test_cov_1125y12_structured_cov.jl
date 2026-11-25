# Lane cursor-cloud-1125y12 — structured_cov.jl (spatial_cov, relatedness_cov)
using GLLVModels, Test, LinearAlgebra, SpecialFunctions

@testset "cov 1125y12 structured covariance builders" begin
    coords = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]

    @testset "spatial_cov exponential" begin
        C = spatial_cov(coords; kernel = :exponential, range = 1.0, sill = 2.0, nugget = 1e-5)
        @test C isa Symmetric{Float64}
        @test size(C) == (4, 4)
        @test C[1, 1] ≈ 2.0 + 1e-5 atol = 1e-8
        @test C[1, 2] ≈ 2.0 * exp(-1.0) rtol = 1e-12
        @test all(eigvals(Matrix(C)) .> 0)
    end

    @testset "spatial_cov gaussian" begin
        C = spatial_cov(coords; kernel = :gaussian, range = 0.5, sill = 1.0)
        d12 = norm(coords[1, :] - coords[2, :])
        @test C[1, 2] ≈ exp(-(d12 / 0.5)^2) rtol = 1e-12
    end

    @testset "spatial_cov matern nu=0.5 matches exponential" begin
        r = 1.2
        Ce = spatial_cov(coords[1:2, :]; kernel = :exponential, range = r, sill = 1.0, nugget = 0.0)
        Cm = spatial_cov(coords[1:2, :]; kernel = :matern, range = r, smoothness = 0.5, sill = 1.0, nugget = 0.0)
        @test Ce[1, 2] ≈ Cm[1, 2] rtol = 1e-10
    end

    @testset "spatial_cov matern off-diagonal" begin
        C = spatial_cov(coords; kernel = :matern, range = 1.5, smoothness = 1.5, sill = 1.0, nugget = 1e-6)
        d = norm(coords[1, :] - coords[3, :])
        ν = 1.5
        x = sqrt(2ν) * d / 1.5
        ref = (2.0^(1.0 - ν) / gamma(ν)) * x^ν * SpecialFunctions.besselk(ν, x)
        @test C[1, 3] ≈ ref rtol = 1e-10
        @test C[3, 3] ≈ 1.0 + 1e-6
    end

    @testset "spatial_cov argument guards" begin
        @test_throws ArgumentError spatial_cov(coords; kernel = :exponential, range = 0.0)
        @test_throws ArgumentError spatial_cov(coords; kernel = :exponential, range = 1.0, sill = -1.0)
        @test_throws ArgumentError spatial_cov(coords; kernel = :exponential, range = 1.0, nugget = -1.0)
        @test_throws ArgumentError spatial_cov(coords; kernel = :matern, range = 1.0, smoothness = 0.0)
        @test_throws ArgumentError spatial_cov(coords; kernel = :unknown, range = 1.0)
    end

    @testset "relatedness_cov" begin
        A = [1.0 0.5 0.25; 0.49 1.0 0.6; 0.2 0.55 1.0]
        R = relatedness_cov(A)
        @test R isa Symmetric{Float64}
        @test size(R) == (3, 3)
        @test R[1, 2] ≈ (A[1, 2] + A[2, 1]) / 2 rtol = 0 atol = 1e-12
        @test R[1, 1] ≈ 1.0 + 1e-6
        Rj = relatedness_cov(A; jitter = 0.01)
        @test Rj[2, 2] ≈ 1.0 + 0.01
        @test_throws ArgumentError relatedness_cov(randn(2, 3))
        @test_throws ArgumentError relatedness_cov(A; jitter = -1.0)
    end
end