# Lane cursor-cloud-1125y6: lv_targets.jl centre columns and eta slopes.
using Test, GLLVModels, LinearAlgebra, Random, Statistics

const G = GLLVModels

@testset "1125y6 lv_targets" begin
    rng = MersenneTwister(112506)
    n, q, p, K = 12, 2, 3, 1
    X_lv = hcat(collect(range(-0.5, 0.5; length = n)),
        sin.(collect(range(0, 2π; length = n))))
    Z = randn(rng, n, K)
    Λ = reshape([0.5, -0.3, 0.8], p, K)

    @testset "_center_columns" begin
        Xc = G._center_columns(X_lv)
        @test size(Xc) == size(X_lv)
        @test all(abs.(mean(Xc; dims = 1)) .< 1e-14)
        @test mean(X_lv; dims = 1) != zeros(1, q)
    end

    @testset "_eta_realized_lv_effects" begin
        B = G._eta_realized_lv_effects(X_lv, Z, Λ)
        @test size(B) == (p, q)
        @test all(isfinite, B)
        eta = Z * Λ'
        etac = G._center_columns(eta)
        Xc = G._center_columns(X_lv)
        ref = Matrix(transpose(Xc \ etac))
        @test B ≈ ref rtol = 1e-10
        @test_throws ArgumentError G._eta_realized_lv_effects(X_lv[1:(n - 1), :], Z, Λ)
        Zwide = hcat(Z, randn(rng, n))
        @test_throws ArgumentError G._eta_realized_lv_effects(X_lv, Zwide, Λ)
        rank1_X = ones(n, 1)
        @test_throws ArgumentError G._eta_realized_lv_effects(rank1_X, Z, Λ)
    end
end