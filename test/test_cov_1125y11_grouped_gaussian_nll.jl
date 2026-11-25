# Lane cursor-cloud-1125y11 — grouped_gaussian.jl internal NLL kernels
using GLLVModels, Test, LinearAlgebra, SparseArrays

@testset "cov 1125y11 grouped Gaussian factor NLL" begin
    p, n = 2, 5
    Y = [1.0 2.0 0.5 1.5 2.5; 0.5 1.0 1.5 2.0 0.8]
    beta = [0.1, -0.2]
    Z = sparse(collect(1:n), collect(1:n), ones(n), n, n)
    L = [0.4 0.0; 0.1 0.3]

    @testset "zero sources reduces to iid residual" begin
        nll0 = GLLVModels._grouped_gaussian_factor_nll(Y, beta, SparseMatrixCSC{Float64,Int}[],
            Matrix{Float64}[], 0.7)
        res = vec(Y .- beta)
        expected = (p * n * log(2π) + p * n * log(0.49) + dot(res, res) / 0.49) / 2
        @test nll0 ≈ expected rtol = 1e-10
    end

    @testset "one latent source" begin
        nll1 = GLLVModels._grouped_gaussian_factor_nll(Y, beta, [Z], [L], 0.6)
        @test isfinite(nll1)
        @test nll1 > 0
        B = L * L' + 0.01 * I
        nll_compat = GLLVModels._grouped_gaussian_nll(Y, beta, [Z], [B], 0.6)
        @test isfinite(nll_compat)
        @test nll_compat > 0
    end

    @testset "unique variance augments loadings" begin
        uniques = [[0.0, 0.09]]
        nll_u = GLLVModels._grouped_gaussian_factor_nll(Y, beta, [Z], [L], 0.6; uniques = uniques)
        @test isfinite(nll_u)
        @test nll_u ≠ GLLVModels._grouped_gaussian_factor_nll(Y, beta, [Z], [L], 0.6)
    end

    @testset "validation errors" begin
        @test_throws ArgumentError GLLVModels._grouped_gaussian_factor_nll(Y, beta, [Z], [L], -1.0)
        @test_throws ArgumentError GLLVModels._grouped_gaussian_factor_nll(
            Float32.(Y), beta, [Z], [L], 0.5)
        @test_throws DimensionMismatch GLLVModels._grouped_gaussian_factor_nll(
            Y, [0.0], [Z], [L], 0.5)
        @test_throws DimensionMismatch GLLVModels._grouped_gaussian_nll(
            Y, beta, [Z], [ones(3, 3)], 0.5)
        Bbad = [1.0 1.0; 1.0 1.0]
        @test_throws ArgumentError GLLVModels._grouped_gaussian_nll(Y, beta, [Z], [Bbad], 0.5)
    end
end