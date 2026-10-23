using GLLVModels, Test, LinearAlgebra, Random, SparseArrays

@testset "symmetric admission tolerates rounding asymmetry (#722)" begin
    Random.seed!(1)
    A = randn(2, 2)
    K0 = A * A' + I
    K = inv(inv(K0))
    @test !issymmetric(K)
    @test isapprox(K, K'; rtol = 0, atol = 1e-14)

    Y = randn(2, 2)
    beta = [0.1, -0.2]
    lambda = [0.4; -0.3;;]
    groups = reshape([1, 2], 2, 1)
    sigma = 0.7
    val = GLLVModels._gaussian_source_loglik(Y, beta, lambda, [K], groups, sigma)
    @test isfinite(val)

    P = [1.0 0.0; 0.0 1.0]
    sc = GLLVModels.SourceCovariance(K, P; mode = :latent, rank = 1)
    @test sc.covariance ≈ Symmetric(K)
end
