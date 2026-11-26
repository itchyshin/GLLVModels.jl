# Zero-mean Gaussian contract: document + warn, no silent centring (refs #1419, #1434).
# Masked Pearson NaN by design (refs #1415).
using Test, GLLVModels, Random, Statistics, LinearAlgebra, Distributions

@testset "zero-mean Gaussian contract (#1419, #1434)" begin
    reset! = GLLVModels._reset_unidentified_warnings!

    @testset "large trait means warn on direct zero-mean route" begin
        reset!()
        rng = MersenneTwister(112624)
        p, n = 8, 50
        Y = randn(rng, p, n) .+ 1000.0
        @test_logs (:warn, r"zero-mean") (:warn, r"fit_gllvm\(Y; family = Normal") match_mode = :any min_level = Base.CoreLogging.Warn begin
            fit_gaussian_gllvm(Y; K = 2, iterations = 50)
        end
    end

    @testset "centred data stays silent" begin
        reset!()
        rng = MersenneTwister(112624)
        p, n = 8, 50
        Y = randn(rng, p, n)
        @test_logs min_level = Base.CoreLogging.Warn fit_gaussian_gllvm(Y; K = 2, iterations = 50)
    end

    @testset "explicit X suppresses large-mean warning" begin
        reset!()
        rng = MersenneTwister(112624)
        p, n = 8, 50
        Y = randn(rng, p, n) .+ 1000.0
        X = GLLVModels._trait_intercept_design(p, n)
        @test_logs min_level = Base.CoreLogging.Warn fit_gaussian_gllvm(Y; K = 2, X = X, iterations = 50)
    end

    @testset "dof gap vs fit_gllvm Normal() is 2p at matched logLik (issue repro)" begin
        Random.seed!(1126)
        p, n = 5, 80
        Λt = 0.7 .* randn(p, 3)
        K = 2
        y = Λt[:, 1:K] * randn(K, n) .+ 0.4 .* randn(p, n)
        f0 = fit_gaussian_gllvm(y; K = K, iterations = 120)
        g0 = fit_gllvm(y; family = Normal(), K = K, iterations = 120)
        k_r = p + p * K - div(K * (K - 1), 2) + 1
        @test GLLVModels.dof(f0) == p * K - div(K * (K - 1), 2) + 1
        @test GLLVModels.dof(g0) == k_r
        @test 2 * (GLLVModels.dof(g0) - GLLVModels.dof(f0)) == 2p
    end
end

@testset "masked Gaussian Pearson NaN by design (#1415)" begin
    Random.seed!(42611)
    Y = Matrix{Union{Missing, Float64}}(randn(4, 8))
    Y[1, 2] = missing
    fit = fit_gaussian_gllvm(Y; K = 1, iterations = 100)
    f = fitted(fit, Y)
    p = predict(fit, Y)
    r = residuals(fit, Y, type = :pearson)
    @test isfinite(f[1, 2])
    @test isfinite(p[1, 2])
    @test isnan(r[1, 2])
end