# Lane cursor-cloud-1125y10: bootstrap_ci entry validation (not duplicate of test_bootstrap_decisions).
using Test, GLLVModels, LinearAlgebra, Random

const G = GLLVModels

@testset "1125y10 bootstrap_ci guards" begin
    rng = MersenneTwister(112510)
    p, K, n = 4, 1, 30
    Λ = reshape([0.8, 0.3, -0.2, 0.5], p, K)
    y = Λ * randn(rng, K, n) + 0.5 * randn(rng, p, n)
    fit = fit_gaussian_gllvm(y; K = K)

    @testset "level and n_boot" begin
        @test_throws ArgumentError bootstrap_ci(fit; y = y, level = 0.0)
        @test_throws ArgumentError bootstrap_ci(fit; y = y, level = 1.0)
        @test_throws ArgumentError bootstrap_ci(fit; y = y, level = -0.1)
        @test_throws ArgumentError bootstrap_ci(fit; y = y, n_boot = 0)
        @test_throws ArgumentError bootstrap_ci(fit; y = y, n_boot = -3)
    end

    @testset "n_sites resolution" begin
        @test_throws ArgumentError bootstrap_ci(fit)
        @test_throws ArgumentError bootstrap_ci(fit; y = y[1:(end - 1), :])
        @test_throws ArgumentError bootstrap_ci(fit; y = y, n_sites = n + 1)
        @test_throws ArgumentError bootstrap_ci(fit; y = y, n_sites = 1)
    end

    @testset "fixed effects need X" begin
        X = randn(rng, p, n, 1)
        yx = y + reshape(X[:, :, 1], p, n)
        fx = fit_gaussian_gllvm(yx; K = K, X = X)
        @test_throws ArgumentError bootstrap_ci(fx; y = yx)
        @test_throws ArgumentError bootstrap_ci(fx; y = yx, X = X[1:(end - 1), :, :])
        @test_throws ArgumentError bootstrap_ci(fx; y = yx, X = X[:, 1:(end - 1), :])
        @test_throws ArgumentError bootstrap_ci(fx; y = yx, X = X[:, 1:(end - 1), :])
    end

    @testset "phylogenetic block needs Σ_phy" begin
        Σ_bad = [1.0 0.2; 0.2 1.0]
        Σ4 = Matrix(Symmetric([1.0 0.2 0.1 0.0; 0.2 1.0 0.3 0.1;
            0.1 0.3 1.0 0.2; 0.0 0.1 0.2 1.0]))
        yp = randn(rng, p, n)
        fp = fit_gaussian_gllvm(yp; K = K, K_phy = 1, Σ_phy = Σ4)
        @test_throws ArgumentError bootstrap_ci(fp; y = yp)
        @test_throws ArgumentError bootstrap_ci(fp; y = yp, Σ_phy = Σ_bad)
    end

    @testset "lambda_constraint pins refused" begin
        M = fill(NaN, p, K)
        M[1, 1] = 0.5
        fit_p = fit_gaussian_gllvm(y; K = K, lambda_constraint = M)
        @test_throws ArgumentError bootstrap_ci(fit_p; y = y, n_boot = 2)
    end
end