# Lane cursor-cloud-1125y13 — src/kernel_indep.jl
using GLLVModels, Test, LinearAlgebra, Random

@testset "cov 1125y13 kernel_indep_gllvm" begin
    rng = MersenneTwister(1125131)
    p, n = 3, 9
    Y = randn(rng, p, n)
    groups = repeat(1:3; inner = 3)
    L = randn(rng, 3, 3)
    K = Symmetric(L * L' + 0.4I)

    @testset "tiny Gaussian fit" begin
        fit = fit_kernel_indep_gllvm(Y, Matrix(K), groups; name = :patch, iterations = 60, g_tol = 1e-4)
        @test fit isa GaussianSourcesFit
        @test isfinite(fit.loglik)
        @test length(fit.sources) == 1
        @test fit.sources[1].mode == :indep
    end

    @testset "common trait variances" begin
        fit = fit_kernel_indep_gllvm(Y, Matrix(K), groups; common = true, iterations = 40)
        @test fit.sources[1].common
        @test isfinite(fit.loglik)
    end

    @testset "validation refusals" begin
        @test_throws ArgumentError fit_kernel_indep_gllvm(Y, Matrix(K))
        Kbad = K[1:2, :]
        @test_throws ArgumentError fit_kernel_indep_gllvm(Y, Kbad, groups)
        @test_throws ArgumentError fit_kernel_indep_gllvm(Y, Matrix(K), 1:(n + 1))
        @test_throws ArgumentError fit_kernel_indep_gllvm(Y, Matrix(K), fill(0, n))
        @test_throws ArgumentError fit_kernel_indep_gllvm(Y, Matrix(K), groups; rho = 0.5)
        @test_throws ArgumentError fit_kernel_indep_gllvm(Y, Matrix(K), groups; K = 2)
        @test_throws ArgumentError fit_kernel_indep_gllvm(Y, Matrix(K), groups; num_lv = 1)
        @test_throws ArgumentError fit_kernel_indep_gllvm(Y, Matrix(K), groups; sources = [])
    end
end