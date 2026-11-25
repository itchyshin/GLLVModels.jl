# Lane cursor-cloud-1125y12 — coevolution_kronecker.jl, coevolution_blockna.jl
using GLLVModels, Test, LinearAlgebra, Random, Statistics

const _CEV = GLLVModels

@testset "cov 1125y12 coevolution Kronecker and block-NA" begin
    rng = MersenneTwister(112512)

    function _tiny_kstar(n_H, n_P)
        A_H = Matrix{Float64}(I, n_H, n_H)
        A_P = Matrix{Float64}(I, n_P, n_P)
        W = fill(1.0 / sqrt(n_H * n_P), n_H, n_P)
        return make_cross_kernel(A_H, A_P, W; rho = 0.25, eps = 1e-8)
    end

    @testset "_coevolution_kron_precompute" begin
        K = _tiny_kstar(3, 2)
        V, dv = _CEV._coevolution_kron_precompute(K)
        @test length(dv) == 5
        @test all(dv .> 0)
        @test V * Diagonal(dv) * V' ≈ K rtol = 1e-10
        bad = fill(1.0, 3, 3)
        bad[1, 1] = -0.5
        @test_throws ArgumentError _CEV._coevolution_kron_precompute(bad)
    end

    @testset "_coevolution_kron_nll finite" begin
        n_H, n_P = 3, 2
        K = _tiny_kstar(n_H, n_P)
        V, dv = _CEV._coevolution_kron_precompute(K)
        T, n = 3, n_H + n_P
        Y = 0.3 .* randn(rng, T, n)
        θ = vcat(vec(0.1 .* randn(rng, T, 1)), log(0.4))
        nll = _CEV._coevolution_kron_nll(θ, Y, V, dv, T, n, 1)
        @test isfinite(nll) && nll > 0
    end

    @testset "fit_coevolution_gaussian tiny" begin
        n_H, n_P = 3, 2
        K = _tiny_kstar(n_H, n_P)
        Y = randn(rng, 3, n_H + n_P)
        fit = fit_coevolution_gaussian(Y, K; d = 1, iterations = 120, g_tol = 1e-4)
        @test haskey(fit, :Λ) && haskey(fit, :σ) && haskey(fit, :logLik)
        @test isfinite(fit.logLik)
        @test fit.σ > 0
    end

    @testset "fit_coevolution_gaussian refusals" begin
        K = _tiny_kstar(2, 2)
        Y = randn(2, 3)
        @test_throws ArgumentError fit_coevolution_gaussian(Y, K[1:2, 1:2]; d = 1)
        @test_throws ArgumentError fit_coevolution_gaussian(Y, K; d = 0)
        @test_throws ArgumentError fit_coevolution_gaussian(Y, K; d = 10)
        Yna = Matrix{Union{Missing, Float64}}(missing, 2, 4)
        @test_throws ArgumentError fit_coevolution_gaussian(Yna, K; d = 1)
    end

    @testset "_blockna_cov structure" begin
        T_H, T_P = 2, 2
        T = T_H + T_P
        Σ_T = [1.0 0.2 0.1 0.05;
               0.2 1.0 0.15 0.1;
               0.1 0.15 1.0 0.3;
               0.05 0.1 0.3 1.0]
        A_H = Matrix{Float64}(I, 3, 3)
        A_P = Matrix{Float64}(I, 2, 2)
        K_HP = fill(0.2, 3, 2)
        M = _CEV._blockna_cov(Σ_T, A_H, A_P, K_HP, T_H, T_P)
        @test size(M) == (T_H * 3 + T_P * 2, T_H * 3 + T_P * 2)
        @test issymmetric(M)
    end

    @testset "fit_coevolution_blockna tiny" begin
        n_H, n_P = 3, 2
        A_H = Matrix{Float64}(I, n_H, n_H)
        A_P = Matrix{Float64}(I, n_P, n_P)
        K_HP = fill(0.15, n_H, n_P)
        Y_HH = randn(rng, 2, n_H)
        Y_PP = randn(rng, 2, n_P)
        fit = fit_coevolution_blockna(Y_HH, Y_PP, A_H, A_P, K_HP; d = 1, iterations = 100, g_tol = 1e-4)
        @test isfinite(fit.logLik)
        Γ = fit.Λ * fit.Λ'
        @test size(Γ) == (4, 4)
    end

    @testset "fit_coevolution_blockna refusals" begin
        Y_HH = randn(2, 3)
        Y_PP = randn(2, 2)
        A_H = Matrix{Float64}(I, 3, 3)
        A_P = Matrix{Float64}(I, 2, 2)
        K_HP = ones(3, 2)
        @test_throws ArgumentError fit_coevolution_blockna(Y_HH, Y_PP, A_H[1:2, 1:2], A_P, K_HP; d = 1)
        @test_throws ArgumentError fit_coevolution_blockna(Y_HH, Y_PP, A_H, A_P, K_HP; d = 0)
    end
end