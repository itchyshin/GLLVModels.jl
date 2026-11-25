# Lane cursor-cloud-1125y11 — coevolution_glm.jl guards and tiny marginal
using GLLVModels, Test, LinearAlgebra, Distributions

@testset "cov 1125y11 coevolution GLM guards" begin
    K = [1.0 0.3; 0.3 1.0]
    Y = [1.0 2.0; 3.0 4.0]
    N = ones(2, 2)

    @testset "marginal at fixed point is finite" begin
        β = [0.1, 0.2]
        Λ = [0.3 0.0; 0.0 0.3]
        ll = coevolution_glm_marginal_loglik(Poisson(), Y, N, β, Λ, 1.0, K;
            maxiter = 20, tol = 1e-6)
        @test isfinite(ll)
    end

    @testset "fit_coevolution_glm tiny" begin
        fit = fit_coevolution_glm(Y, K; d = 1, n_host_traits = 1,
            iterations = 40, g_tol = 1e-3)
        @test fit isa CoevolutionGLMFit
        @test length(fit.β) == 2
        @test size(fit.Λ) == (2, 1)
        @test isfinite(fit.loglik)
        Γ = coevolution_gamma(fit)
        @test size(Γ) == (1, 1)
    end

    @testset "dimension errors" begin
        @test_throws ArgumentError fit_coevolution_glm(Y, K[1:1, :]; d = 1)
        @test_throws ArgumentError fit_coevolution_glm(Y, K; d = 0)
        @test_throws ArgumentError fit_coevolution_glm(Y, K; d = 3)
        @test_throws ArgumentError coevolution_glm_marginal_loglik(
            Poisson(), Y, ones(1, 2), zeros(2), zeros(2, 1), 1.0, K)
    end

    @testset "block-NA mask path" begin
        Ym = [1.0 missing; missing 2.0]
        mask = BitMatrix([true false; false true])
        ll = coevolution_glm_marginal_loglik(Poisson(), Ym, N, zeros(2), 0.1 .* ones(2, 1),
            1.0, K; mask = mask, maxiter = 15)
        @test isfinite(ll)
    end
end