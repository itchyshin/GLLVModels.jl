# Lane cursor-cloud-1125y15 — truncated_nbinom2, negbin1, variational_exponential
using GLLVModels, Test, Random, Distributions, LinearAlgebra, ForwardDiff

const G = GLLVModels

@testset "cov 1125y15 truncated NB, NB1, VA exponential" begin
    @testset "fit_truncated_nbinom2 hessian guard" begin
        Y = rand(Poisson(1.5), 3, 8) .+ 1
        @test_throws ArgumentError fit_truncated_nbinom2_gllvm(Y; K = 1, hessian = :wrong)
    end

    @testset "truncated per-trait r length guard" begin
        Y = rand(NegativeBinomial(2.0, 0.4), 4, 6) .+ 1
        @test_throws ArgumentError fit_truncated_nbinom2_gllvm_pertrait(Y; K = 1,
            r_init = fill(1.0, 3), iterations = 0)
    end

    @testset "fit_nb1_gllvm hessian" begin
        Y = rand(NegativeBinomial(3.0, 0.5), 3, 10)
        @test_throws ArgumentError fit_nb1_gllvm(Y; K = 1, hessian = :wrong)
        fit = fit_nb1_gllvm(Y; K = 1, iterations = 50, g_tol = 1e-3)
        @test fit isa NB1Fit
        @test isfinite(fit.loglik)
    end

    @testset "VA exponential ELBO at prior" begin
        p, K = 3, 2
        Λ = 0.05 .* randn(p, K)
        Λ2 = Λ .^ 2
        β = zeros(p)
        y = abs.(randn(p)) .+ 0.2
        ψ0 = vcat(zeros(K), zeros(K)) # m=0, logv=0 ⇒ v=1
        elbo0 = G._va_site_exponential_elbo(ψ0, y, Λ, Λ2, β)
        indep = sum(-β .- y)
        @test isfinite(elbo0) && elbo0 < 0
        @test abs(elbo0 - indep) < 0.05
    end

    @testset "VA exponential gradient FD" begin
        Λ = [0.2 0.05; -0.1 0.15; 0.0 -0.05]
        Λ2 = Λ .^ 2
        β = [0.1, -0.05, 0.0]
        y = [1.1, 0.7, 1.8]
        ψ = [0.05, -0.02, log(0.9), log(1.1)]
        Gvec = zeros(length(ψ))
        G._va_site_exponential_grad!(Gvec, ψ, y, Λ, Λ2, β)
        f(ψv) = -G._va_site_exponential_elbo(ψv, y, Λ, Λ2, β)
        fd = ForwardDiff.gradient(f, ψ)
        @test Gvec ≈ fd rtol = 1e-5 atol = 1e-6
    end

    @testset "VA site profile optimiser" begin
        Λl = [0.2 0.05; -0.1 0.15; 0.0 -0.05]
        Λ2l = Λl .^ 2
        βl = [0.1, -0.05, 0.0]
        yv = [1.1, 0.7, 1.8]
        val = G._va_site_exponential(yv, Λl, Λ2l, βl; maxiter = 40, tol = 1e-6)
        @test isfinite(val)
        ψ0 = zeros(4)
        ψ1 = [0.05, -0.02, log(0.9), log(1.1)]
        val1 = G._va_site_exponential_elbo(ψ1, yv, Λl, Λ2l, βl; obs = trues(3))
        @test val1 <= G._va_site_exponential_elbo(ψ0, yv, Λl, Λ2l, βl) + 1e-6
        val_masked = G._va_site_exponential_elbo(ψ0, yv, Λl, Λ2l, βl;
            obs = [true, false, true])
        @test isfinite(val_masked)
    end

    @testset "tiny truncated Poisson vs main tests" begin
        Y = [1 2 3; 2 3 4]
        fit = fit_truncated_poisson_gllvm(Y; K = 1, iterations = 40, g_tol = 1e-3)
        @test isfinite(fit.loglik)
        @test fit isa TruncatedPoissonFit
        shown = sprint(show, fit)
        @test occursin("TruncatedPoisson", shown)
    end
end