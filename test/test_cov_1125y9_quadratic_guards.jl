# Lane cursor-cloud-1125y9: quadratic.jl kernel guards and show.
using Test, GLLVModels, Random, Distributions

const G = GLLVModels

@testset "1125y9 quadratic kernel guards" begin
    p, K, n = 4, 1, 6
    β = log.(fill(3.0, p))
    Λ = reshape(0.2 .* ones(p), p, K)
    D = reshape(-0.05 .* ones(p), p, K)
    Y = ones(Int, p, n)
    N = ones(Int, p, n)

    @testset "hessian keyword" begin
        @test_throws ArgumentError G.quadratic_loglik_site(
            Poisson(), Y[:, 1], N[:, 1], Λ, D, β, LogLink(); hessian = :bogus)
    end

    @testset "D dimension mismatch" begin
        @test_throws DimensionMismatch G.quadratic_marginal_loglik_laplace(
            Poisson(), Y, N, Λ, zeros(p, K + 1), β, LogLink())
    end

    @testset "_quadratic_mode finite" begin
        z = G._quadratic_mode(Poisson(), Y[:, 1], N[:, 1], Λ, D, β, LogLink())
        @test length(z) == K
        @test all(isfinite, z)
    end

    @testset "QuadraticFit show" begin
        fit = G.QuadraticFit(Poisson(), β, Λ, D, 1.0, LogLink(), -12.0, true, 8)
        s = sprint(show, fit)
        @test occursin("QuadraticFit", s)
        @test occursin("Poisson", s)
        snc = sprint(show, G.QuadraticFit(Poisson(), β, Λ, D, 1.0, LogLink(), -5.0, false, 0))
        @test occursin("NOT CONVERGED", snc)
    end

    @testset "_nparams length" begin
        fit = G.QuadraticFit(Poisson(), β, Λ, D, 1.0, LogLink(), -1.0, true, 1)
        @test G._nparams(fit) == p + G.rr_theta_len(p, K) + p * K + 1
    end
end