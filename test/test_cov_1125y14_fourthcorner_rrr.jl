# Lane cursor-cloud-1125y14 — src/families/fourthcorner.jl, src/families/rrr.jl
using GLLVModels, Test, Random, Distributions

const G = GLLVModels

@testset "cov 1125y14 fourthcorner and rrr" begin
    rng = MersenneTwister(11251403)
    p, n, q, r = 4, 10, 2, 2
    Xenv = randn(rng, n, q)
    TR = randn(rng, p, r)
    C = randn(rng, q, r)

    @testset "_build_offset_fourthcorner" begin
        O = G._build_offset_fourthcorner(Xenv, TR, C)
        @test size(O) == (p, n)
        @test_throws DimensionMismatch G._build_offset_fourthcorner(Xenv, TR, zeros(1, 1))
        Cbad = zeros(3, r)
        @test_throws DimensionMismatch G._build_offset_fourthcorner(Xenv, TR, Cbad)
    end

    @testset "FourthCornerFit show and predict guards" begin
        fit = G.FourthCornerFit(
            Poisson(), zeros(p), C, randn(rng, p, 1), 1.0, LogLink(), -10.0, true, 5)
        shown = sprint(show, fit)
        @test occursin("FourthCornerFit", shown)
        Y = rand(rng, 0:5, p, n)
        @test_throws ArgumentError predict(fit, Y, Xenv, TR; type = :foo)
        η = predict(fit, Y, Xenv, TR; type = :link)
        @test size(η) == size(Y)
        @test G._nparams(fit) == p + q * r + p * 1 + 1
    end

    @testset "tiny fit_fourthcorner_gllvm" begin
        Y = rand(rng, 0:4, p, n)
        fit = fit_fourthcorner_gllvm(Y; family = Poisson(), Xenv = Xenv, TR = TR, K = 1,
            iterations = 50, g_tol = 1e-3)
        @test fit isa FourthCornerFit
        @test isfinite(fit.loglik)
        μ = fitted(fit, Y, Xenv, TR)
        @test all(μ .>= 0)
    end

    @testset "_rrr_eta and rrr_marginal_loglik" begin
        K = 1
        Λ = randn(rng, p, K)
        B = randn(rng, q, K)
        β = randn(rng, p)
        X = Xenv
        η = G._rrr_eta(β, Λ, B, X)
        @test size(η) == (p, n)
        @test_throws DimensionMismatch G._rrr_eta(β, Λ, B[1:1, :], X)
        Y = rand(rng, 0:3, p, n)
        N = ones(Int, p, n)
        ll = G.rrr_marginal_loglik(Poisson(), Y, N, Λ, B, β, X, LogLink())
        @test isfinite(ll)
        ll0 = G.rrr_marginal_loglik(Poisson(), Y, N, Λ, zeros(q, K), β, X, LogLink())
        @test isfinite(ll0)
    end

    @testset "fit_fourthcorner dimension guards" begin
        Y = rand(rng, 0:2, p, n)
        @test_throws DimensionMismatch fit_fourthcorner_gllvm(
            Y; family = Poisson(), Xenv = randn(n + 1, q), TR = TR, K = 1)
        @test_throws DimensionMismatch fit_fourthcorner_gllvm(
            Y; family = Poisson(), Xenv = Xenv, TR = randn(p + 1, r), K = 1)
    end
end