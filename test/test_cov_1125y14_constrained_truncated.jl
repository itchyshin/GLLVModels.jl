# Lane cursor-cloud-1125y14 — src/families/constrained_ordination.jl, src/families/truncated_poisson.jl
using GLLVModels, Test, Random, Distributions

const G = GLLVModels

@testset "cov 1125y14 constrained ordination and truncated Poisson" begin
    rng = MersenneTwister(11251404)
    p, n, q, K = 4, 12, 2, 1
    X = randn(rng, n, q)
    Λ = 0.1 .* randn(rng, p, K)
    B = 0.05 .* randn(rng, q, K)
    β = randn(rng, p) .* 0.1

    @testset "_build_offset_constrained" begin
        O = G._build_offset_constrained(Λ, B, X)
        @test size(O) == (p, n)
        @test_throws DimensionMismatch G._build_offset_constrained(Λ, B[1:1, :], X)
        Xbad = randn(n, q + 1)
        @test_throws DimensionMismatch G._build_offset_constrained(Λ, B, Xbad)
        O0 = G._build_offset_constrained(Λ, zeros(q, K), X)
        @test all(O0 .== 0)
    end

    @testset "constrained_marginal_loglik_laplace" begin
        Y = rand(rng, 1:6, p, n)
        N = ones(Int, p, n)
        ll = G.constrained_marginal_loglik_laplace(
            Poisson(), Y, N, Λ, β, B, X, LogLink())
        @test isfinite(ll)
        ll0 = G.constrained_marginal_loglik_laplace(
            Poisson(), Y, N, Λ, β, zeros(q, K), X, LogLink())
        @test isfinite(ll0)
    end

    @testset "ConstrainedOrdinationFit show" begin
        fit = G.ConstrainedOrdinationFit(
            Poisson(), β, Λ, B, NaN, LogLink(), -5.0, true, 3)
        shown = sprint(show, fit)
        @test occursin("ConstrainedOrdinationFit", shown)
    end

    @testset "truncated Poisson helpers" begin
        μ = 2.5
        μtr, vartr = G._truncpois_mean_var(μ)
        @test μtr > μ
        @test vartr > 0
        s = G._glm_score(G.TruncatedPoisson(), μ, 1, μ, 3)
        w = G._glm_weight(G.TruncatedPoisson(), μ, 1, μ)
        @test isfinite(s)
        @test w > 0
        @test G._glm_logpdf(G.TruncatedPoisson(), μ, 1, 0) == -Inf
        @test isfinite(G._glm_logpdf(G.TruncatedPoisson(), μ, 1, 2))
        @test G._laplace_mode_should_backtrack(G.TruncatedPoisson())
    end

    @testset "tiny fit_truncated_poisson_gllvm" begin
        Y = rand(rng, 1:5, p, n)
        fit = fit_truncated_poisson_gllvm(Y; K = 1, iterations = 40, g_tol = 1e-3)
        @test fit isa TruncatedPoissonFit
        @test isfinite(fit.loglik)
        shown = sprint(show, fit)
        @test occursin("TruncatedPoissonFit", shown)
    end

    @testset "truncated fit y<1 guard" begin
        Y = fill(1, 2, 3)
        Y[1, 1] = 0
        @test_throws ArgumentError fit_truncated_poisson_gllvm(Y; K = 1)
    end
end