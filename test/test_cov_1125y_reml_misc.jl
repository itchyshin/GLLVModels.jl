# Coverage round 4: reml.jl Woodbury tier helpers and GaussianREMLFit show.
using Test
using GLLVModels
using Random

const G = GLLVModels

@testset "1125y reml misc" begin
    Random.seed!(1125)
    p, n, q, K = 4, 20, 2, 1
    Y = randn(p, n)
    X = randn(p, n, q)
    Λ = 0.4 .* randn(p, K)
    σ = 0.5
    σ²_B = fill(0.1, p)
    σ²_W = fill(0.05, p)
    Λ_W = 0.2 .* randn(p, 1)

    @testset "_gaussian_gls with extra tiers" begin
        β0, _ = G._gaussian_gls(Y, X, Λ, σ)
        β1, ld1 = G._gaussian_gls(Y, X, Λ, σ; σ²_B = σ²_B)
        β2, ld2 = G._gaussian_gls(Y, X, hcat(Λ, Λ_W), σ; σ²_B = σ²_B, σ²_W = σ²_W)
        @test all(isfinite, β1)
        @test all(isfinite, β2)
        @test isfinite(ld1)
        @test isfinite(ld2)
        ll0 = G.gaussian_reml_loglik(Y, X, Λ, σ)
        ll1 = G.gaussian_reml_loglik(Y, X, Λ, σ; σ²_B = σ²_B)
        ll2 = G.gaussian_reml_loglik(Y, X, hcat(Λ, Λ_W), σ; σ²_B = σ²_B, σ²_W = σ²_W)
        @test all(isfinite, (ll0, ll1, ll2))
    end

    @testset "GaussianREMLFit show and mask refusal" begin
        reml = fit_gaussian_reml(Y, X; K = 1, iterations = 25)
        s = sprint(show, reml)
        @test occursin("GaussianREMLFit", s)
        @test occursin("σ_eps", s)
        @test occursin("reml_loglik", s)
        @test_throws ArgumentError fit_gaussian_reml(Y, X; K = 1, mask = trues(p, n))
        @test_throws ArgumentError G._refuse_missing_gaussian_reml("fit_gaussian_reml", fill(missing, p, n))
    end
end