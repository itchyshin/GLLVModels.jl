# Coverage round 4: link_residual.jl branches beyond test_link_residual_undefined_link.jl.
using Test
using GLLVModels
using Distributions
using SpecialFunctions

const G = GLLVModels

@testset "1125y link_residual misc" begin
    @testset "binomial link constants" begin
        @test G._binomial_link_residual(G.CLogLogLink()) ≈ π^2 / 6
        @test_throws ArgumentError G._binomial_link_residual(G.LogLink())
    end

    @testset "_link_residual_one specials" begin
        @test G._link_residual_one(Normal(), G.IdentityLink(), 0.0, 1.0) == 0.0
        @test G._link_residual_one(Poisson(), G.LogLink(), 0.0, nothing) == 0.0
        @test G._link_residual_one(Poisson(), G.LogLink(), 2.0, nothing) ≈ log1p(0.5)
        @test G._link_residual_one(Gamma(), G.LogLink(), 1.0, 4.0) ≈ trigamma(4.0)
        @test G._link_residual_one(NegativeBinomial(), G.LogLink(), 1.0, 2.0) ≈ trigamma(2.0)
        @test G._link_residual_one(G.Ordinal(), G.LogitLink(), 0.0, nothing) ≈ π^2 / 3
        st = G.StudentTFamily(5.0, 0.7)
        @test G._link_residual_one(st, G.IdentityLink(), 0.0, 0.7) ≈ 0.7^2 * 5 / 3
        @test G._link_residual_one(G.StudentTFamily(2.0, 1.0), G.IdentityLink(), 0.0, 1.0) == Inf
        @test G._link_residual_one(Beta(1.0, 1.0), G.LogitLink(), 0.5, 10.0) ≈
            0.4426459114742304 rtol = 1e-9
    end

    @testset "_masked_trait_mean" begin
        μ = [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test G._masked_trait_mean(μ, nothing) ≈ [2.0, 5.0]
        mask = [true false true; true true false]
        @test G._masked_trait_mean(μ, mask) ≈ [2.0, 4.5]
        mask2 = falses(2, 3)
        @test G._masked_trait_mean(μ, mask2) == [0.0, 0.0]
    end

    @testset "link_residual on small Poisson fit" begin
        Y = [1 0 2 1 0 2 1 1 0 2 1 0 1 2 0
             2 1 0 2 1 0 1 2 0 1 2 1 0 1 2]
        fit = fit_poisson_gllvm(Y; K = 1, iterations = 12)
        σ2 = link_residual(fit, Y)
        @test length(σ2) == size(Y, 1)
        @test all(x -> x > 0, σ2)
        mask = trues(size(Y))
        mask[1, 1] = false
        σ2m = link_residual(fit, Y; mask = mask)
        @test σ2m ≈ σ2 rtol = 0.05
    end
end