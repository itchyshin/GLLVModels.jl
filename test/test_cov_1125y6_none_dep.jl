# Lane cursor-cloud-1125y6: none_dep.jl fit_dep_gllvm guard messages and identity.
using Test, GLLVModels, Random, Distributions, LinearAlgebra

const G = GLLVModels

@testset "1125y6 none x dep" begin
    rng = MersenneTwister(112506)
    p, n = 4, 25
    Y = randn(rng, p, n)
    f = fit_dep_gllvm(Y; family = Normal())
    g = fit_gaussian_gllvm(Y; K = p)
    @test f.logLik ≈ g.logLik atol = 1e-8
    @test size(f.pars.Λ) == (p, p)
    @test f.model.K == p
    @test_throws ArgumentError fit_dep_gllvm(Y; family = Normal(), K = 2)
    @test_throws ArgumentError fit_dep_gllvm(Y; family = Normal(), num_lv = 2)
    @test_throws ArgumentError fit_dep_gllvm(Y; family = Normal(), K_W = 1)
    @test_throws ArgumentError fit_dep_gllvm(Y; family = Normal(), has_diag = true)
    @test_throws ArgumentError fit_dep_gllvm(Y; family = Normal(), K_phy = 1)
    @test_throws ArgumentError fit_dep_gllvm(Y; family = Normal(), has_phy_unique = true)
    @test_throws ArgumentError fit_dep_gllvm(Y; family = Normal(),
        Σ_phy = Matrix{Float64}(I, p, p))
    @test_throws ArgumentError fit_dep_gllvm(Y; family = Poisson())
end