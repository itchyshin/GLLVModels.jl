# Lane cursor-cloud-1125y6: gaussian_intercept.jl design tagging helpers.
using Test, GLLVModels, Random, LinearAlgebra, Statistics, Distributions

const G = GLLVModels

@testset "1125y6 gaussian intercept internals" begin
    p, n = 4, 7
    @testset "design matrices" begin
        Xt = G._trait_intercept_design(p, n)
        @test size(Xt) == (p, n, p)
        for t in 1:p, s in 1:n
            @test Xt[t, s, t] == 1.0
            @test sum(Xt[t, s, :]) == 1.0
        end
        Xc = G._common_intercept_design(p, n)
        @test size(Xc) == (p, n, 1)
        @test all(x -> x == 1.0, Xc)
    end

    @testset "fit tags and intercept mean" begin
        rng = MersenneTwister(112506)
        Y = randn(rng, p, n)
        f_trait = fit_gllvm(Y; family = Normal(), K = 1)
        @test G._has_trait_intercepts(f_trait)
        @test G._has_intercept_design(f_trait)
        @test !G._has_common_intercept(f_trait)
        @test G._intercept_mean(f_trait) == f_trait.pars.β
        β = G._intercept_mean(f_trait)
        @test length(β) == p
        phy = G.random_balanced_tree(p; branch_length = 0.4)
        Σ_phy = Matrix(Symmetric(G.sigma_phy_dense(phy; σ²_phy = 1.0)))
        f_phy = fit_gllvm(Y; family = Normal(), K = 1, has_phy_unique = true, Σ_phy = Σ_phy)
        @test G._has_common_intercept(f_phy)
        @test length(G._intercept_mean(f_phy)) == p
        @test all(x -> x == f_phy.pars.β[1], G._intercept_mean(f_phy))
        f_zero = fit_gaussian_gllvm(Y; K = 1)
        @test !G._has_intercept_design(f_zero)
    end

    @testset "_mean_X routing" begin
        rng = MersenneTwister(3)
        Y = randn(rng, p, n)
        f = fit_gllvm(Y; family = Normal(), K = 1)
        f_zero = fit_gaussian_gllvm(Y; K = 1)
        Xuser = randn(rng, p, n, 2)
        @test G._mean_X(f, nothing, n) == G._trait_intercept_design(p, n)
        @test G._mean_X(f, Xuser, n) === Xuser
        @test G._mean_X(f_zero, nothing, n) === nothing
    end

    @testset "_fit_gaussian_trait_intercepts passthrough" begin
        rng = MersenneTwister(9)
        Y = randn(rng, p, n)
        X = ones(p, n, 1)
        a = G._fit_gaussian_trait_intercepts(Y; K = 1)
        b = G._fit_gaussian_trait_intercepts(Y; K = 1, X = X)
        @test b.logLik == fit_gaussian_gllvm(Y; K = 1, X = X).logLik
        @test !haskey(b.pars, :mean_design)
        @test a.logLik ≈ fit_gllvm(Y; family = Normal(), K = 1).logLik atol = 1e-10
    end
end