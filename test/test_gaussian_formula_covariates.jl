# Gaussian `@formula(y ~ x)` has one intercept per trait and a shared slope, as
# the gllvm docstring states, as the non-Gaussian `_cov` routes and the other
# Gaussian formula routes (pervar, sources, grouping) do, and as gllvmTMB's
# `value ~ 0 + trait + x` does. Before the fix it built a site-only design with
# no intercepts, so shifting Y moved logLik.
using Test, GLLVModels, Random, Distributions, Statistics, LinearAlgebra

@testset "Gaussian formula with covariates has trait intercepts" begin
    rng = MersenneTwister(11)
    p, n, K = 5, 90, 1
    temp = randn(rng, n)
    Y0 = 0.6 .* randn(rng, p, K) * randn(rng, K, n) .+ 0.4 .* temp' .+ 0.5 .* randn(rng, p, n)
    shift = collect(1.0:p) .+ 3.0
    Y1 = Y0 .+ shift
    data = (temp = temp,)

    f0 = gllvm(@formula(y ~ temp), Y0, data; family = Normal(), K = K)
    f1 = gllvm(@formula(y ~ temp), Y1, data; family = Normal(), K = K)

    @testset "shift equivariance" begin
        @test length(f0.pars.β) == p + 1
        @test isapprox(f0.logLik, f1.logLik; atol = 1e-6)
        @test isapprox(f1.pars.β[1:p], f0.pars.β[1:p] .+ shift; atol = 1e-4)
        @test isapprox(f1.pars.β[p + 1], f0.pars.β[p + 1]; atol = 1e-5)
    end

    @testset "design is trait intercepts plus a shared slope" begin
        X = zeros(p, n, p + 1)
        for t in 1:p
            X[t, :, t] .= 1.0
        end
        for s in 1:n, t in 1:p
            X[t, s, p + 1] = temp[s]
        end
        ref = fit_gaussian_gllvm(Y1; K = K, X = X)
        @test isapprox(f1.logLik, ref.logLik; atol = 1e-8)
        @test isapprox(f1.pars.β, ref.pars.β; atol = 1e-6)
        f1b = gllvm(@formula(y ~ 1 + temp), Y1, data; family = Normal(), K = K)
        @test f1b.logLik == f1.logLik
    end

    @testset "y ~ 0 + x keeps no intercepts" begin
        Xs = zeros(p, n, 1)
        for s in 1:n, t in 1:p
            Xs[t, s, 1] = temp[s]
        end
        g = gllvm(@formula(y ~ 0 + temp), Y1, data; family = Normal(), K = K)
        @test length(g.pars.β) == 1
        @test isapprox(g.logLik, fit_gaussian_gllvm(Y1; K = K, X = Xs).logLik; atol = 1e-8)
    end
end

# With a phylogeny, per-trait intercepts would absorb the species-constant phylo
# effect, so `y ~ 1 + x` uses one common intercept plus the shared slope.
@testset "Gaussian formula with covariates and a phylogeny" begin
    Random.seed!(21)
    p, n = 6, 200
    temp = randn(n)
    phy = GLLVModels.random_balanced_tree(p; branch_length = 0.5)
    Σ_phy = Matrix(Symmetric(GLLVModels.sigma_phy_dense(phy; σ²_phy = 1.0)))
    y = 0.5 .* randn(p, 1) * randn(1, n) .+ 0.4 .* temp'
    y .+= 0.8 .* (cholesky(Symmetric(Σ_phy)).L * randn(p))
    y .+= 0.5 .* randn(p, n)
    f = gllvm(@formula(y ~ 1 + temp), y, (temp = temp,); family = Normal(), K = 1,
              has_phy_unique = true, Σ_phy = Σ_phy)
    X = zeros(p, n, 2)
    X[:, :, 1] .= 1.0
    for s in 1:n, t in 1:p
        X[t, s, 2] = temp[s]
    end
    ref = fit_gaussian_gllvm(y; K = 1, X = X, has_phy_unique = true, Σ_phy = Σ_phy)
    @test length(f.pars.β) == 2
    @test isapprox(f.logLik, ref.logLik; atol = 1e-8)
    @test maximum(abs.(f.pars.σ_phy)) > 0.1
end
