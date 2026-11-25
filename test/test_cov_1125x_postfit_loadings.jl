# Lane 1125x round 3: postfit.jl _loadings and diagnostics n_sim guard.
using Test, GLLVModels, Random, Distributions

const G = GLLVModels

@testset "1125x postfit _loadings and light diagnostics guard" begin
    rng = MersenneTwister(112507)
    p, n, K = 4, 30, 1
    Λ = reshape([0.6, 0.4, -0.3, 0.2], p, K)
    Z = randn(rng, K, n)
    η = Λ * Z .+ 0.1
    Yp = [Float64(rand(rng, Poisson(exp(η[t, s])))) for t in 1:p, s in 1:n]

    fit_p = fit_poisson_gllvm(Yp; K = K, iterations = 80)
    @test G._loadings(fit_p) === fit_p.Λ

    Yg = Λ * randn(rng, K, n) .+ 0.2 .* randn(rng, p, n)
    fit_g = fit_gaussian_gllvm(Yg; K = K)
    @test G._loadings(fit_g) === fit_g.pars.Λ

    Yb = Yp .> median(Yp)
    fit_b = fit_binomial_gllvm(Yb; K = K, iterations = 80)
    @test G._loadings(fit_b) === fit_b.Λ

    @test_throws ArgumentError G._loadings(fit_g.model)

    ysmall = Yg[:, 1:10]
    fit_gs = fit_gaussian_gllvm(ysmall; K = K)
    @test_throws ArgumentError G.gllvmTMB_check_consistency(fit_gs, ysmall; n_sim = 1)
end