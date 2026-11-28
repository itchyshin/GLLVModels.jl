# Poisson Laplace weights with K ≥ 1 are not frequency weights (#1181).

using GLLVModels, Random, Test
using Distributions: Poisson as DPoisson

@testset "Poisson weights with latent term (#1181)" begin
    rng = MersenneTwister(101)
    p, n = 6, 40
    η = 0.8 .+ 0.3 .* randn(rng, p) .+ 0.6 .* randn(rng, p, 1) * randn(rng, 1, n)
    Y = [rand(rng, DPoisson(exp(η[i, j]))) for i in 1:p, j in 1:n]
    w = Float64.(rand(rng, 1:3, n))
    Ydup = Y[:, vcat([fill(s, Int(w[s])) for s in 1:n]...)]
    f0 = fit_gllvm(Y; family = GLLVModels.Poisson(), K = 0)
    f1 = fit_gllvm(Y; family = GLLVModels.Poisson(), K = 1)
    fw = fit_gllvm(Y; family = GLLVModels.Poisson(), K = 1, weights = w)
    f2 = fit_gllvm(Y; family = GLLVModels.Poisson(), K = 1, weights = 2.0)
    fd = fit_gllvm(Ydup; family = GLLVModels.Poisson(), K = 0)
    f0w = fit_gllvm(Y; family = GLLVModels.Poisson(), K = 0, weights = w)
    @test f0.converged
    @test f1.converged
    @test isfinite(fw.loglik)
    @test abs(fd.loglik - f0w.loglik) < 1e-4
    @test abs(f2.loglik - f1.loglik) > 1e-3
    @test abs(2 * f1.loglik - f2.loglik) > 1e-3
end