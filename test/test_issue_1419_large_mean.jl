# Issue #1419: large-mean uncentred Y must not report a spurious optimum on the
# documented zero-mean `fit_gaussian_gllvm` path.
using Test, GLLVModels, Random, Distributions, Statistics

@testset "fit_gaussian_gllvm large-mean shift equivariance (#1419)" begin
    rng = MersenneTwister(112624)
    p, n = 8, 50
    Y = randn(rng, p, n) .+ 1000.0
    Yc = Y .- mean(Y, dims = 2)
    b = fit_gaussian_gllvm(Y; K = 2, iterations = 200)
    c = fit_gaussian_gllvm(Yc; K = 2, iterations = 200)
    g = fit_gllvm(Y; family = Normal(), K = 2, iterations = 200)
    @test b.converged
    @test isapprox(b.logLik, c.logLik; atol = 1e-6, rtol = 0)
    @test isapprox(b.logLik, g.logLik; atol = 1e-6, rtol = 0)
    @test isempty(b.pars.β)
end