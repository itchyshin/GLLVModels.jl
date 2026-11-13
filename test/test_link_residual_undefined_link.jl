# A fit whose (family, link) pair has no link-implicit residual variance (Beta with probit
# or cloglog, Poisson with identity) fits and converges, but `communality`, `sigma_y_site`
# and `link_residual` used to stop with `MethodError: no method matching
# _link_residual_one(...)`, naming an internal function. They now throw an ArgumentError
# that names the family and the link, as the Binomial path already does.
using GLLVModels, Test, Random, Distributions
const GM = GLLVModels

@testset "link_residual: undefined (family, link) pair gives a clear ArgumentError" begin
    # direct calls
    @test_throws ArgumentError GM.link_residual(Beta(1.0, 1.0), GM.ProbitLink(), 0.5, 10.0)
    @test_throws ArgumentError GM.link_residual(Beta(1.0, 1.0), GM.CLogLogLink(), 0.5, 10.0)
    @test_throws ArgumentError GM.link_residual(Poisson(), GM.IdentityLink(), 2.0, nothing)
    err = try GM.link_residual(Beta(1.0, 1.0), GM.ProbitLink(), 0.5, 10.0); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("Beta", err.msg) && occursin("ProbitLink", err.msg)
    # defined pairs keep their values
    @test GM.link_residual(Binomial(), GM.ProbitLink(), 0.5, nothing) == 1.0
    @test GM.link_residual(Binomial(), GM.LogitLink(), 0.5, nothing) ≈ π^2 / 3
    @test GM.link_residual(Poisson(), GM.LogLink(), 2.0, nothing) ≈ log1p(1 / 2.0)
    @test GM.link_residual(Beta(1.0, 1.0), GM.LogitLink(), 0.5, 10.0) ≈ 0.4426459114742304   # 2 trigamma(5)
    @test GM.link_residual(GM.Ordinal(), GM.ProbitLink(), 0.0, nothing) == 1.0

    # through a fitted model
    rng = MersenneTwister(4); p, n = 4, 40
    z = randn(rng, n); L = [0.9, 0.6, -0.7, 0.5]
    Φ(x) = cdf(Normal(), x)
    Yb = [clamp(rand(rng, Beta(10 * Φ(0.2 + L[j] * z[i]), 10 * (1 - Φ(0.2 + L[j] * z[i])))), 1e-4, 1 - 1e-4) for j in 1:p, i in 1:n]
    fb = fit_beta_gllvm(Yb; K = 1, link = GM.ProbitLink())
    @test isfinite(fb.loglik)
    @test_throws ArgumentError communality(fb, Yb)
    @test_throws ArgumentError sigma_y_site(fb, Yb)
    fl = fit_beta_gllvm(Yb; K = 1)
    @test all(0 .<= communality(fl, Yb) .<= 1)
end
