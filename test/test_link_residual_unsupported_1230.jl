# Issue #1230: unsupported family/link pairs must not reach a raw MethodError.
using GLLVModels, Test, Random, Distributions

@testset "link_residual unsupported family/link (#1230)" begin
    rng = MersenneTwister(4)
    p, n = 6, 60
    z = randn(rng, n)
    L = [0.9, 0.6, -0.7, 0.5, 0.8, -0.4]
    Yb = [clamp(rand(rng, Beta(10 * cdf(Normal(), 0.2 + L[j] * z[i]),
                      10 * (1 - cdf(Normal(), 0.2 + L[j] * z[i])))), 1e-4, 1 - 1e-4)
          for j in 1:p, i in 1:n]
    Yp = [rand(rng, Poisson(3 + 2 * abs(L[j] * z[i]))) for j in 1:p, i in 1:n]
    fβ = fit_beta_gllvm(Yb; K = 1, link = ProbitLink())
    @test fβ.converged
    @test_throws ArgumentError communality(fβ, Yb)
    fp = fit_poisson_gllvm(Yp; K = 1, link = IdentityLink())
    @test_throws ArgumentError sigma_y_site(fp, Yp)
end