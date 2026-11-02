# #782: the truncated-Poisson bootstrap simulator must draw exact zero-truncated
# Poisson values in bounded time, even on a near-degenerate trait (mu -> 0).
using Test
using GLLVModels
using Random
using Distributions: Poisson

@testset "#782 truncated-Poisson bootstrap simulate: speed" begin
    Random.seed!(3)
    p, n = 4, 50
    z = randn(n)
    Y = [t == 1 ? 1 : (y = 0; while y < 1; y = rand(Poisson(exp(0.8 + 0.4 * z[s]))); end; y)
         for t in 1:p, s in 1:n]
    fit = fit_truncated_poisson_gllvm(Y; K = 1)
    # Guard: trait 1 (all ones) drives beta[1] to the degenerate regime; if this
    # setup stops reproducing it, the speed check below would pass vacuously.
    @test fit.β[1] < -8
    ad = GLLVModels._family_ci(fit, Y)
    ad.simulate(MersenneTwister(0))              # warm-up / compile
    t = @elapsed Yb = ad.simulate(MersenneTwister(1))
    @test all(Yb .>= 1)
    # Pre-fix: ~2.5 s (about 1/mu rejection draws). Fixed: milliseconds.
    @test t < 0.5
end

@testset "#782 _rand_truncpois: exact zero-truncated Poisson" begin
    N = 200_000
    for μ in (1e-12, 1e-6, 1e-3, 0.1, 0.9, 1.0, 2.5, 10.0)
        rng = MersenneTwister(782)
        t = @elapsed ys = [GLLVModels._rand_truncpois(rng, μ) for _ in 1:N]
        @test all(ys .>= 1)
        @test t < 5
        m = μ / (-expm1(-μ))                      # E[Y | Y >= 1]
        v = (μ + μ^2) / (-expm1(-μ)) - m^2        # Var[Y | Y >= 1]
        se = sqrt(v / N)
        @test abs(sum(ys) / N - m) <= 5 * se + 1e-12
        # P(Y = 1 | Y >= 1) = mu e^-mu / (1 - e^-mu)
        @test abs(count(==(1), ys) / N - μ * exp(-μ) / (-expm1(-μ))) <= 5 * sqrt(0.25 / N)
    end
end
