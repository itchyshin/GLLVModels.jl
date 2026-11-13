using GLLVModels, Test, Random, Distributions

# simulate(::BinomialFit / ::GllvmCovFit; N) must refuse a trials matrix that is not p × n.
# The sampling loops run under @inbounds, so a too-small N was read out of bounds:
# a 2 × 10 N returned counts up to 2.5e13 in a default build and a BoundsError under
# --check-bounds=yes.
@testset "simulate: trials N must be p × n" begin
    rng = MersenneTwister(2); p, n = 5, 100
    Y = [rand(rng, Binomial(5, 0.3 + 0.1 * randn(rng)^2 / 4)) for t in 1:p, s in 1:n]
    fb = fit_binomial_gllvm(Y; K = 1, N = fill(5, p, n))
    @test_throws DimensionMismatch simulate(fb, 10; N = fill(5, 2, 10))
    @test_throws DimensionMismatch simulate(fb, 10; N = fill(5, p, 3))
    Ys = simulate(fb, 10; N = fill(5, p, 10), rng = MersenneTwister(1))
    @test size(Ys) == (p, 10) && all(0 .<= Ys .<= 5)

    X = reshape(randn(rng, n), 1, n, 1) .* ones(p, 1, 1)
    fc = fit_gllvm_cov(Y; family = Binomial(), X = X, K = 1, N = fill(5, p, n))
    @test_throws DimensionMismatch simulate(fc, zeros(p, 10, 1); N = fill(5, p, 3))
    @test_throws DimensionMismatch simulate(fc, zeros(p, 10, 1); N = fill(5, 2, 10))
    Yc = simulate(fc, zeros(p, 10, 1); N = fill(5, p, 10), rng = MersenneTwister(1))
    @test size(Yc) == (p, 10) && all(0 .<= Yc .<= 5)
end
