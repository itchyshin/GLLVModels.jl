using Test, GLLVModels, Random, Distributions

@testset "getLV refuses mask, N and X size mismatch (refs 1219)" begin
    rng = MersenneTwister(2)
    p, n = 5, 100
    Y = [rand(rng, Binomial(5, 0.3)) for _ in 1:p, _ in 1:n]
    fp = fit_poisson_gllvm(Int.(Y); K = 1)
    fb = fit_binomial_gllvm(Int.(Y); K = 1, N = fill(5, p, n))
    X = reshape(randn(rng, n), 1, n, 1) .* ones(p, 1, 1)
    fc = fit_gllvm_cov(Int.(Y); family = Binomial(), X = X, K = 1, N = fill(5, p, n))
    @test_throws DimensionMismatch predict(fp, Y; mask = trues(p, 3))
    @test_throws DimensionMismatch getLV(fb, Y; N = fill(5, p, 3))
    @test_throws DimensionMismatch getLV(fc, Y, X[:, 1:10, :])
end