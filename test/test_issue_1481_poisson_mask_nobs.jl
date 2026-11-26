# Issue #1481: Poisson masked fit — nobs/bic(fit, Y) must use the training mask.
using GLLVModels, Test, Random, Distributions
using GLLVModels: StatsAPI, nobs, bic, loglikelihood

@testset "Poisson mask retained for nobs/bic (#1481)" begin
    Random.seed!(1126)
    p, n, K = 5, 30, 1
    Y = [rand(Poisson(2)) for _ in 1:p, _ in 1:n]
    M = trues(p, n)
    M[1, 1] = false
    M[2, 2] = false

    f = fit_gllvm(Y; family = Poisson(), K = K, mask = M, iterations = 150)
    n_masked = p * n - 2

    @test f.obs_mask == M
    @test nobs(f, Y) == n_masked
    @test nobs(f, Y; mask = M) == n_masked
    @test bic(f, Y) ≈ bic(f, Y; mask = M)
    @test !isapprox(bic(f, Y), StatsAPI.dof(f) * log(p * n) - 2 * loglikelihood(f); rtol = 1e-10)

    f_full = fit_gllvm(Y; family = Poisson(), K = K, iterations = 150)
    @test f_full.obs_mask === nothing
    @test nobs(f_full, Y) == p * n
end