# GllvmFit: nobs(fit) without Y (#1569)

using GLLVModels, Test, Random
using GLLVModels: StatsAPI

@testset "GllvmFit nobs(fit) without response (#1569)" begin
    Random.seed!(1127)
    Y = randn(5, 12)
    K = 1
    fit = fit_gaussian_gllvm(Y; K = K)
    pn = 5 * 12
    @test isfinite(StatsAPI.loglikelihood(fit))
    @test isfinite(StatsAPI.aic(fit))
    @test StatsAPI.nobs(fit) == pn
    @test StatsAPI.nobs(fit, Y) == pn
    @test StatsAPI.bic(fit, StatsAPI.nobs(fit)) ≈ StatsAPI.bic(fit, Y)
end