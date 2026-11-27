# refs #1547 — ZiFit post-fit API (predict, confint, aic, simulate, …).
using Test, GLLVModels, Random, Distributions
using GLLVModels: predict, fitted, confint, aic, simulate, getLV, vcov, nobs

@testset "ZiFit post-fit (#1547)" begin
    Random.seed!(11270401)
    N, p, n = 4, 3, 35
    Y = rand(0:N, p, n)

    fit = fit_zi_gllvm(Y; family = ZiBinomial(), K = 1, trials = N, iterations = 80)
    @test fit isa ZiFit

  @test size(predict(fit, Y)) == (p, n)
  @test size(fitted(fit, Y)) == (p, n)
  @test all(isfinite, predict(fit, Y))
  @test isfinite(aic(fit))
  @test nobs(fit, Y) == p * n
  @test size(getLV(fit, Y)) == (n, 1)

  ci = confint(fit, Y; method = :wald, parm = "betac[1]")
  @test ci.method == :wald
  @test length(ci.term) >= 1

  Ys = simulate(fit, 12; rng = MersenneTwister(47))
  @test size(Ys) == (p, 12)
  @test all(0 .<= Ys .<= N)
end