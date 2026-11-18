# #780: confint must not hand back unflagged intervals for a non-converged input fit.
# #757: confint_lv_effects(...; method = :bootstrap) must drop non-converged refits and
#       count them, rather than reporting every finite refit as converged.
using Test
using GLLVModels
using Random
using Logging
using Distributions: Poisson

@testset "#780 confint flags a non-converged input fit" begin
    Random.seed!(3)
    Y = rand(Poisson(3), 5, 60)
    bad = fit_poisson_gllvm(Y; K = 1, iterations = 2)
    @test !bad.converged                       # guard: the repro regime is reproduced
    ci = @test_logs (:warn, r"did not converge") match_mode = :any confint(bad, Y; method = :wald)
    @test ci.converged === false
    @test_logs (:warn, r"did not converge") match_mode = :any vcov(bad, Y)

    good = fit_poisson_gllvm(Y; K = 1)
    @test good.converged                       # guard: the control fit is a converged one
    cig = @test_logs min_level = Logging.Warn confint(good, Y; method = :wald)
    @test cig.converged === true
end

@testset "#757 lv-effects bootstrap drops non-converged refits" begin
    Random.seed!(11)
    p, n = 4, 40
    x = reshape(randn(n), n, 1)
    λ = [1.0, 0.5, -0.5, 0.8]
    Y = [rand(Poisson(exp(0.5 + λ[t] * 0.7 * x[s, 1]))) for t in 1:p, s in 1:n]
    fit = fit_poisson_gllvm(Y; K = 1, X_lv = x)
    r = with_logger(NullLogger()) do
        confint_lv_effects(fit, Y, x; method = :bootstrap, n_boot = 20, seed = 1,
                           bootstrap_iterations = 3)
    end
    # 3 iterations cannot converge a Laplace refit: on the base commit every finite refit was
    # counted ("n_converged = 20 of 20"); now they are dropped and reported.
    @test r.n_converged < 20
    @test r.n_dropped == 20 - r.n_converged
end
