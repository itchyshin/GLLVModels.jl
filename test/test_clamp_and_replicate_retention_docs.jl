using GLLVModels, Test, Distributions, Random, StableRNGs

# Characterisation tests for two behaviours documented in docs/src/confidence-intervals.md
# and docs/src/pitfalls.md. They pin CURRENT behaviour; no default is changed here.
#   1. Gaussian `bootstrap_ci` keeps non-converged refits in `replicates`/`converged`
#      (for inspection) but leaves them out of the percentile bounds.
#   2. The Binomial Laplace path clamps mu to [1e-12, 1-1e-12] and eta to +-30.

@testset "documented: bootstrap_ci replicate retention" begin
    rng = StableRNG(20261031)
    p, K, n = 4, 1, 120
    Λ = reshape([0.7, 0.5, 0.4, -0.3], p, K)
    y = Λ * randn(rng, K, n) + 0.5 * randn(rng, p, n)
    fit = fit_gaussian_gllvm(y; K = K)
    ncall = Ref(0)
    refit = function (yb; kwargs...)
        ncall[] += 1
        fb = fit_gaussian_gllvm(yb; kwargs...)
        ncall[] % 5 == 0 || return fb
        return GllvmFit(fb.model, fb.pars, fb.logLik, fb.n_iter, false, fb.optim_result, fb.cputime)
    end
    nb = 30
    ci = bootstrap_ci(fit; y = y, n_boot = nb, seed = 3, _refit = refit)
    # rows of non-converged refits are retained (finite, not NaN) ...
    @test size(ci.replicates, 1) == nb
    @test all(isfinite, ci.replicates)
    @test count(!, ci.converged) == nb ÷ 5
    # ... but excluded from the bounds and counted
    @test ci.n_used == ci.n_converged == nb - nb ÷ 5
    @test ci.n_dropped == nb ÷ 5
    kept = exp.(ci.replicates[ci.converged, 1])   # packed entry 1 is log(sigma_eps)
    @test ci.lower[1] ≈ quantile(kept, 0.025) rtol = 1e-12
    @test ci.upper[1] ≈ quantile(kept, 0.975) rtol = 1e-12
end

@testset "documented: Binomial Laplace mu/eta clamps" begin
    @test GLLVModels._clamp_mu(Binomial(), 0.0) == 1e-12
    @test GLLVModels._clamp_mu(Binomial(), 1.0) == 1 - 1e-12
    @test GLLVModels._clamp_eta(100.0) == 30.0
    @test GLLVModels._clamp_eta(-100.0) == -30.0

    y = [1, 0, 1]; n = [1, 1, 1]; Λ0 = zeros(3, 1)
    ll(b) = GLLVModels.laplace_loglik_site(y, n, Λ0, fill(b, 3), LogitLink())
    # the objective is flat beyond |eta| = 30 (the clamp)
    @test ll(60.0) == ll(30.0)
    @test ll(-60.0) == ll(-30.0)
    # each saturated misfit cell floors at log(1e-12), not at its true value -eta
    @test ll(30.0) ≈ log(1e-12) rtol = 1e-5          # one y = 0 cell at mu = 1 - 1e-12
    @test ll(-30.0) ≈ 2 * log(1e-12) rtol = 1e-5     # two y = 1 cells at mu = 1e-12
    # the unclamped log-likelihood would be -30 / -60 (clamp makes it less negative)
    @test_broken ll(30.0) ≈ -30.0 atol = 1e-6
end
