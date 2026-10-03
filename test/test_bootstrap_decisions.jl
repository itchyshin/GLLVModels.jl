using GLLVModels, Test, Random, LinearAlgebra, Statistics, StableRNGs

# Bootstrap decisions (#140, #156).
#
# Expected values are computed independently of the code under test:
#   - percentiles with Statistics.quantile over the replicates the function
#     returns, restricted by a convergence log kept by the test's own refit
#     wrapper (not by the function's bookkeeping);
#   - SD scale from the Wald estimate exp(log sd) and from fit.pars directly;
#   - non-convergence is injected deterministically through the internal
#     `_refit` keyword: every k-th refit is replaced by a finite fit that is
#     flagged not converged and whose sigma_eps is three times too large (what an
#     optimiser stopped at its iteration cap far from the optimum looks like).

# A finite, biased, non-converged copy of a converged refit.
function _bd_cripple(fb::GllvmFit)
    θ = copy(fb.pars.θ_packed)
    θ[1] += log(3.0)                        # packed entry 1 is log sigma_eps (no X)
    pars = merge(fb.pars, (σ_eps = 3 * fb.pars.σ_eps, θ_packed = θ))
    return GllvmFit(fb.model, pars, fb.logLik, fb.n_iter, false, fb.optim_result, fb.cputime)
end

# Refit wrapper: `bad(call_index)` selects the refits to cripple; `flags` records
# the `converged` flag of every fit handed back to the bootstrap.
function _bd_refit(bad::Function, flags::Vector{Bool})
    ncall = Ref(0)
    return function (yb; kwargs...)
        ncall[] += 1
        fb = fit_gaussian_gllvm(yb; kwargs...)
        out = bad(ncall[]) ? _bd_cripple(fb) : fb
        push!(flags, out.converged)
        return out
    end
end

# Raw-scale draws for packed column j: exp() for the log-SD terms (sigma_eps,
# sigma_B, sigma_W), identity for everything else (sigma_phy is signed).
_bd_is_log_sd(term::AbstractString) =
    startswith(term, "sigma_") && !startswith(term, "sigma_phy")
_bd_draws(term, col) = _bd_is_log_sd(term) ? exp.(col) : col

@testset "bootstrap decisions" begin
    rng = StableRNG(20261002)
    p, K, n = 4, 1, 150
    Λ = reshape([0.7, 0.5, 0.4, -0.3], p, K)
    y = Λ * randn(rng, K, n) + 0.5 * randn(rng, p, n)
    fit = fit_gaussian_gllvm(y; K = K)
    @test fit.converged
    wald = confint(fit; y = y)
    α = 0.025

    @testset "#156 SD terms are on the raw scale and agree with Wald" begin
        nb = 60
        ci = bootstrap_ci(fit; y = y, n_boot = nb, seed = 11)
        @test ci.term == wald.term
        i = findfirst(==("sigma_eps"), ci.term)
        @test i == 1
        # estimate is sigma_eps itself, not log sigma_eps
        @test ci.estimate[i] ≈ fit.pars.σ_eps rtol = 1e-12
        @test ci.estimate[i] ≈ wald.estimate[i] rtol = 1e-12
        @test ci.estimate[i] > 0
        # positive bounds that bracket the estimate on the same scale
        @test 0 < ci.lower[i] < ci.estimate[i] < ci.upper[i]
        # the bootstrap and Wald intervals for the same term overlap and have
        # comparable width (log-scale numbers near -0.7 could do neither)
        @test ci.lower[i] < wald.upper[i] && wald.lower[i] < ci.upper[i]
        wb = ci.upper[i] - ci.lower[i]
        ww = wald.upper[i] - wald.lower[i]
        @test 0.5 < wb / ww < 2.0
        # bounds are percentiles of the raw-scale draws (all refits converge here)
        @test ci.n_converged == nb
        draws = exp.(ci.replicates[:, i])
        @test ci.lower[i] ≈ quantile(draws, α) rtol = 1e-12
        @test ci.upper[i] ≈ quantile(draws, 1 - α) rtol = 1e-12
        # linear terms (beta, Lambda) are unchanged: same scale as Wald
        for j in 2:length(ci.term)
            @test ci.estimate[j] ≈ wald.estimate[j] rtol = 1e-12
            @test ci.lower[j] ≈ quantile(ci.replicates[:, j], α) rtol = 1e-12
            @test ci.upper[j] ≈ quantile(ci.replicates[:, j], 1 - α) rtol = 1e-12
        end
        # a single-term selection keeps the same scale
        one = bootstrap_ci(fit; y = y, n_boot = nb, seed = 11, parms = "sigma_eps")
        @test one.term == ["sigma_eps"]
        @test one.estimate[1] ≈ fit.pars.σ_eps rtol = 1e-12
        @test one.lower[1] ≈ ci.lower[i] rtol = 1e-12
    end

    @testset "#156 signed sigma_phy is not exponentiated" begin
        rngp = StableRNG(20261002)
        Σphy = [1 .5 .2 .2; .5 1 .2 .2; .2 .2 1 .5; .2 .2 .5 1.0]
        σphy = [0.5, -0.4, 0.6, 0.3]
        φ = cholesky(Σphy).L * randn(rngp, p)
        yp = Λ * randn(rngp, K, n) + 0.5 * randn(rngp, p, n) .+ (σphy .* φ)
        fp = fit_gaussian_gllvm(yp; K = K, has_phy_unique = true, Σ_phy = Σphy)
        # this draw gives a negative sigma_phy[1], which exp() could never return
        @test fp.pars.σ_phy[1] < 0
        wp = confint(fp; y = yp, Σ_phy = Σphy)
        nb = 24
        cp = bootstrap_ci(fp; y = yp, Σ_phy = Σphy, n_boot = nb, seed = 5)
        @test cp.term == wp.term
        @test cp.n_converged == nb
        for (j, t) in enumerate(cp.term)
            @test cp.estimate[j] ≈ wp.estimate[j] rtol = 1e-12
            draws = _bd_draws(t, cp.replicates[:, j])
            @test cp.lower[j] ≈ quantile(draws, α) rtol = 1e-12
            @test cp.upper[j] ≈ quantile(draws, 1 - α) rtol = 1e-12
        end
        for t in 1:p
            j = findfirst(==("sigma_phy[$t]"), cp.term)
            @test cp.estimate[j] == fp.pars.σ_phy[t]
        end
        js = findfirst(==("sigma_eps"), cp.term)
        @test cp.estimate[js] ≈ fp.pars.σ_eps rtol = 1e-12
        @test cp.lower[js] > 0
    end

    @testset "#156 Gaussian-record route (offset) uses the same scale" begin
        yr = Λ * randn(StableRNG(7), K, 100) + 0.5 * randn(StableRNG(8), p, 100)
        fr = fit_gaussian_gllvm(yr; K = K, offset = zeros(p, 100))
        @test GLLVModels._has_gaussian_record(fr)
        wr = confint(fr, yr)
        cr = bootstrap_ci(fr; y = yr, n_boot = 14, seed = 3)
        @test cr.term == wr.term
        @test cr.estimate[1] ≈ fr.pars.σ_eps rtol = 1e-12
        @test cr.estimate[1] ≈ wr.estimate[1] rtol = 1e-12
        @test cr.lower[1] > 0
        @test cr.lower[1] ≈ quantile(exp.(cr.replicates[:, 1]), α) rtol = 1e-12
        @test cr.upper[1] ≈ quantile(exp.(cr.replicates[:, 1]), 1 - α) rtol = 1e-12
        @test cr.estimate[2:end] ≈ wr.estimate[2:end] rtol = 1e-12
    end

    @testset "#156 term names agree between bootstrap and confint layouts" begin
        # the SD back-transform reads the term kinds from the confint layout, so
        # the two name builders must stay in step for every structure
        for kw in ((;), (has_diag = true,), (has_diag = true, K_W = 1))
            f = fit_gaussian_gllvm(y; K = K, kw...)
            @test GLLVModels._bootstrap_term_names(f) == GLLVModels._confint_all_term_names(f)[1]
        end
    end

    @testset "#140 non-converged refits are excluded from percentiles and counted" begin
        nb = 40
        flags = Bool[]
        ci = bootstrap_ci(fit; y = y, n_boot = nb, seed = 7,
                          _refit = _bd_refit(b -> b % 4 == 0, flags))
        used = copy(flags)
        @test length(used) == nb
        @test count(!, used) == 10          # the injection is deterministic: refits 4, 8, ..., 40
        @test used == [b % 4 != 0 for b in 1:nb]
        # the non-converged draws are finite, so only the flag can exclude them
        @test all(isfinite, ci.replicates)
        for (j, t) in enumerate(ci.term)
            kept = _bd_draws(t, ci.replicates[used, j])
            @test ci.lower[j] ≈ quantile(kept, α) rtol = 1e-12
            @test ci.upper[j] ≈ quantile(kept, 1 - α) rtol = 1e-12
        end
        # guard against a vacuous test: the biased draws do move sigma_eps's upper bound
        allrows = exp.(ci.replicates[:, 1])
        @test quantile(allrows, 1 - α) > 1.2 * ci.upper[1]
        # counts and flags
        @test ci.n_converged == 30
        @test ci.n_used == 30
        @test ci.n_dropped == 10
        @test ci.converged == used
        @test size(ci.replicates) == (nb, length(fit.pars.θ_packed))
        # existing fields keep their meaning
        @test ci.estimate[1] ≈ fit.pars.σ_eps rtol = 1e-12
        @test all(ci.lower .< ci.upper)
        # a quarter dropped: no warning
        @test_logs min_level = Base.CoreLogging.Warn bootstrap_ci(fit; y = y, n_boot = nb, seed = 7,
                                          _refit = _bd_refit(b -> b % 4 == 0, Bool[]))
    end

    @testset "#140 warning only when more than half are dropped" begin
        nb = 40
        # exactly half dropped: no warning
        half = @test_logs min_level = Base.CoreLogging.Warn bootstrap_ci(fit; y = y, n_boot = nb, seed = 7,
                                                 _refit = _bd_refit(b -> iseven(b), Bool[]))
        @test half.n_dropped == 20 && half.n_used == 20
        # three quarters dropped: exactly one warning, naming the counts
        flags = Bool[]
        ci = @test_logs (:warn, r"30 of 40") bootstrap_ci(fit; y = y, n_boot = nb, seed = 7,
                                           _refit = _bd_refit(b -> b % 4 != 0, flags))
        @test ci.n_dropped == 30 && ci.n_used == 10 && ci.n_converged == 10
        kept = exp.(ci.replicates[flags, 1])
        @test length(kept) == 10
        @test ci.lower[1] ≈ quantile(kept, α) rtol = 1e-12
        # every refit non-converged: NaN bounds (the old fewer-than-10 rule), one warning
        none = @test_logs (:warn, r"40 of 40") bootstrap_ci(fit; y = y, n_boot = nb, seed = 7,
                                             _refit = _bd_refit(b -> true, Bool[]))
        @test none.n_used == 0 && none.n_dropped == 40
        @test all(isnan, none.lower) && all(isnan, none.upper)
    end

    @testset "#140 bootstrap_ci_derived drops non-converged refits" begin
        nb = 40
        sig = fb -> fb.pars.σ_eps
        flags = Bool[]
        d = GLLVModels.bootstrap_ci_derived(fit, sig; y = y, n_boot = nb, seed = 7,
                                 _refit = _bd_refit(b -> b % 4 == 0, flags))
        used = copy(flags)
        @test used == [b % 4 != 0 for b in 1:nb]
        @test all(isfinite, d.replicates)       # biased draws are finite: only the flag excludes them
        kept = d.replicates[used]
        @test d.lower ≈ quantile(kept, α) rtol = 1e-12
        @test d.upper ≈ quantile(kept, 1 - α) rtol = 1e-12
        @test quantile(d.replicates, 1 - α) > 1.2 * d.upper     # not vacuous
        @test d.estimate ≈ fit.pars.σ_eps rtol = 1e-12
        @test d.n_converged == 30
        @test d.n_used == 30 && d.n_dropped == 10
        @test d.n_valid == nb                   # n_valid keeps its meaning: finite derived values
        @test d.converged == used
        @test_logs min_level = Base.CoreLogging.Warn GLLVModels.bootstrap_ci_derived(fit, sig; y = y, n_boot = nb, seed = 7,
                                                  _refit = _bd_refit(b -> b % 4 == 0, Bool[]))
        # more than half dropped: one warning
        flags2 = Bool[]
        d2 = @test_logs (:warn, r"30 of 40") GLLVModels.bootstrap_ci_derived(fit, sig; y = y, n_boot = nb, seed = 7,
                                                   _refit = _bd_refit(b -> b % 4 != 0, flags2))
        @test d2.n_used == 10 && d2.n_dropped == 30
        @test d2.lower ≈ quantile(d2.replicates[flags2], α) rtol = 1e-12
    end
end
