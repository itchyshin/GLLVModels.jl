# #794 / #764: Wald inference on a fit made with `fit_gaussian_gllvm(y; K, lambda_constraint = M)`.
#
# Pinned loadings are not parameters (gllvmTMB maps them off), so the Wald routes invert the
# observed information of the free parameters only (R's `sd_report$cov.fixed`), and a raw
# pinned loading has se = 0 and lower = upper = estimate. That holds only while the free
# information is positive definite. gllvmTMB's `loading_ci()` (R/loading-ci.R at aec24e16,
# lines 266-291) returns `NA` se and bounds for every row, pinned rows included, with
# `pd_hessian = FALSE`, when `pdHess` is FALSE. `confint` and `vcov` already did this; on such
# a fit `loading_ci`, `raw_loading_wald_ci` and `flag_unreliable_loadings` gave the pinned rows
# se = 0 and finite bounds, and `raw_loading_wald_ci` reported `pd_hessian = true` for a pinned
# or structural entry without forming the Hessian.
#
# `fit_lognormal_gllvm` passed `lambda_constraint` through to the Gaussian fitter, so the pins
# were applied, but a `LognormalFit` does not carry them, and its `confint` and `vcov` inverted
# the information of every loading, pinned ones included: the #794 defect on a sibling route.
# It now refuses `lambda_constraint`, as the other non-Gaussian fitters do.
using Test, GLLVModels, LinearAlgebra, ForwardDiff, StableRNGs

const _L10 = GLLVModels

# The #764 design: p = 6 traits, n = 200 units, K = 2.
function _l10_data()
    rng = StableRNG(21)
    p, n, K = 6, 200, 2
    Λ = randn(rng, p, K)
    return Λ * randn(rng, K, n) .+ 0.7 .* randn(rng, p, n)
end

# Observed information of the free parameters, by ForwardDiff on the packed NLL.
function _l10_free_information(fit, Y)
    nll = _L10._confint_reconstruct_nll(fit, Y, nothing, nothing)
    H = ForwardDiff.hessian(nll, fit.pars.θ_packed)
    pins = _L10._lambda_constraint_pinned_theta_indices(fit)
    free = setdiff(1:length(fit.pars.θ_packed), pins)
    return free, pins, Symmetric((H[free, free] .+ H[free, free]') ./ 2)
end

# Same model and pins at a chosen parameter vector (a non-MLE point for the non-PD case).
function _l10_at(fit, θ)
    spec = _L10._profile_spec(fit)
    u = _L10._derived_unpack(θ, spec)
    return _L10.GllvmFit(fit.model, merge(fit.pars, (θ_packed = θ, Λ = Matrix(u.Λ_B), σ_eps = u.σ_eps)),
                         fit.logLik, fit.n_iter, fit.converged, fit.optim_result, fit.cputime,
                         fit.integration)
end

@testset "lambda_constraint Wald routes (#794, #764)" begin
    Y = _l10_data()
    p, K = size(Y, 1), 2
    f0 = fit_gaussian_gllvm(Y; K = K)
    M = fill(NaN, p, K)
    M[4, 1] = f0.pars.Λ[4, 1]           # the #764 pin, at the unconstrained estimate
    M[5, 2] = 0.3                       # a pin away from it
    fit = fit_gaussian_gllvm(Y; K = K, lambda_constraint = M)
    pinned_entries = [(4, 1), (5, 2)]
    is_pinned(t, k) = k > t || (t, k) in pinned_entries

    @testset "positive definite free information: one rule in every Wald route" begin
        free, pins, Hf = _l10_free_information(fit, Y)
        @test issuccess(cholesky(Hf; check = false))
        Vf = inv(Hf)
        se_ref = zeros(length(fit.pars.θ_packed))
        se_ref[free] = sqrt.(diag(Vf))

        ci = confint(fit; y = Y)
        @test ci.pd_hessian
        @test isapprox(ci.se, se_ref; atol = 1e-10, rtol = 0)
        for (t, k) in pinned_entries
            j = findfirst(==("Lambda_B[$t,$k]"), ci.term)
            @test ci.se[j] == 0.0
            @test ci.lower[j] == ci.estimate[j] == ci.upper[j] == M[t, k]
        end
        V = vcov(fit, Y)
        @test all(iszero, V[pins, :]) && all(iszero, V[:, pins])
        @test isapprox(V[free, free], Vf; atol = 1e-12, rtol = 0)

        lc = loading_ci(fit, Y)
        @test [(r.trait, r.axis) for r in lc] == [(t, k) for k in 1:K for t in 1:p]
        @test [r.pinned for r in lc] == [is_pinned(t, k) for k in 1:K for t in 1:p]
        @test [r.estimate for r in lc] == vec(fit.pars.Λ)
        for r in lc
            w = raw_loading_wald_ci(fit, r.trait, r.axis; y = Y)
            @test w.pd_hessian
            @test (r.se, r.lower, r.upper) == (w.se_transformed, w.lower, w.upper)
            if r.pinned
                @test w.method === :pinned
                @test r.se == 0.0 && r.lower == r.estimate == r.upper
            else
                j = _L10._lambda_b_theta_index(fit, r.trait, r.axis)
                @test isapprox(r.se, se_ref[j]; atol = 1e-10, rtol = 0)
            end
        end
        fl = flag_unreliable_loadings(fit, Y)
        @test all(r -> r.pd_hessian, fl)
        @test isapprox([r.se for r in fl], [r.se for r in lc]; atol = 1e-10, rtol = 0)

        # Standardised scale: a pinned loading divided by the model SD still depends on the
        # free parameters, so its delta-method SE is not 0 (gllvmTMB keeps `pinned = TRUE` as
        # provenance and does not force the derived SE to 0).
        ls = loading_ci(fit, Y; loading_scale = :standardized)
        for r in ls
            w = standardized_loading_wald_ci(fit, r.trait, r.axis; y = Y)
            @test (r.estimate, r.se, r.lower, r.upper) ==
                  (w.estimate, w.se_transformed, w.lower, w.upper)
        end
        r41 = only(filter(r -> (r.trait, r.axis) == (4, 1), ls))
        @test r41.pinned && r41.se > 0
    end

    @testset "non-positive-definite free information: every row NaN, pinned rows included" begin
        # A saddle point of the #764 confirmatory model: the free loadings of axis 2 set to 0.
        fc = fit_gaussian_gllvm(Y; K = K, lambda_constraint = [i == 4 && k == 1 ? M[4, 1] : NaN
                                                              for i in 1:p, k in 1:K])
        θ = copy(fc.pars.θ_packed)
        for t in 2:p
            θ[_L10._lambda_b_theta_index(fc, t, 2)] = 0.0
        end
        fB = _l10_at(fc, θ)
        _, _, Hf = _l10_free_information(fB, Y)
        @test eigmin(Matrix(Hf)) < -1.0                        # premise: far from positive definite
        @test !issuccess(cholesky(Hf; check = false))

        ci = confint(fB; y = Y)
        @test !ci.pd_hessian
        @test all(isnan, ci.se) && all(isnan, ci.lower) && all(isnan, ci.upper)
        @test all(isnan, vcov(fB, Y))

        for (t, k) in ((4, 1), (1, 2), (3, 1))                # pinned, structural, free
            w = raw_loading_wald_ci(fB, t, k; y = Y)
            @test w.pd_hessian === false
            @test w.method === :failed
            @test isnan(w.se_transformed) && isnan(w.lower) && isnan(w.upper)
            @test w.estimate == fB.pars.Λ[t, k]
        end
        for scale in (:raw, :standardized)
            lc = loading_ci(fB, Y; loading_scale = scale)
            @test all(r -> isnan(r.se) && isnan(r.lower) && isnan(r.upper), lc)
            @test [r.pinned for r in lc] == [k > t || (t, k) == (4, 1) for k in 1:K for t in 1:p]
        end
        @test [r.estimate for r in loading_ci(fB, Y)] == vec(fB.pars.Λ)
        fl = flag_unreliable_loadings(fB, Y)
        @test all(r -> r.pd_hessian === false, fl)
        @test all(r -> isnan(r.se) && isnan(r.lower) && isnan(r.upper), fl)
        @test all(r -> r.unreliable === missing, fl)
    end

    @testset "a pinned or structural entry needs y to be judged" begin
        @test_throws ArgumentError raw_loading_wald_ci(fit, 4, 1)
        @test_throws ArgumentError raw_loading_wald_ci(fit, 1, 2)
        @test_throws ArgumentError raw_loading_wald_ci(fit, 3, 1)
    end

    @testset "exploratory fit: the structural zero is unchanged" begin
        w = raw_loading_wald_ci(f0, 1, 2; y = Y)
        @test w == (; estimate = 0.0, lower = 0.0, upper = 0.0, se_transformed = 0.0,
                    transform = :identity, pd_hessian = true, method = :pinned)
        lc = loading_ci(f0, Y)
        for r in lc
            r.pinned || continue
            @test r.se == 0.0 && r.lower == r.estimate == r.upper == 0.0
        end
    end

    @testset "lognormal refuses lambda_constraint (sibling of #794)" begin
        YL = exp.((Y .+ [5.0, -3.0, 2.0, 10.0, 0.0, 1.0]) ./ 2)
        ML = fill(NaN, p, K)
        ML[3, 1] = 0.0
        for f in (() -> fit_lognormal_gllvm(YL; K = K, lambda_constraint = ML),
                  () -> fit_gllvm(YL; family = Lognormal(), K = K, lambda_constraint = ML))
            err = try
                f()
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test err isa ArgumentError && occursin("lambda_constraint", err.msg)
            @test err isa ArgumentError && occursin("fit_gaussian_gllvm", err.msg)
        end
        # `lambda_constraint = nothing` is no constraint: the fit is unchanged.
        a = fit_lognormal_gllvm(YL; K = K)
        b = fit_lognormal_gllvm(YL; K = K, lambda_constraint = nothing)
        @test a.loglik == b.loglik && a.Λ == b.Λ && a.σ == b.σ
        # The route the error names fits the same loadings, with the pinned entry held.
        Z = log.(YL)
        g = fit_gaussian_gllvm(Z .- sum(Z; dims = 2) ./ size(Z, 2); K = K, lambda_constraint = ML)
        @test g.pars.Λ[3, 1] == 0.0
        gci = confint(g; y = Z .- sum(Z; dims = 2) ./ size(Z, 2), parm = "Lambda_B[3,1]")
        @test gci.se == [0.0]
    end
end
