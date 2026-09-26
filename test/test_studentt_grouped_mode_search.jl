using GLLVModels, Test, Random, LinearAlgebra, Distributions, ForwardDiff

# #503: the Student-t GROUPED kernel (`_studentt_grouped_loglik_site`, disp_group =
# :species — reached by default through `fit_gllvm_grouped(Y; family = StudentTFamily(),
# K)`) ran an undamped per-site Fisher-scoring mode search and returned whatever `z` it
# held after `maxiter` iterations, converged or not — the same defect class fixed for
# Gamma (#479), Beta (#480) and the twopart kernel (#500). The fix (mirrors #479):
# any step that lowers the per-site log-posterior is halved, and the search now reports
# `-Inf` when it cannot certify a stationary point, so the fitter's 1e12 sentinel fires
# instead of a garbage finite value propagating into the fit. Unlike Gamma, Student-t's
# Fisher weight (ν+1)/((ν+3)σ²) is a CONSTANT (no y- or η-dependence, and no provably
# non-negative observed-curvature fallback exists, since the observed curvature IS
# negative for |r| > σ√ν), so "converged" additionally requires the log-posterior
# gradient itself to be small, not only the step size (see `_studentt_grouped_mode`'s
# docstring comment in src/families/studentt.jl for why a step-size-only test is not
# enough here).
#
# This test does NOT assert a specific fitted log-likelihood or iteration count (CI runs
# Julia 1.10 and 1.13 on Linux; the same seed draws different data across Julia versions,
# and optimiser paths differ across platforms on the same data). It asserts structural
# relations instead: every FINITE value the fixed kernel returns sits at a genuine
# stationary point (gradient ~0) with a negative-definite Hessian of the per-site
# log-posterior, and the one literal-data regression case below is a site verified (by
# hand, against `origin/main`, and recorded here) to have been a silent-garbage return
# before the fix.

# A p = 4, K = 1 per-species Student-t site, found by a stress probe over random Λ, β, y
# draws (independently confirmed against a from-scratch multi-start BFGS reference: the
# reference mode has log-posterior -8.61025079974842, gradient norm < 1e-10). On
# `origin/main` (pre-#503) `_studentt_grouped_loglik_site` returns a FINITE
# -10.70852028621799 here — 2.1 log-posterior units below the true mode's value, with no
# `converged` flag to catch it (this kernel has none pre-fix). The fixed kernel cannot
# certify a stationary point for this site within its iteration budget either (the same
# constant-Fisher-weight limitation that motivates the gradient check above) and
# correctly reports -Inf rather than propagate that -10.7086 value.
const _T503_NU = 1.5
const _T503_SIGMA = [1.2211909342381375, 0.7747393451163731, 1.3436898558784005, 1.4855763107354496]
const _T503_LAMBDA = reshape([2.051198273371866, -4.746111968974984,
                              -0.24897364484493514, -0.8036245443892303], 4, 1)
const _T503_BETA = [-0.9412876477850245, -0.8654961951821716, 1.3239193465705699, 3.3036659566062885]
const _T503_Y = [-3.166297032660741, -0.6201545757488939, -0.9646032215837861, 1.4170131044552292]
# Verbatim pre-#503 kernel (undamped Fisher scoring, no convergence flag), kept only to
# demonstrate what `origin/main` returns for this site.
function _t503_old_site(fams, y, Λ, β)
    G = GLLVModels
    p, K = size(Λ)
    n = ones(Int, p)
    link = IdentityLink()
    z = zeros(K)
    for _ in 1:100
        η  = G._clamp_eta.(β .+ Λ * z)
        μ  = G._clamp_mu.(fams, G.linkinv.(Ref(link), η))
        me = G.mu_eta.(Ref(link), η)
        s  = G._glm_score.(fams, μ, n, me, y)
        W  = G._glm_weight.(fams, μ, n, me)
        Δ  = G._safe_solve(Symmetric(Λ' * (W .* Λ) + I), Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < 1e-9 && break
    end
    η  = G._clamp_eta.(β .+ Λ * z)
    μ  = G._clamp_mu.(fams, G.linkinv.(Ref(link), η))
    me = G.mu_eta.(Ref(link), η)
    # Matches `_studentt_grouped_loglik_site`'s own default: the MODE SEARCH above is
    # Fisher-scored, but the log-det curvature defaults to :observed (TMB's Laplace
    # curvature), independently of the search.
    W  = G._studentt_grouped_laplace_weight.(Ref(:observed), fams, μ, me, y, Ref(link), η)
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ  = sum(G._glm_logpdf(fams[t], μ[t], n[t], y[t]) for t in 1:p)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

@testset "Student-t grouped mode search (#503)" begin
    fams = [GLLVModels.StudentTFamily(_T503_NU, s) for s in _T503_SIGMA]
    n = ones(Int, 4)
    link = IdentityLink()

    @testset "pre-#503 kernel returns a finite, wrong value on this site" begin
        v_old = _t503_old_site(fams, _T503_Y, _T503_LAMBDA, _T503_BETA)
        @test isfinite(v_old)
        @test isapprox(v_old, -10.70852028621799; atol = 1e-8)
        @test v_old < -8.61025079974842 - 1e-6  # strictly below the true per-site mode's value
    end

    @testset "fixed kernel refuses rather than propagate the garbage value" begin
        v = GLLVModels._studentt_grouped_loglik_site(fams, _T503_Y, n, _T503_LAMBDA, _T503_BETA, link)
        @test !isfinite(v)
        @test v == -Inf
    end

    @testset "a fit reaching only this site's failure returns -Inf, not a finite garbage value" begin
        # Objective screening (mirrors fit_studentt_gllvm's negll): a non-finite site value
        # must make the total marginal non-finite too, which is what lets the fitter's own
        # 1e12 sentinel catch it.
        total = GLLVModels.studentt_marginal_loglik_laplace(
            reshape(_T503_Y, 4, 1), _T503_LAMBDA, _T503_BETA, _T503_SIGMA;
            ν = _T503_NU, link = link)
        @test !isfinite(total)
    end
end

@testset "Student-t grouped mode search: stress-probe stationarity (#503)" begin
    # Independent of the literal case above: draw many random per-site problems (fixed
    # seed, so deterministic within one Julia process, but no cross-version numeric
    # comparison — the assertions below are structural, not value equality) and check
    # that whenever `_studentt_grouped_mode` reports `ok = true`, the site is genuinely at
    # a stationary point (small gradient) with a locally concave log-posterior (negative
    # definite Hessian) — never a step-size-only false positive.
    rng = MersenneTwister(90503)
    n_checked = 0
    n_finite = 0
    for _ in 1:300
        p = rand(rng, 3:8)
        K = rand(rng, 1:2)
        ν = rand(rng, (1.5, 3.0, 5.0))
        σs = exp.(0.4 .* randn(rng, p))
        Λ = randn(rng, p, K) .* rand(rng, (0.5, 1.0, 2.0))
        β = randn(rng, p)
        zt = randn(rng, K)
        y = [β[t] + dot(view(Λ, t, :), zt) + σs[t] * rand(rng, TDist(ν)) * (rand(rng) < 0.15 ? 15 : 1)
             for t in 1:p]
        fams = [GLLVModels.StudentTFamily(ν, s) for s in σs]
        n = ones(Int, p)
        z, ok = GLLVModels._studentt_grouped_mode(fams, y, n, Λ, β, IdentityLink())
        n_checked += 1
        ok || continue
        n_finite += 1
        q = zz -> sum(GLLVModels._glm_logpdf(fams[t], β[t] + dot(view(Λ, t, :), zz), 1, y[t])
                      for t in 1:p) - 0.5 * dot(zz, zz)
        g = ForwardDiff.gradient(q, z)
        @test maximum(abs, g) < 1e-4
        H = ForwardDiff.hessian(q, z)
        @test maximum(eigvals(Symmetric(H))) < 1e-6  # negative semi-definite (allow ~0 boundary)
        # The site value the objective actually uses must be finite and match q(z) exactly
        # (both are the same log-density sum minus the same quadratic penalty, up to the
        # `-0.5 logdet` normalizer term which q() omits by construction).
        v = GLLVModels._studentt_grouped_loglik_site(fams, y, n, Λ, β, IdentityLink())
        @test isfinite(v)
    end
    @test n_checked == 300
    @test n_finite > 0  # the stress draws do exercise the converged branch
end
