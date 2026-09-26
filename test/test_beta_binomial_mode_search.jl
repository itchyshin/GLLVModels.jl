using GLLVModels, Test, Random, LinearAlgebra, ForwardDiff

# #503: `_beta_binomial_mode` (the per-site Laplace mode search shared by every
# BetaBinomial fitter/getLV/predict route) ran an undamped Newton loop on the
# clamped observed curvature (`_bb_score_weight` floors the weight to `>= 1e-8`,
# which keeps `Λ'WΛ + I` SPD by construction but does not keep any step a
# DESCENT step) and returned whatever `z` it held at `maxiter`, converged or
# not -- the same defect class fixed for Gamma (#479), NB1/Student-t/twopart
# (#500/#507/#509). The class audit (Lambda scaled up to 3x, warm start
# perturbed +-50%) measured 27/500 non-mode sites.
#
# The fix adds `_beta_binomial_mode_search`, which halves any step that lowers
# the per-site log-posterior and certifies convergence only when the FULL
# proposed step is below `tol`; `_beta_binomial_loglik_site` uses it directly
# and returns -Inf when the search cannot certify a stationary point (retried
# once at a 20x iteration budget first, mirroring #507's review fix and #509's
# fallback). `_beta_binomial_mode` (the name `getLV`/`predict` call) keeps
# returning `z` only, now via the same damped+retried search.
#
# CI runs Julia 1.10 and 1.13 on Linux, and the same seed draws different data
# across Julia versions with optimiser paths differing by platform, so this
# file asserts no specific fitted log-likelihood or iteration count from a
# seed. It asserts structural relations instead: every FINITE site value the
# fixed kernel returns sits at a genuine stationary point of the exact
# per-site log-posterior, verified independently (via ForwardDiff, sharing no
# code with the kernel) by a SCALE-INVARIANT Newton decrement
# `lambda = sqrt(g' * (-H)^{-1} * g)` (not a raw gradient-norm threshold: a
# large, ill-conditioned Hessian can make a raw gradient look small at a point
# that is still far from the mode in the natural, curvature-normalised sense
# the decrement measures), together with a negative-definite Hessian.

# Literal p=5, K=3 site (found by a stress probe over random Lambda, beta, y
# draws with Lambda scaled 3x; independently confirmed against a from-scratch
# damped search at maxiter=5000, tol=1e-12: reference mode has Newton decrement
# 5.46e-16). On `origin/main` (pre-#503) `_beta_binomial_mode`'s undamped loop
# returns a FINITE z with Newton decrement 17.2 (nowhere near the mode), and
# the pre-fix per-site log-marginal formula evaluated there is -71.1947894104373
# -- 52.9 log-posterior units below the true mode's value -- with no
# `converged` flag to catch it (this kernel had none pre-fix).
const _BB503_LAMBDA = [2.192958676797323 4.844630960287616 -0.31692001822920823
                        2.638046921495141 2.6837190473998187 0.10969017620187156
                        -3.4612431495947105 3.999767949264439 1.5035676366258794
                        -2.970421994643132 1.5172202048614616 -1.2275206750344458
                        -1.9573432691401522 -0.813428634083826 2.916839444963217]
const _BB503_BETA = [0.4786035607538619, 1.2479907935959433, 2.3528010080360087,
                      1.0993853792705917, -0.687779494389844]
const _BB503_PHI = [2.4422878841991436, 1.0995023133357014, 3.8571373608057278,
                     1.2247359710726193, 0.7925539960450333]
const _BB503_N = [27, 7, 15, 18, 18]
const _BB503_Y = [5.0, 3.0, 8.0, 16.0, 3.0]

# Verbatim pre-#503 kernel (undamped Newton on the clamped observed curvature,
# no convergence flag), kept only to demonstrate what `origin/main` returns for
# this site.
function _bb503_old_mode(y, N, Λ, β, φ; link = GLLVModels.LogitLink(), maxiter = 100, tol = 1e-9)
    G = GLLVModels
    p, K = size(Λ)
    z = zeros(K)
    for _ in 1:maxiter
        η = β .+ Λ * z
        s = Vector{Float64}(undef, p)
        W = Vector{Float64}(undef, p)
        @inbounds for t in 1:p
            st, Wt = G._bb_score_weight(y[t], η[t], N[t], G._bb_phi_at(φ, t); link = link)
            s[t] = st
            W[t] = Wt
        end
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = G._safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && break
    end
    return z
end

function _bb503_old_site_value(y, N, Λ, β, φ; link = GLLVModels.LogitLink())
    G = GLLVModels
    p = size(Λ, 1)
    z = _bb503_old_mode(y, N, Λ, β, φ; link = link)
    η = β .+ Λ * z
    W = Float64[G._bb_score_weight(y[t], η[t], N[t], G._bb_phi_at(φ, t); link = link)[2] for t in 1:p]
    A = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = sum(G.betabinomial_logp(y[t], η[t], N[t], G._bb_phi_at(φ, t); link = link) for t in 1:p)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

# Exact per-site log-posterior q(z), independent of the kernel's own internal
# `_bb_mode_logpost` (a fresh expression built from `betabinomial_logp`), used
# only to certify stationarity via the Newton decrement below.
function _bb503_logpost(y, N, Λ, β, φ, link)
    return z -> begin
        G = GLLVModels
        η = β .+ Λ * z
        q = -0.5 * dot(z, z)
        for t in eachindex(y)
            q += G.betabinomial_logp(y[t], η[t], N[t], G._bb_phi_at(φ, t); link = link)
        end
        q
    end
end

# Scale-invariant Newton decrement at z for objective q: lambda^2 = g'(-H)^-1 g,
# where g, H are the ForwardDiff gradient/Hessian of q at z. At a genuine local
# maximum, -H is positive definite and lambda measures the (curvature-
# normalised) distance from the quadratic model's own optimum -- unlike a raw
# gradient norm, this is invariant to an affine rescaling of z. Returns
# `(lambda, H)` so callers can also assert negative-definiteness of H itself.
function _newton_decrement(q, z)
    g = ForwardDiff.gradient(q, z)
    H = ForwardDiff.hessian(q, z)
    Hs = Symmetric(-H)
    λ2 = try
        dot(g, Hs \ g)
    catch
        Inf
    end
    λ = (λ2 < 0 || !isfinite(λ2)) ? Inf : sqrt(λ2)
    return λ, H
end

@testset "BetaBinomial mode search (#503)" begin
    link = GLLVModels.LogitLink()

    @testset "pre-#503 kernel returns a finite, wrong value on this site" begin
        q = _bb503_logpost(_BB503_Y, _BB503_N, _BB503_LAMBDA, _BB503_BETA, _BB503_PHI, link)
        z_old = _bb503_old_mode(_BB503_Y, _BB503_N, _BB503_LAMBDA, _BB503_BETA, _BB503_PHI; link = link)
        λ_old, _ = _newton_decrement(q, z_old)
        @test λ_old > 1.0  # nowhere near stationary (measured 17.2)
        v_old = _bb503_old_site_value(_BB503_Y, _BB503_N, _BB503_LAMBDA, _BB503_BETA, _BB503_PHI; link = link)
        @test isfinite(v_old)
        @test isapprox(v_old, -71.1947894104373; atol = 1e-6)
    end

    @testset "fixed kernel finds the true stationary point on this site" begin
        z_new, ok = GLLVModels._beta_binomial_mode_search(_BB503_Y, _BB503_N, _BB503_LAMBDA,
                                                           _BB503_BETA, _BB503_PHI; link = link)
        @test ok
        q = _bb503_logpost(_BB503_Y, _BB503_N, _BB503_LAMBDA, _BB503_BETA, _BB503_PHI, link)
        λ_new, H_new = _newton_decrement(q, z_new)
        @test λ_new < 1e-6
        @test maximum(eigvals(Symmetric(H_new))) < 1e-6  # negative (semi-)definite

        v_new = GLLVModels._beta_binomial_loglik_site(_BB503_Y, _BB503_N, _BB503_LAMBDA,
                                                       _BB503_BETA, _BB503_PHI; link = link)
        @test isfinite(v_new)
        @test v_new > -71.1947894104373 + 1.0  # strictly, and by a wide margin, above the old value
        @test isapprox(v_new, -18.31571576277004; atol = 1e-6)
    end

    @testset "a search that cannot converge returns -Inf, not a finite garbage value" begin
        # maxiter = 0 (and its 20x retry, 0) cannot take a single step; the
        # loop's `for _ in 1:0` body never runs, so it returns the z=0 start
        # unconverged.
        z0, ok0 = GLLVModels._beta_binomial_mode_search(_BB503_Y, _BB503_N, _BB503_LAMBDA,
                                                        _BB503_BETA, _BB503_PHI; link = link, maxiter = 0)
        @test !ok0
        v0 = GLLVModels._beta_binomial_loglik_site(_BB503_Y, _BB503_N, _BB503_LAMBDA,
                                                    _BB503_BETA, _BB503_PHI; link = link, maxiter = 0)
        @test v0 == -Inf
    end
end

@testset "BetaBinomial mode search: stress-probe stationarity (#503)" begin
    # Independent of the literal case above: draw many random per-site
    # problems (Lambda scaled up to 3x, fixed seed so deterministic within one
    # Julia process but asserting no cross-version numeric equality) and check
    # that whenever `_beta_binomial_mode_search` reports `ok = true`, the site
    # is genuinely at a stationary point of the exact log-posterior (small
    # scale-invariant Newton decrement) with a negative-(semi-)definite
    # Hessian -- never a step-size-only false positive.
    rng = MersenneTwister(20260926)
    link = GLLVModels.LogitLink()
    n_trials = 220
    n_checked_ok = 0
    n_finite = 0

    for _ in 1:n_trials
        p = rand(rng, 4:12)
        K = rand(rng, 1:3)
        scale = rand(rng, (1.0, 2.0, 3.0))
        Λ = scale .* randn(rng, p, K)
        β = randn(rng, p) .* 1.2
        φ = exp.(randn(rng, p) .* 1.0)
        N = rand(rng, 5:40, p)
        ztrue = randn(rng, K)
        η0 = β .+ Λ * ztrue
        μ0 = clamp.(GLLVModels.linkinv.(Ref(link), η0), 1e-6, 1 - 1e-6)
        y = Float64[clamp(round(N[t] * μ0[t] + N[t] * 0.15 * randn(rng)), 0, N[t]) for t in 1:p]

        z, ok = GLLVModels._beta_binomial_mode_search(y, N, Λ, β, φ; link = link)
        n_checked_ok += ok ? 1 : 0
        ok || continue

        q = _bb503_logpost(y, N, Λ, β, φ, link)
        λ, H = _newton_decrement(q, z)
        @test λ < 1e-3
        @test maximum(eigvals(Symmetric(H))) < 1e-6

        v = GLLVModels._beta_binomial_loglik_site(y, N, Λ, β, φ; link = link)
        @test isfinite(v)
        n_finite += 1
    end
    @test n_checked_ok > 0.9 * n_trials  # the damped search converges at almost every draw
    @test n_finite == n_checked_ok       # every converged site yields a finite site value
end
