using GLLVModels, Test, Random, LinearAlgebra, Distributions, ForwardDiff

# #503: `_compoisson_mode` (the CMP family's own per-site Laplace mode search,
# `src/families/com_poisson.jl`) ran an undamped Newton step every iteration and
# returned whatever `z` it held at `maxiter`, converged or not — the same defect
# class as #479/#480/#484/#500/#507/#509. `W` is always clamped to `>= 1e-8`
# (`_cmp_score_weight`), so `A = Λ'WΛ + I` is SPD by construction and every Newton
# step solves in a well-defined direction, but that only floors the curvature used
# for the step; it never certified the step as a descent step on the true per-site
# log-posterior. The fix halves any step that lowers the log-posterior and reports
# `-Inf` (via `_compoisson_loglik_site`) when it cannot certify a stationary point,
# so the fitter's own 1e12 sentinel fires instead of a garbage finite value
# propagating into L-BFGS.
#
# This file does NOT assert a specific fitted log-likelihood, iteration count, or
# seed-tied numeric outcome (CI runs Julia 1.10 and 1.13 on Linux; the same seed
# draws different data across Julia versions, and optimiser paths differ across
# platforms on the same data). It asserts structural relations instead: every
# FINITE value the fixed search reports sits at a genuine stationary point (a
# scale-invariant Newton decrement near zero) with a negative-definite Hessian of
# the per-site log-posterior, checked independently via `ForwardDiff` on the
# public `compoisson_logpdf`, and the old (undamped, pre-#503) loop is reproduced
# verbatim below only to confirm the fix leaves its converged cases unchanged.

# Verbatim pre-#503 kernel (undamped Newton, no convergence flag), kept only to
# demonstrate what the old loop returns and to compare against cases where it
# happens to converge.
function _cmp503_old_site(y, Λ, β, ν; maxiter = 100, tol = 1e-9)
    G = GLLVModels
    p, K = size(Λ)
    z = zeros(K)
    conv = false
    for _ in 1:maxiter
        η = β .+ Λ * z
        s = Vector{Float64}(undef, p)
        W = Vector{Float64}(undef, p)
        for t in 1:p
            st, Wt = G._cmp_score_weight(y[t], η[t], ν)
            s[t] = st
            W[t] = Wt
        end
        Δ = G._safe_solve(Symmetric(Λ' * (W .* Λ) + I), Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && (conv = true; break)
    end
    η = β .+ Λ * z
    ℓ = sum(G.compoisson_logpdf(y[t], η[t], ν) for t in 1:p)
    W = Vector{Float64}(undef, p)
    for t in 1:p
        _, W[t] = G._cmp_score_weight(y[t], η[t], ν)
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A), conv
end

# Independent per-site log-posterior, built only from the PUBLIC `compoisson_logpdf`
# (shares no code with `_compoisson_mode`'s internal `_cmp_score_weight` /
# `_compoisson_mode_logpost`), so `ForwardDiff` through it is a from-scratch check
# that a claimed mode is genuinely stationary.
_cmp503_q(y, Λ, β, ν) = zz -> sum(
    GLLVModels.compoisson_logpdf(y[t], β[t] + dot(view(Λ, t, :), zz), ν) for t in eachindex(y)
) - 0.5 * dot(zz, zz)

@testset "COM-Poisson mode search: stress-probe stationarity (#503)" begin
    rng = MersenneTwister(90503)
    n_checked = 0
    n_finite = 0
    for _ in 1:300
        p = rand(rng, 3:8)
        K = rand(rng, 1:2)
        ν = rand(rng, (0.6, 1.0, 1.5, 2.5))
        Λ = randn(rng, p, K) .* rand(rng, (0.5, 1.0, 2.0))
        β = randn(rng, p) .* 0.4 .+ 1.0
        zt = randn(rng, K)
        y = [Float64(rand(rng, Poisson(exp(clamp(β[t] + dot(view(Λ, t, :), zt), -10.0, 10.0)))))
             for t in 1:p]

        z, ok = GLLVModels._compoisson_mode(y, Λ, β, ν)
        n_checked += 1
        ok || continue
        n_finite += 1

        q = _cmp503_q(y, Λ, β, ν)
        g = ForwardDiff.gradient(q, z)
        H = ForwardDiff.hessian(q, z)
        negH = Symmetric(-H)
        @test isposdef(negH)
        @test 0.5 * dot(g, negH \ g) < 1e-6   # scale-invariant Newton decrement

        v = GLLVModels._compoisson_loglik_site(y, Λ, β, ν)
        @test isfinite(v)
    end
    @test n_checked == 300
    @test n_finite > 0   # the stress draws do exercise the converged branch
end

@testset "COM-Poisson mode search: a search that cannot converge returns -Inf" begin
    rng = MersenneTwister(90504)
    p, K = 5, 2
    Λ = randn(rng, p, K)
    β = randn(rng, p) .* 0.4 .+ 1.0
    y = Float64.(rand(rng, 0:10, p))
    # A zero-iteration budget cannot reach any tolerance from z = 0, and the 20x
    # retry (20 * 0 = 0) does not change that, so this is a genuine,
    # budget-independent non-convergence case.
    @test GLLVModels._compoisson_loglik_site(y, Λ, β, 1.2; maxiter = 0) == -Inf
    Y = reshape(y, p, 1)
    @test GLLVModels.compoisson_marginal_loglik_laplace(Y, Λ, β, 1.2; maxiter = 0) == -Inf
end

@testset "COM-Poisson mode search: where the old undamped loop converged, the value is unchanged" begin
    # Small, well-behaved literal points chosen so the pre-#503 undamped loop
    # actually converges, to check the damped fix leaves those cases alone (small
    # steps and accepted full steps are bit-identical to the old loop by
    # construction).
    well_behaved = [
        (reshape([0.3, -0.2, 0.4, 0.1], 4, 1), [0.5, 0.2, -0.1, 0.3], 1.0,
         [1.0, 2.0, 0.0, 3.0]),
        (reshape([0.2, 0.15, -0.3, 0.25, -0.1], 5, 1), [0.1, -0.2, 0.3, 0.0, 0.2], 1.5,
         [1.0, 0.0, 3.0, 1.0, 2.0]),
        (reshape([0.1, -0.15, 0.2, 0.05], 4, 1), [-0.2, 0.4, 0.1, -0.3], 0.7,
         [0.0, 2.0, 1.0, 3.0]),
    ]
    n_compared = 0
    worst = 0.0
    for (Λ, β, ν, y) in well_behaved
        old, conv = _cmp503_old_site(y, Λ, β, ν)
        @test conv   # these points are chosen precisely so the old loop converges
        conv || continue
        new = GLLVModels._compoisson_loglik_site(y, Λ, β, ν)
        worst = max(worst, abs(new - old))
        n_compared += 1
    end
    @test n_compared >= 3
    @test worst < 1e-8
end
