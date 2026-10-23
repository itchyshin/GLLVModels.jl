using GLLVModels, Test, LinearAlgebra, TOML, SHA

# `_grouped_laplace_mode` must converge to the per-site mode, not stop wherever its
# iteration count runs out. Fisher scoring overshoots where the observed curvature is
# about twice the Fisher weight or more (here y = 121 at μ ≈ 27.5), and the small-step
# path skipped every safeguard, so the iterates 2-cycled around the mode. Where they
# stopped jumped with the parameters: a 1e-5 step in log r moved the zero-truncated
# NB2 Laplace objective by 5.5e-4 one way and 1.5e-8 the other.

const _TNB2_MS = joinpath(@__DIR__, "fixtures", "truncnb2_mode_search_oscillation.toml")

# Closed-form zero-truncated NB2/log score in η (independent of the package kernels):
# d/dη [log NB2(y; μ, r) − log(1 − p0)], p0 = (1 + μ/r)^(−r).
function _tnb2_score_eta(y, μ, r)
    p0 = exp(-r * log1p(μ / r))
    return r / (r + μ) * ((y - μ) - μ * p0 / (-expm1(-r * log1p(μ / r))))
end
_tnb2_grad(y, λ, β, r, z) =
    sum(λ[t] * _tnb2_score_eta(y[t], exp(β[t] + λ[t] * z), r) for t in eachindex(y)) - z

# Reference mode (K = 1) by bisection on the closed-form gradient.
function _tnb2_ref_mode(y, λ, β, r; lo = -3.0, hi = 0.0)
    @assert _tnb2_grad(y, λ, β, r, lo) > 0 && _tnb2_grad(y, λ, β, r, hi) < 0
    for _ in 1:200
        mid = (lo + hi) / 2
        _tnb2_grad(y, λ, β, r, mid) > 0 ? (lo = mid) : (hi = mid)
        hi - lo < 1e-15 && break
    end
    return (lo + hi) / 2
end

@testset "truncated NB2: grouped mode search converges at the 2-cycle site" begin
    d = TOML.parsefile(_TNB2_MS)
    p, n = d["p"], d["n"]
    Y = reshape(Int.(d["Y_column_major"]), p, n)
    @test bytes2hex(sha256(join(string.(vec(Y)), ","))) == d["Y_sha256"]
    θ = Float64.(d["theta_packed"])
    β = θ[1:p]; λ = θ[(p + 1):(2p)]; logr = θ[end]
    y = Y[:, d["site"]]
    @test y == [121, 1, 1, 2]
    for dl in (0.0, 1e-5, -1e-5)
        r = exp(logr + dl)
        zref = _tnb2_ref_mode(y, λ, β, r)
        @test abs(_tnb2_grad(y, λ, β, r, zref)) < 1e-10
        z = GLLVModels._grouped_laplace_mode(fill(TruncatedNegBin2(r), p), y, ones(Int, p),
                                             reshape(λ, p, 1), β, LogLink())[1]
        # Before the fix: 8e-6 (dl = 0) and 1.3e-3 (dl = +1e-5) away from the mode.
        @test abs(z - zref) < 1e-9
    end
end

@testset "truncated NB2: grouped mode search converges on a large-step 2-cycle" begin
    # genTNB(MersenneTwister(104), 4, 150), site 98, at a profile-refit point for r
    # (log r = -0.605561). Steps of about 0.15 overshoot by just under 2×, so each
    # one still raises the log-posterior and the old ascent test accepted it: after
    # 100 iterations z was 4.5e-3 off the mode, and the r profile's refits failed
    # with a finite-difference gradient of 36.
    θ = [0.8678285463054805, 0.8194055155119475, 0.3219143147081221, 1.6449882265791687,
         -0.313295521566255, 0.9724740050184992, -0.2697100802853465, 0.34046635510884227,
         -0.605561]
    y = [2, 64, 1, 3]; p = 4
    β = θ[1:p]; λ = θ[(p + 1):(2p)]; r = exp(θ[end])
    zref = _tnb2_ref_mode(y, λ, β, r; lo = 1.0, hi = 2.5)
    @test abs(_tnb2_grad(y, λ, β, r, zref)) < 1e-10
    z = GLLVModels._grouped_laplace_mode(fill(TruncatedNegBin2(r), p), y, ones(Int, p),
                                         reshape(λ, p, 1), β, LogLink())[1]
    @test abs(z - zref) < 1e-9
end

@testset "truncated NB2: Laplace objective is smooth in log r at the fitted optimum" begin
    d = TOML.parsefile(_TNB2_MS)
    p, n = d["p"], d["n"]
    Y = reshape(Int.(d["Y_column_major"]), p, n)
    θ = Float64.(d["theta_packed"])
    β = θ[1:p]; Λ = reshape(θ[(p + 1):(2p)], p, 1); logr = θ[end]
    f(dl) = truncated_nbinom2_marginal_loglik_laplace(Y, Λ, β, exp(logr + dl))
    h = 1e-5
    f0, fp, fm = f(0.0), f(h), f(-h)
    @test all(isfinite, (f0, fp, fm))
    # Both one-sided moves are small and of opposite sign, as for a smooth curve near
    # its optimum (measured +2.1e-8 and -2.7e-8); before the fix they were -5.5e-4
    # and -1.5e-8.
    @test abs(fp - f0) < 1e-7
    @test abs(fm - f0) < 1e-7
    # The second difference over a grid is that of a smooth curve: no jumps.
    grid = [f(k * 1e-4) for k in -10:10]
    # Measured 6.4e-7 after the fix (curvature ≈ 64 at step 1e-4), 5.5e-4 before it.
    @test maximum(abs, diff(diff(grid))) < 1e-5
end
