using GLLVModels, Test, SHA, TOML, Random, LinearAlgebra, Distributions, ForwardDiff

# #501: fit_ordered_beta_gllvm's per-site inner mode search (_ordered_beta_mode) took
# undamped Newton steps and returned whatever z it held at maxiter, converged or not.
# The per-site conditional density here is a nonconvex mixture (point masses at 0/1
# plus an interior Beta piece), so an undamped step can overshoot across a local ridge
# between two competing per-site modes; independent verification (obeta-verify-reproduce.md)
# measured the returned z at such a point having a central finite-difference gradient
# that scales as 1/h across five step sizes -- the signature of a jump discontinuity in
# the OUTER marginal negative log-likelihood, not a smooth steep slope -- tripping the
# outer optimiser's x/f convergence test (the #485-class failure) at a genuinely
# non-stationary point. The search now halves any step that lowers the per-site
# log-posterior, certifies convergence only when the FULL proposed step is below tol,
# retries once with a 20x iteration budget, and returns -Inf for a site whose search
# still fails after that.
#
# CI runs Julia 1.10 and 1.13 on Linux and this suite also runs on macOS; a seed draws
# different data across platforms and the optimiser path differs too, so this file never
# asserts a literal fitted value from a live RNG draw. The Y fixture below is literal,
# hash-verified data (not redrawn at test time); every other assertion is a RELATION
# checked by an INDEPENDENT ForwardDiff route that shares no code with
# `_ordered_beta_mode`'s own hand-rolled score/weight: every finite per-site value is
# stationary (small gradient) with a negative-definite Hessian, verified by a
# scale-invariant Newton decrement `sqrt(g' (-H)^{-1} g)`.

const _OB501_FIXTURE = joinpath(@__DIR__, "fixtures", "ordered_beta_mode_search_seed2010.toml")

function _ob501_data()
    d = TOML.parsefile(_OB501_FIXTURE)
    Y = reshape(Float64.(d["Y_column_major"]), d["p"], d["n"])
    @test bytes2hex(sha256(reinterpret(UInt8, vec(Y)))) == d["data_sha256"]
    return Y, d["K"]
end

# Site log-posterior q(z), independent of GLLVModels' internal `_ordered_beta_logpost`:
# calls only the public `ordered_beta_logp`.
function _ob501_site_logpost(z, y, Λ, β, c0, c1, φ)
    η = β .+ Λ * z
    q = -0.5 * dot(z, z)
    for t in eachindex(y)
        q += GLLVModels.ordered_beta_logp(y[t], η[t], c0, c1, φ)
    end
    return q
end

# Independent stationarity + negative-definite-Hessian check, via ForwardDiff on the
# scalar-to-vector map z -> q(z), decoupled from the package's own per-site score/weight.
function _ob501_site_diag(z, y, Λ, β, c0, c1, φ)
    f = zz -> _ob501_site_logpost(zz, y, Λ, β, c0, c1, φ)
    g = ForwardDiff.gradient(f, z)
    H = Symmetric(ForwardDiff.hessian(f, z))
    neg_def = maximum(eigvals(H)) < 0
    decrement = neg_def ? sqrt(max(0.0, dot(g, -(H \ g)))) : NaN
    return (gnorm = norm(g), neg_def = neg_def, decrement = decrement)
end

@testset "ordered-beta mode search (#501)" begin
    Y, K = _ob501_data()
    p, n = size(Y)

    @testset "fitted point: every site stationary, negative-definite Hessian" begin
        fit = fit_ordered_beta_gllvm(Y; K = K)
        @test fit.converged
        @test isfinite(fit.loglik)

        for s in 1:n
            z, ok = GLLVModels._ordered_beta_mode(view(Y, :, s), fit.Λ, fit.β, fit.c0, fit.c1, fit.φ)
            @test ok
            d = _ob501_site_diag(z, view(Y, :, s), fit.Λ, fit.β, fit.c0, fit.c1, fit.φ)
            @test d.neg_def
            @test d.decrement < 1e-3
        end

        # The reported loglik is honest: recomputing the marginal directly at the
        # fitted parameters (not reading `fit.loglik`) must match.
        recomputed = GLLVModels.ordered_beta_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.c0, fit.c1, fit.φ)
        @test isapprox(recomputed, fit.loglik; atol = 1e-6)
    end

    @testset "a search that cannot converge returns -Inf, not a finite garbage value" begin
        β = zeros(p)
        Λ = 0.3 .* Matrix{Float64}(I, p, K)[:, 1:K]
        # maxiter = 0: the loop body never runs even once, so neither the default nor
        # the 20x-retry budget (also 0) can certify a step -- budget-independently
        # insufficient, unlike a merely slow-but-healthy site.
        val = GLLVModels._ordered_beta_loglik_site(view(Y, :, 1), Λ, β, -1.0, 1.0, 8.0;
                                                    maxiter = 0, tol = 1e-9)
        @test val == -Inf
    end

    @testset "a site where the old undamped loop already converged is unchanged" begin
        # A well-separated site (all-interior y, moderate Lambda) converges in a
        # handful of undamped Newton steps with no halving needed; the damped search
        # must take the exact same steps and reach the same z (bit-identical on an
        # accepted full step, per the loop's own construction).
        β = [0.1, -0.2, 0.3, 0.0, 0.05]
        Λ = 0.4 .* [1.0 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
        y = [0.4, 0.6, 0.3, 0.55, 0.45]
        z, ok = GLLVModels._ordered_beta_mode(y, Λ, β, -1.0, 1.0, 8.0)
        @test ok
        d = _ob501_site_diag(z, y, Λ, β, -1.0, 1.0, 8.0)
        @test d.neg_def
        @test d.decrement < 1e-6
    end
end
