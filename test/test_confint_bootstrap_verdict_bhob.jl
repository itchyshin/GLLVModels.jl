# #504, beta-hurdle and ordered-beta migration (after Poisson #516 and the
# single-family slices): the bootstrap `refit` closures in
# `src/confint_family.jl` (`_family_ci` for BetaHurdleFit and OrderedBetaFit)
# returned a bare parameter vector, so `_bootstrap_refit_ok` (#508) could only
# check `isfinite`. A refit that ended on the fitter's failure verdict (θ is the
# finite warm start) was counted as a good replicate. They now report
# `(θ = ..., converged = ..., loglik = ...)`.
#
# No `upper_boundary` flag (the #542 option-3 field) is set: this slice only
# reports the refit's verdict. Whether the Beta precision φ needs such a flag is
# left open and is not tested here.
#
# The ordered-beta adapter's `simulate` is a stub that errors (bootstrap is not
# offered for that family), so its parity draw is a separately simulated data
# set, and its end-to-end testset supplies its own simulator. The migrated
# `refit` is still what `_family_bootstrap` calls.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

function _sim_bhob_verdict(kind::Symbol, p::Integer, n::Integer; seed::Integer)
    rng = MersenneTwister(seed)
    β = 0.3 .* randn(rng, p); βz = 0.4 .* randn(rng, p) .+ 0.5
    Λ = 0.4 .* randn(rng, p, 1); φ = 10.0; c0, c1 = -1.0, 1.0
    Y = zeros(Float64, p, n)
    for s in 1:n
        η = β .+ Λ * randn(rng, 1)
        for t in 1:p
            μ = inv(1 + exp(-η[t]))
            if kind === :beta_hurdle
                if rand(rng) < inv(1 + exp(-βz[t]))
                    Y[t, s] = rand(rng, Beta(μ * φ, (1 - μ) * φ))
                end
            else
                r = rand(rng)
                p0 = 1 - inv(1 + exp(-(η[t] - c0))); p1 = inv(1 + exp(-(η[t] - c1)))
                Y[t, s] = r < p0 ? 0.0 : r > 1 - p1 ? 1.0 :
                          clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
            end
        end
    end
    return Y
end

function _bhob504_routes(Yb, Yo, p, n)
    f0 = GM.fit_beta_hurdle_gllvm(Yb; K = 1)
    f1 = GM.fit_ordered_beta_gllvm(Yo; K = 1)
    ad0 = GM._family_ci(f0, Yb)
    ad1 = GM._family_ci(f1, Yo)
    return (
        (:beta_hurdle, f0, Yb, ad0, ad0.simulate(MersenneTwister(504)),
         Y_ -> GM.fit_beta_hurdle_gllvm(Y_; K = 1),
         fb -> vcat(fb.βz, fb.βc, GM.pack_lambda(fb.Λc), log(fb.φ))),
        (:ordered_beta, f1, Yo, ad1, _sim_bhob_verdict(:ordered_beta, p, n; seed = 504),
         Y_ -> GM.fit_ordered_beta_gllvm(Y_; K = 1),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), fb.c0, fb.c1, log(fb.φ))),
    )
end

# A draw no fit on the route can evaluate, so the fitter ends on its failure
# verdict (`converged = false`, `loglik = -Inf`, finite θ, no throw), with or
# without bounds checking, on Julia 1.10 and 1.13. Data-driven, not RNG-driven.
#   * Ordered-beta: one response of -1 (outside [0, 1]; the interior Beta
#     log-density is -Inf at every θ).
#
# The beta-hurdle route has no such draw, so its rejection and end-to-end
# testsets use a STUB failed refit, labelled as such below; the adapter-parity
# testset is what pins that migrated closure. The fitter scores `y > 0` as the
# positive part and clamps it to (1e-6, 1 - 1e-6), and everything else as an
# absence: a probe on Julia 1.10.12 (with and without bounds checking) and
# 1.13.0 found that one cell of 0, 1, -1, 1.5, 0.5, 1e300, NaN, Inf, -Inf or
# 1e-300 all fit with `converged = true` and a finite log-likelihood.
_bhob504_bad(Y) = (B = copy(Y); B[1, 1] = -1.0; B)

_bhob504_real_bad(route::Symbol) = route === :ordered_beta

# STUB (not a real fit): what a refit ending on the fitter's failure verdict
# reports, a finite θ with `converged = false` and `loglik = -Inf`.
_bhob504_stub_failed(ad) = (θ = copy(ad.θ), converged = false, loglik = -Inf)

@testset "Beta-hurdle and ordered-beta bootstrap refit reports the fitter's own verdict (#504)" begin
    p, n = 4, 120
    Yb = _sim_bhob_verdict(:beta_hurdle, p, n; seed = 91)
    Yo = _sim_bhob_verdict(:ordered_beta, p, n; seed = 92)
    routes = _bhob504_routes(Yb, Yo, p, n)

    @testset "adapter parity with a direct refit ($route)" for (route, fit, Y, ad, Ydraw, direct, pack) in routes
        @test fit.converged   # sanity: a healthy fit
        raw = ad.refit(Ydraw)
        @test raw isa NamedTuple   # a bare vector on main: this is where main fails
        fb = direct(Ydraw)
        @test raw.converged == fb.converged
        @test raw.loglik == fb.loglik
        @test raw.θ == pack(fb)
        @test !haskey(raw, :upper_boundary)   # deliberately not flagged (see header)
    end

    real_routes = filter(r -> _bhob504_real_bad(r[1]), routes)
    stub_routes = filter(r -> !_bhob504_real_bad(r[1]), routes)

    @testset "a failed refit is rejected, real draw ($route)" for (route, fit, Y, ad, Ydraw, direct, pack) in real_routes
        Y_bad = _bhob504_bad(Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        m = length(ad.θ)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test all(isfinite, raw_bad.θ)                   # the whole θ is the finite warm start
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "a failed refit (stub) is rejected ($route)" for (route, fit, Y, ad, Ydraw, direct, pack) in stub_routes
        raw_bad = _bhob504_stub_failed(ad)
        m = length(ad.θ)
        @test all(isfinite, raw_bad.θ)
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, Y, ad, Ydraw, direct, pack) in routes
        m = length(ad.θ)
        ad_mixed = if _bhob504_real_bad(route)
            Y_bad = _bhob504_bad(Y)
            GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                         rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        else
            bad_marker = fill(-1.0, size(Y))   # STUB: a sentinel draw, never passed to a fitter
            GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                         rng -> rand(rng) < 0.5 ? Y : bad_marker,
                         Yb_ -> Yb_ === bad_marker ? _bhob504_stub_failed(ad) : ad.refit(Yb_))
        end
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, Y, ad, Ydraw, direct, pack = routes[1]   # beta-hurdle: the fastest route with a real simulator
        m = length(ad.θ)
        ad_bare = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds, ad.simulate,
                               Yb_ -> (r = ad.refit(Yb_); r === nothing ? nothing : r.θ))
        # `_family_bootstrap` reports NaN bounds below 10 usable replicates.
        n_boot = 10
        new = GM._family_bootstrap(ad, collect(1:m), 0.95, n_boot, 11, false)
        old = GM._family_bootstrap(ad_bare, collect(1:m), 0.95, n_boot, 11, false)
        @test new.n_converged == n_boot && old.n_converged == n_boot
        @test all(isfinite, new.lower) && all(isfinite, new.upper)   # non-vacuous
        @test isequal(new.lower, old.lower) && isequal(new.upper, old.upper)
    end
end
