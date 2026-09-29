# #504, zero-truncated count migration (after Poisson #516, Gamma, Ordinal, the
# zero-inflated and the hurdle/delta slices): the bootstrap `refit` closures in
# `src/confint_family.jl` (`_family_ci` for TruncatedPoissonFit,
# TruncatedNegBin2Fit and TruncatedNegBin2PerTraitFit) returned a bare parameter
# vector, so `_bootstrap_refit_ok` (#508) could only check `isfinite`. A refit
# that ended on the fitter's failure verdict (θ is the finite warm start) was
# counted as a good replicate. They now report `(θ = ..., converged = ...,
# loglik = ...)`.
#
# No `upper_boundary` flag (the #542 option-3 field) is set: this slice only
# reports the refit's verdict. Whether the truncated-NB2 size `r` needs such a
# flag (as NB2 `r` does) is left open and is not tested here.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

function _sim_trunc_verdict(kind::Symbol, p::Integer, n::Integer; seed::Integer,
                            r::Real = 3.0)
    rng = MersenneTwister(seed)
    β = 0.3 .* randn(rng, p) .+ 1.0
    Λ = 0.5 .* randn(rng, p, 1)
    rt = kind === :nb2pt ? [2.0 + t for t in 0:(p - 1)] : fill(float(r), p)
    Y = zeros(Int, p, n)
    for s in 1:n
        η = β .+ Λ * randn(rng, 1)
        for t in 1:p
            μ = exp(η[t])
            Y[t, s] = kind === :pois ? GM._rand_ztpois(rng, μ) : GM._rand_ztnb(rng, rt[t], μ)
        end
    end
    return Y
end

function _trunc504_routes(Yp, Yn, Ynt)
    f0 = GM.fit_truncated_poisson_gllvm(Yp; K = 1)
    f1 = GM.fit_truncated_nbinom2_gllvm(Yn; K = 1)
    f2 = GM.fit_truncated_nbinom2_gllvm_pertrait(Ynt; K = 1)
    return (
        (:truncated_poisson, f0, Yp, GM._family_ci(f0, Yp),
         Yb_ -> GM.fit_truncated_poisson_gllvm(Yb_; K = 1),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ))),
        (:truncated_nbinom2, f1, Yn, GM._family_ci(f1, Yn),
         Yb_ -> GM.fit_truncated_nbinom2_gllvm(Yb_; K = 1),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.r))),
        (:truncated_nbinom2_pertrait, f2, Ynt, GM._family_ci(f2, Ynt),
         Yb_ -> GM.fit_truncated_nbinom2_gllvm_pertrait(Yb_; K = 1),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log.(fb.r))),
    )
end

# The truncated-Poisson route has a real failing draw: one cell holding the
# integer count 10^18 makes the fitter stop after one iteration with its
# negative log-likelihood at or above the 1e11 failure threshold, so
# `_fit_verdict` reports `converged = false`, `loglik = -Inf`, with a finite θ
# and no throw, with or without bounds checking. Data-driven, not RNG-driven.
#
# The two truncated-NB2 routes have no such draw, so their rejection and
# end-to-end testsets use a STUB failed refit, labelled as such below; the
# adapter-parity testset is what pins those migrated closures. A probe on Julia
# 1.10.12 swept 0, a negative value, 0.5, NaN, Inf and 1e300 (the fitter throws
# an `ArgumentError` or `InexactError` before optimising, so the refit returns
# `nothing`, which main already rejects), counts of 10^6 to 10^13 (the fit
# converges), and counts of 10^14 to 10^18 in one, two or four cells. The large
# counts do end on the failure verdict, but the size estimate usually underflows
# to `r = 0`, so θ holds `log(0) = -Inf` and the old contract already rejects
# it; the few placements that kept θ finite on one Julia version did not on the
# other (1.10.12 against 1.13.0).
function _trunc504_bad(Y)
    B = copy(Y)
    B[1, 1] = 10^18
    return B
end

_trunc504_real_bad(route::Symbol) = route === :truncated_poisson

# STUB (not a real fit): what a refit ending on the fitter's failure verdict
# reports, a finite θ with `converged = false` and `loglik = -Inf`.
_trunc504_stub_failed(ad) = (θ = copy(ad.θ), converged = false, loglik = -Inf)

@testset "Zero-truncated bootstrap refit reports the fitter's own verdict (#504)" begin
    p, n = 4, 120
    Yp = _sim_trunc_verdict(:pois, p, n; seed = 91)
    Yn = _sim_trunc_verdict(:nb2, p, n; seed = 92)
    Ynt = _sim_trunc_verdict(:nb2pt, p, n; seed = 93)
    routes = _trunc504_routes(Yp, Yn, Ynt)

    @testset "adapter parity with a direct refit ($route)" for (route, fit, Y, ad, direct, pack) in routes
        @test fit.converged   # sanity: a healthy fit
        Ydraw = ad.simulate(MersenneTwister(504))
        raw = ad.refit(Ydraw)
        @test raw isa NamedTuple   # a bare vector on main: this is where main fails
        fb = direct(Ydraw)
        @test raw.converged == fb.converged
        @test raw.loglik == fb.loglik
        @test raw.θ == pack(fb)
        @test !haskey(raw, :upper_boundary)   # deliberately not flagged (see header)
    end

    real_routes = filter(r -> _trunc504_real_bad(r[1]), routes)
    stub_routes = filter(r -> !_trunc504_real_bad(r[1]), routes)

    @testset "a failed refit is rejected, real draw ($route)" for (route, fit, Y, ad, direct, pack) in real_routes
        Y_bad = _trunc504_bad(Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        m = length(ad.θ)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "a failed refit (stub) is rejected ($route)" for (route, fit, Y, ad, direct, pack) in stub_routes
        raw_bad = _trunc504_stub_failed(ad)
        m = length(ad.θ)
        @test all(isfinite, raw_bad.θ)
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, Y, ad, direct, pack) in routes
        m = length(ad.θ)
        ad_mixed = if _trunc504_real_bad(route)
            Y_bad = _trunc504_bad(Y)
            GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                         rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        else
            bad_marker = fill(-1, size(Y))   # STUB: a sentinel draw, never passed to a fitter
            GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                         rng -> rand(rng) < 0.5 ? Y : bad_marker,
                         Yb_ -> Yb_ === bad_marker ? _trunc504_stub_failed(ad) : ad.refit(Yb_))
        end
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, Y, ad, direct, pack = routes[1]   # truncated Poisson: the fastest route
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
