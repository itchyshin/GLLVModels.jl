# #504, single-type continuous and count migration (after Poisson #516, Gamma,
# Ordinal, the zero-inflated, hurdle/delta and zero-truncated slices): the
# bootstrap `refit` closures in `src/confint_family.jl` (`_family_ci` for
# ExponentialFit, LognormalFit, StudentTFit and GP1Fit) returned a bare
# parameter vector, so `_bootstrap_refit_ok` (#508) could only check `isfinite`.
# A refit that ended on the fitter's failure verdict (θ is the finite warm
# start) was counted as a good replicate. They now report `(θ = ...,
# converged = ..., loglik = ...)`.
#
# No `upper_boundary` flag (the #542 option-3 field) is set: this slice only
# reports the refit's verdict. Whether the GP-1 dispersion α (capped by
# `α_bound`) or the Student-t σ needs such a flag is left open and is not
# tested here.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

function _sim_misc_verdict(kind::Symbol, p::Integer, n::Integer; seed::Integer)
    rng = MersenneTwister(seed)
    β = kind === :gp1 ? 0.3 .* randn(rng, p) .+ 1.0 :
        kind === :studentt ? 0.5 .* randn(rng, p) : 0.3 .* randn(rng, p)
    Λ = 0.5 .* randn(rng, p, 1)
    Y = kind === :gp1 ? zeros(Int, p, n) : zeros(Float64, p, n)
    fam = GM.GeneralizedPoisson1(0.1)
    for s in 1:n
        η = β .+ Λ * randn(rng, 1)
        for t in 1:p
            Y[t, s] = kind === :exponential ? rand(rng, Exponential(exp(η[t]))) :
                      kind === :lognormal ? exp(η[t] + 0.5 * randn(rng)) :
                      kind === :studentt ? η[t] + 0.7 * rand(rng, TDist(4.0)) :
                      GM._rand_gp1(rng, fam, exp(η[t]))
        end
    end
    return Y
end

const _MISC504_NU = 4.0

function _misc504_routes(Ye, Yl, Yt, Yg)
    f0 = GM.fit_exponential_gllvm(Ye; K = 1)
    f1 = GM.fit_lognormal_gllvm(Yl; K = 1)
    f2 = GM.fit_studentt_gllvm(Yt; K = 1, nu = _MISC504_NU)
    f3 = GM.fit_studentt_gllvm(Yt; K = 1, nu = _MISC504_NU, disp_group = :species)
    f4 = GM.fit_gp1_gllvm(Yg; K = 1)
    return (
        (:exponential, f0, Ye, GM._family_ci(f0, Ye),
         Yb_ -> GM.fit_exponential_gllvm(Yb_; K = 1),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ))),
        (:lognormal, f1, Yl, GM._family_ci(f1, Yl),
         Yb_ -> GM.fit_lognormal_gllvm(Yb_; K = 1),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.σ))),
        (:studentt, f2, Yt, GM._family_ci(f2, Yt),
         Yb_ -> GM.fit_studentt_gllvm(Yb_; K = 1, nu = _MISC504_NU),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.σ))),
        (:studentt_species, f3, Yt, GM._family_ci(f3, Yt),
         Yb_ -> GM.fit_studentt_gllvm(Yb_; K = 1, nu = _MISC504_NU, disp_group = :species),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log.(fb.σ))),
        (:gp1, f4, Yg, GM._family_ci(f4, Yg),
         Yb_ -> GM.fit_gp1_gllvm(Yb_; K = 1),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), fb.α)),
    )
end

# A draw no fit on the route can evaluate, so the fitter ends on its failure
# verdict (`converged = false`, `loglik = -Inf`, finite θ, no throw), with or
# without bounds checking, on Julia 1.10 and 1.13. Data-driven, not RNG-driven.
#   * Exponential: one negative response (log-density -Inf at every θ).
#   * Student-t (shared and per-species σ): one response of 1e300 (the squared
#     residual overflows, so every evaluation is non-finite).
#   * GP-1: one count of -1 (log-pmf -Inf at every θ).
#
# The lognormal route has no such draw, so its rejection and end-to-end
# testsets use a STUB failed refit, labelled as such below; the adapter-parity
# testset is what pins that migrated closure. The fitter is closed form (it
# reuses the Gaussian profile fit on log y): a probe on Julia 1.10.12 found 0,
# a negative value, NaN, Inf and -Inf throw an `ArgumentError` before fitting
# (the refit returns `nothing`, which main already rejects), and 0.5, 1e-300,
# 1e18, 1e150 and 1e300 all fit with `converged = true`.
function _misc504_bad(route::Symbol, Y)
    B = copy(Y)
    if route === :exponential
        B[1, 1] = -1.0
    elseif route === :studentt || route === :studentt_species
        B[1, 1] = 1e300
    else
        B[1, 1] = -1
    end
    return B
end

_misc504_real_bad(route::Symbol) = route !== :lognormal

# STUB (not a real fit): what a refit ending on the fitter's failure verdict
# reports, a finite θ with `converged = false` and `loglik = -Inf`.
_misc504_stub_failed(ad) = (θ = copy(ad.θ), converged = false, loglik = -Inf)

@testset "Exponential, lognormal, Student-t and GP-1 bootstrap refit reports the fitter's own verdict (#504)" begin
    p, n = 4, 120
    Ye = _sim_misc_verdict(:exponential, p, n; seed = 81)
    Yl = _sim_misc_verdict(:lognormal, p, n; seed = 82)
    Yt = _sim_misc_verdict(:studentt, p, n; seed = 83)
    Yg = _sim_misc_verdict(:gp1, p, n; seed = 84)
    routes = _misc504_routes(Ye, Yl, Yt, Yg)

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

    real_routes = filter(r -> _misc504_real_bad(r[1]), routes)
    stub_routes = filter(r -> !_misc504_real_bad(r[1]), routes)

    @testset "a failed refit is rejected, real draw ($route)" for (route, fit, Y, ad, direct, pack) in real_routes
        Y_bad = _misc504_bad(route, Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        m = length(ad.θ)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "a failed refit (stub) is rejected ($route)" for (route, fit, Y, ad, direct, pack) in stub_routes
        raw_bad = _misc504_stub_failed(ad)
        m = length(ad.θ)
        @test all(isfinite, raw_bad.θ)
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, Y, ad, direct, pack) in routes
        m = length(ad.θ)
        ad_mixed = if _misc504_real_bad(route)
            Y_bad = _misc504_bad(route, Y)
            GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                         rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        else
            bad_marker = fill(-1.0, size(Y))   # STUB: a sentinel draw, never passed to a fitter
            GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                         rng -> rand(rng) < 0.5 ? Y : bad_marker,
                         Yb_ -> Yb_ === bad_marker ? _misc504_stub_failed(ad) : ad.refit(Yb_))
        end
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, Y, ad, direct, pack = routes[2]   # lognormal (closed form): the fastest route
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
