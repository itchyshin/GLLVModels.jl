# #504, hurdle and delta migration (after Poisson #516, Gamma, Ordinal and the
# zero-inflated slice): the bootstrap `refit` closures in `src/confint_family.jl`
# (`_family_ci` for HurdlePoissonFit, HurdleNBFit, DeltaLogNormalFit and
# DeltaGammaFit; the two delta methods each hold one closure for
# `predictor = :shared` and one for the default two-predictor model) returned a
# bare parameter vector, so `_bootstrap_refit_ok` (#508) could only check
# `isfinite`. A refit that ended on the fitter's failure sentinel (θ is the finite
# warm start) was counted as a good replicate. They now report
# `(θ = ..., converged = ..., loglik = ...)`.
#
# No `upper_boundary` flag (the #542 option-3 field) is set: this slice only
# reports the refit's verdict. Whether the hurdle-NB size `r` needs such a flag
# (as NB2 `r` does) is left open and is not tested here.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

function _sim_hd_verdict(kind::Symbol, p::Integer, n::Integer; seed::Integer,
                         r::Real = 3.0, σ::Real = 0.6, α::Real = 3.0)
    rng = MersenneTwister(seed)
    βz = 0.3 .* randn(rng, p) .+ 0.3
    βc = 0.3 .* randn(rng, p) .+ (kind in (:hp, :hnb) ? 1.0 : 0.5)
    Λc = 0.5 .* randn(rng, p, 1)
    Y = zeros(kind in (:hp, :hnb) ? Int : Float64, p, n)
    for s in 1:n
        ηc = βc .+ Λc * randn(rng, 1)
        for t in 1:p
            rand(rng) < inv(1 + exp(-βz[t])) || continue
            μ = exp(ηc[t])
            if kind === :hp
                Y[t, s] = GM._rand_ztpois(rng, μ)
            elseif kind === :hnb
                Y[t, s] = GM._rand_ztnb(rng, r, μ)
            elseif kind === :dln
                Y[t, s] = exp(ηc[t] + σ * randn(rng))
            else
                Y[t, s] = rand(rng, Gamma(α, μ / α))
            end
        end
    end
    return Y
end

function _hd504_routes(Yhp, Yhnb, Ydln, Ydg)
    f0 = GM.fit_hurdle_poisson_gllvm(Yhp; K = 1)
    f1 = GM.fit_hurdle_nb_gllvm(Yhnb; K = 1)
    f2 = GM.fit_delta_lognormal_gllvm(Ydln; K = 1)
    f3 = GM.fit_delta_lognormal_gllvm(Ydln; K = 1, predictor = :shared)
    f4 = GM.fit_delta_gamma_gllvm(Ydg; K = 1)
    f5 = GM.fit_delta_gamma_gllvm(Ydg; K = 1, predictor = :shared)
    return (
        (:hurdle_poisson, f0, Yhp, GM._family_ci(f0, Yhp),
         Yb_ -> GM.fit_hurdle_poisson_gllvm(Yb_; K = 1),
         fb -> vcat(fb.βz, fb.βc, GM.pack_lambda(fb.Λc))),
        (:hurdle_nb, f1, Yhnb, GM._family_ci(f1, Yhnb),
         Yb_ -> GM.fit_hurdle_nb_gllvm(Yb_; K = 1),
         fb -> vcat(fb.βz, fb.βc, GM.pack_lambda(fb.Λc), log(fb.r))),
        (:delta_lognormal, f2, Ydln, GM._family_ci(f2, Ydln),
         Yb_ -> GM.fit_delta_lognormal_gllvm(Yb_; K = 1, disp_group = f2.disp_group),
         fb -> vcat(fb.βz, fb.βc, GM.pack_lambda(fb.Λc), log.(fb.σ))),
        (:delta_lognormal_shared, f3, Ydln, GM._family_ci(f3, Ydln),
         Yb_ -> GM.fit_delta_lognormal_gllvm(Yb_; K = 1, predictor = :shared,
                                             disp_group = f3.disp_group),
         fb -> vcat(fb.βc, GM.pack_lambda(fb.Λc), log.(fb.σ))),
        (:delta_gamma, f4, Ydg, GM._family_ci(f4, Ydg),
         Yb_ -> GM.fit_delta_gamma_gllvm(Yb_; K = 1, disp_group = f4.disp_group),
         fb -> vcat(fb.βz, fb.βc, GM.pack_lambda(fb.Λc), log.(fb.α))),
        (:delta_gamma_shared, f5, Ydg, GM._family_ci(f5, Ydg),
         Yb_ -> GM.fit_delta_gamma_gllvm(Yb_; K = 1, predictor = :shared,
                                         disp_group = f5.disp_group),
         fb -> vcat(fb.βc, GM.pack_lambda(fb.Λc), log.(fb.α))),
    )
end

# A draw no fit on the route can evaluate, so each fitter ends on its failure
# sentinel (`converged = false`, `loglik = -Inf`, finite θ, no throw), with or
# without bounds checking. Data-driven, not RNG-driven: one cell holds 1e300.
# On the hurdle routes the count log-density converts `y` with `Int(y)` and the
# `InexactError` is caught by the objective; on the delta routes the objective
# already returns the sentinel at the warm start and the optimiser never leaves
# it. Not failing draws: a negative value or NaN (`y > 0` is false, so it is
# scored as a zero and the fit converges), a half count on the delta routes (a
# valid positive value), and Inf (the fitter throws an `ArgumentError`, so the
# refit returns `nothing`, which main already rejects).
function _hd504_bad(Y)
    B = Float64.(Y)
    B[1, 1] = 1e300
    return B
end

@testset "Hurdle and delta bootstrap refit reports the fitter's own verdict (#504)" begin
    p, n = 4, 120
    Yhp = _sim_hd_verdict(:hp, p, n; seed = 81)
    Yhnb = _sim_hd_verdict(:hnb, p, n; seed = 82)
    Ydln = _sim_hd_verdict(:dln, p, n; seed = 83)
    Ydg = _sim_hd_verdict(:dg, p, n; seed = 84)
    routes = _hd504_routes(Yhp, Yhnb, Ydln, Ydg)

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

    @testset "a failed refit is rejected ($route)" for (route, fit, Y, ad, direct, pack) in routes
        Y_bad = _hd504_bad(Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        m = length(ad.θ)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, Y, ad, direct, pack) in routes
        Y_bad = _hd504_bad(Y)
        m = length(ad.θ)
        ad_mixed = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, Y, ad, direct, pack = routes[1]   # Hurdle-Poisson: the fastest route
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
