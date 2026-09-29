# #504, Tweedie migration (after Poisson #516): the bootstrap `refit` closures in
# `src/confint_family.jl` (`_family_ci` for TweedieFit, TweedieGroupedFit and
# TweediePerTraitPowerFit) returned a bare parameter vector, so
# `_bootstrap_refit_ok` (#508) could only check `isfinite`. A refit that ended on
# the fitter's failure sentinel (`_tweedie_verdict` reports `:objective_failed`
# with a finite warm-start θ) was counted as a good replicate. They now report
# `(θ = ..., converged = ..., loglik = ...)`.
#
# No `upper_boundary` flag (the #542 option-3 field) is set: the power is held
# fixed in the CI layer, and the Tweedie verdict already flags a power at the
# closed end of (1, 2) itself (`:power_at_boundary`).
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random
const GM = GLLVModels

function _sim_tweedie_verdict(p::Integer, K::Integer, n::Integer, φ::Real, pw::Real; seed::Integer)
    rng = MersenneTwister(seed)
    β = 0.3 .* randn(rng, p) .+ 0.5
    Λ = 0.4 .* randn(rng, p, K)
    Z = randn(rng, K, n)
    μ = exp.(β .+ Λ * Z)
    return [GM._tweedie_sample(μ[t, s], φ, pw, rng) for t in 1:p, s in 1:n]
end

function _tweedie504_routes(Y)
    p = size(Y, 1)
    one_ = ones(Int, p)
    f0 = GM.fit_tweedie_gllvm(Y; K = 1)
    f1 = GM.fit_tweedie_gllvm_grouped(Y; K = 1, group = one_, power_group = :shared)
    f2 = GM.fit_tweedie_gllvm_grouped(Y; K = 1, group = one_, power_group = :species)
    return (
        (:shared, f0, GM._family_ci(f0, Y),
         Yb -> GM.fit_tweedie_gllvm(Yb; K = 1, link = f0.link, hessian = f0.hessian),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.φ))),
        (:grouped, f1, GM._family_ci(f1, Y),
         Yb -> GM.fit_tweedie_gllvm_grouped(Yb; K = 1, group = one_, power_group = :shared,
                                            link = f1.link, hessian = f1.hessian),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log.(fb.φ))),
        (:per_trait_power, f2, GM._family_ci(f2, Y),
         Yb -> GM.fit_tweedie_gllvm_grouped(Yb; K = 1, group = one_, power_group = :species,
                                            link = f2.link, hessian = f2.hessian),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log.(fb.φ))),
    )
end

# A draw no Tweedie fit can evaluate: one cell at 1e300, so the Laplace marginal
# fails at every accepted point and each fitter ends on its failure sentinel
# (`converged = false`, `loglik = -Inf`, finite warm-start θ) without throwing.
# A negative, NaN or Inf cell throws instead, which the adapter already maps to
# `nothing`. Data-driven, not RNG-driven.
function _tweedie504_bad(Y)
    B = copy(Y)
    B[1, 1] = 1e300
    return B
end

@testset "Tweedie bootstrap refit reports the fitter's own verdict (#504)" begin
    Y = _sim_tweedie_verdict(4, 1, 50, 1.0, 1.5; seed = 71)
    routes = _tweedie504_routes(Y)

    @testset "adapter parity with a direct refit ($route)" for (route, fit, ad, direct, pack) in routes
        @test fit.converged   # sanity: φ = 1, power = 1.5, a healthy fit
        Yb = ad.simulate(MersenneTwister(504))
        raw = ad.refit(Yb)
        @test raw isa NamedTuple   # a bare vector on main: this is where main fails
        fb = direct(Yb)
        @test raw.converged == fb.converged
        @test raw.loglik == fb.loglik
        @test raw.θ == pack(fb)
        @test !haskey(raw, :upper_boundary)   # deliberately not flagged (see header)
    end

    @testset "a failed refit is rejected ($route)" for (route, fit, ad, direct, pack) in routes
        Y_bad = _tweedie504_bad(Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        m = length(ad.θ)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test all(isfinite, raw_bad.θ)                   # the warm start, not NaN
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, ad, direct, pack) in routes
        Y_bad = _tweedie504_bad(Y)
        m = length(ad.θ)
        ad_mixed = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, ad, direct, pack = routes[1]   # shared, the fastest route
        m = length(ad.θ)
        ad_bare = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds, ad.simulate,
                               Yb -> (r = ad.refit(Yb); r === nothing ? nothing : r.θ))
        # `_family_bootstrap` reports NaN bounds below 10 usable replicates.
        n_boot = 10
        new = GM._family_bootstrap(ad, collect(1:m), 0.95, n_boot, 11, false)
        old = GM._family_bootstrap(ad_bare, collect(1:m), 0.95, n_boot, 11, false)
        @test new.n_converged == n_boot && old.n_converged == n_boot
        @test all(isfinite, new.lower) && all(isfinite, new.upper)   # non-vacuous
        @test isequal(new.lower, old.lower) && isequal(new.upper, old.upper)
    end
end
