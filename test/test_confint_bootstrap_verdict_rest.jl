# #504, last three refit closures (after Poisson #516 and the per-family
# slices): the bootstrap `refit` closures in `src/confint_family.jl`
# (`_family_ci` for RowRandomFit, MultinomialFit and GllvmCovFit) returned a
# bare parameter vector, so `_bootstrap_refit_ok` (#508) could only check
# `isfinite`. A refit that ended on the fitter's failure verdict (θ is the
# finite warm start) was counted as a good replicate. They now report
# `(θ = ..., converged = ..., loglik = ...)`.
#
# RowRandomFit and GllvmCovFit build θ with a ternary on whether the family
# carries a dispersion, so each is tested with Poisson (no dispersion) and
# NegativeBinomial (dispersion): both branches changed.
#
# No `upper_boundary` flag (the #542 option-3 field) is set: this slice only
# reports the refit's verdict.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

function _sim_rest_counts(p::Integer, n::Integer; seed::Integer, σrow::Real = 0.0,
                          nb::Bool = false, X = nothing, γ::Real = 0.0)
    rng = MersenneTwister(seed)
    β = 0.3 .* randn(rng, p) .+ 0.5
    Λ = 0.5 .* randn(rng, p, 1)
    Y = zeros(Float64, p, n)
    for s in 1:n
        ρ = σrow * randn(rng)
        η = β .+ ρ .+ Λ * randn(rng, 1)
        for t in 1:p
            X === nothing || (η[t] += γ * X[t, s, 1])
            μ = exp(η[t])
            Y[t, s] = nb ? rand(rng, NegativeBinomial(3.0, 3.0 / (3.0 + μ))) :
                           rand(rng, Poisson(μ))
        end
    end
    return Y
end

function _sim_rest_multinomial(n::Integer; seed::Integer)
    rng = MersenneTwister(seed)
    X = randn(rng, n, 1)
    β = [0.3, -0.2]; γ = [0.5, -0.4]
    Y = zeros(Int, 1, n)
    for i in 1:n
        η = [0.0, β[1] + γ[1] * X[i, 1], β[2] + γ[2] * X[i, 1]]
        π = exp.(η) ./ sum(exp.(η))
        Y[1, i] = something(findfirst(rand(rng) .≤ cumsum(π)), 3)
    end
    return Y, X
end

function _rest504_routes()
    p, n = 4, 120
    Xc = zeros(p, n, 1)
    x = randn(MersenneTwister(90), n)
    for s in 1:n, t in 1:p
        Xc[t, s, 1] = x[s]
    end
    Yrp = _sim_rest_counts(p, n; seed = 91, σrow = 0.5)
    Yrn = _sim_rest_counts(p, n; seed = 92, σrow = 0.5, nb = true)
    Ycp = _sim_rest_counts(p, n; seed = 93, X = Xc, γ = 0.4)
    Ycn = _sim_rest_counts(p, n; seed = 94, X = Xc, γ = 0.4, nb = true)
    Ym, Xm = _sim_rest_multinomial(200; seed = 95)

    rr(fam) = Y_ -> GM.fit_row_random_gllvm(Y_; family = fam, K = 1)
    cv(fam) = Y_ -> GM.fit_gllvm_cov(Y_; family = fam, X = Xc, K = 1)
    mn = Y_ -> GM.fit_multinomial_gllvm(Y_; X = Xm, n_categories = 3)

    f0 = rr(Poisson())(Yrp); f1 = rr(NegativeBinomial())(Yrn)
    f2 = cv(Poisson())(Ycp); f3 = cv(NegativeBinomial())(Ycn)
    f4 = mn(Ym)
    return (
        (:rowrandom_poisson, f0, Yrp, GM._family_ci(f0, Yrp), rr(Poisson()),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.σ_row))),
        (:rowrandom_nb, f1, Yrn, GM._family_ci(f1, Yrn), rr(NegativeBinomial()),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.σ_row), log(fb.dispersion))),
        (:cov_poisson, f2, Ycp, GM._family_ci(f2, Ycp; X = Xc), cv(Poisson()),
         fb -> vcat(fb.β, fb.γ, GM.pack_lambda(fb.Λ))),
        (:cov_nb, f3, Ycn, GM._family_ci(f3, Ycn; X = Xc), cv(NegativeBinomial()),
         fb -> vcat(fb.β, fb.γ, GM.pack_lambda(fb.Λ), log(fb.dispersion))),
        (:multinomial, f4, Ym, GM._family_ci(f4, Ym; X = Xm), mn,
         fb -> copy(fb.theta_packed)),
    )
end

# A draw no fit on the route can evaluate, so the fitter ends on its failure
# verdict (`converged = false`, `loglik = -Inf`, finite θ, no throw), with or
# without bounds checking, on Julia 1.10 and 1.13. Data-driven, not RNG-driven.
#   * Row-random and covariate GLLVM (Poisson and NB): one response of -1
#     (log-pmf -Inf at every θ, so every evaluation hits the 1e12 sentinel).
#
# The multinomial route has no such draw, so its rejection and end-to-end
# testsets use a STUB failed refit, labelled as such below; the adapter-parity
# testset is what pins that migrated closure. A probe on Julia 1.10.12 (with and
# without `--check-bounds=yes`) and 1.13.0 found that a category of 0, -1 or 4
# (with 3 categories), and 0.5, NaN or Inf, all throw before fitting (the refit
# returns `nothing`, which main already rejects), while draws with a category
# never observed, or every observation in one category (complete separation),
# fit with `converged = true`.
_rest504_bad(Y) = (B = copy(Y); B[1, 1] = -1.0; B)

_rest504_real_bad(route::Symbol) = route !== :multinomial

# STUB (not a real fit): what a refit ending on the fitter's failure verdict
# reports, a finite θ with `converged = false` and `loglik = -Inf`.
_rest504_stub_failed(ad) = (θ = copy(ad.θ), converged = false, loglik = -Inf)

@testset "Row-random, multinomial and covariate-GLLVM bootstrap refit reports the fitter's own verdict (#504)" begin
    routes = _rest504_routes()

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

    real_routes = filter(r -> _rest504_real_bad(r[1]), routes)
    stub_routes = filter(r -> !_rest504_real_bad(r[1]), routes)

    @testset "a failed refit is rejected, real draw ($route)" for (route, fit, Y, ad, direct, pack) in real_routes
        Y_bad = _rest504_bad(Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        m = length(ad.θ)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test all(isfinite, raw_bad.θ)                   # the whole θ, not one entry
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "a failed refit (stub) is rejected ($route)" for (route, fit, Y, ad, direct, pack) in stub_routes
        raw_bad = _rest504_stub_failed(ad)
        m = length(ad.θ)
        @test all(isfinite, raw_bad.θ)
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, Y, ad, direct, pack) in routes
        m = length(ad.θ)
        ad_mixed = if _rest504_real_bad(route)
            Y_bad = _rest504_bad(Y)
            GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                         rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        else
            bad_marker = fill(-1, size(Y))   # STUB: a sentinel draw, never passed to a fitter
            GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                         rng -> rand(rng) < 0.5 ? Y : bad_marker,
                         Yb_ -> Yb_ === bad_marker ? _rest504_stub_failed(ad) : ad.refit(Yb_))
        end
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, Y, ad, direct, pack = routes[5]   # multinomial (fixed effects): the fastest route
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
