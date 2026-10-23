# #504, Ordinal migration (after Poisson #516 and Gamma): the bootstrap `refit`
# closures in `src/confint_family.jl` (`_family_ci` for OrdinalFit,
# OrdinalPerTraitFit and OrdinalPerTraitCovFit) returned a bare parameter vector,
# so `_bootstrap_refit_ok` (#508) could only check `isfinite`. A refit that ended
# on the fitter's failure sentinel (θ is the finite warm start) was counted as a
# good replicate. They now report `(θ = ..., converged = ..., loglik = ...)`.
#
# No `upper_boundary` flag is set: the ordinal parameters (loadings, intercepts,
# cutpoints) have no flat-likelihood limit that the #542 field describes.
#
# Unlike Gamma, no data-driven draw makes an ordinal fitter fail softly
# (`converged = false`, `loglik = -Inf`, no throw) under bounds checking. Every
# category probability is clamped at 1e-12, so any draw coded 1:C evaluates to
# a finite likelihood; a level of 0 or below throws a BoundsError (with
# `--check-bounds=yes`, as `Pkg.test` runs); a level above C changes the
# category count, which the closure already drops. So the rejection and
# end-to-end testsets use a STUB failed refit, labelled as such below; the
# adapter-parity testset is what pins the migrated closures themselves.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random, LinearAlgebra
const GM = GLLVModels

function _sim_ordinal_verdict(p::Integer, K::Integer, n::Integer, C::Integer; seed::Integer)
    rng = MersenneTwister(seed)
    Λ = 0.7 .* randn(rng, p, K)
    τ = collect(range(-1.0, 1.2; length = C - 1))
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        η = Λ * randn(rng, K)
        for t in 1:p
            u = rand(rng); cum = 0.0; cat = C
            for c in 1:C
                cum += GM._ord_prob(c, η[t], τ, GM.LogitLink())
                if u <= cum
                    cat = c; break
                end
            end
            Y[t, s] = cat
        end
    end
    return Y
end

_ordinal504_X(p, n) = reshape(Float64[((7s + 3t) % 11) / 5 - 1 for t in 1:p, s in 1:n], p, n, 1)

function _ordinal504_routes(Y)
    p, n = size(Y)
    X = _ordinal504_X(p, n)
    f0 = GM.fit_ordinal_gllvm(Y; K = 1)
    f1 = GM.fit_ordinal_gllvm_pertrait(Y; K = 1)
    f2 = GM.fit_ordinal_gllvm_pertrait_cov(Y; X = X, K = 1)
    return (
        (:shared, f0, GM._family_ci(f0, Y),
         Yb -> GM.fit_ordinal_gllvm(Yb; K = 1, link = f0.link),
         fb -> vcat(GM.pack_lambda(fb.Λ), fb.τ)),
        (:pertrait, f1, GM._family_ci(f1, Y),
         Yb -> GM.fit_ordinal_gllvm_pertrait(Yb; K = 1, link = f1.link),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), GM._pack_free_tau_pertrait(fb.τ, f1.C))),
        (:pertrait_cov, f2, GM._family_ci(f2, Y; X = X),
         Yb -> GM.fit_ordinal_gllvm_pertrait_cov(Yb; X = X, K = 1, link = f2.link,
                                                  γ_fixed = f2.γ_fixed),
         fb -> vcat(fb.β, fb.γ[findall(!, f2.γ_fixed)], GM.pack_lambda(fb.Λ),
                    GM._pack_free_tau_pertrait(fb.τ, f2.C))),
    )
end

# STUB (not a real fit): what a refit ending on the fitter's failure sentinel
# reports, a finite θ with `converged = false` and `loglik = -Inf`.
_ordinal504_stub_failed(ad) = (θ = copy(ad.θ), converged = false, loglik = -Inf)

@testset "Ordinal bootstrap refit reports the fitter's own verdict (#504)" begin
    Y = _sim_ordinal_verdict(4, 1, 150, 4; seed = 34)
    routes = _ordinal504_routes(Y)

    @testset "adapter parity with a direct refit ($route)" for (route, fit, ad, direct, pack) in routes
        @test fit.converged   # sanity: a healthy fit
        Yb = ad.simulate(MersenneTwister(504))
        raw = ad.refit(Yb)
        @test raw isa NamedTuple   # a bare vector on main: this is where main fails
        fb = direct(Yb)
        @test fb.C == fit.C        # same category count, so the closure keeps the draw
        @test raw.converged == fb.converged
        @test raw.loglik == fb.loglik
        @test raw.θ == pack(fb)
        @test !haskey(raw, :upper_boundary)   # deliberately not flagged (see header)
    end

    @testset "a category-count change is still dropped ($route)" for (route, fit, ad, direct, pack) in routes
        Y_wide = copy(Y)
        Y_wide[1, 1] = maximum(Y) + 1    # a new top level changes C
        @test ad.refit(Y_wide) === nothing
    end

    @testset "a failed refit (stub) is rejected ($route)" for (route, fit, ad, direct, pack) in routes
        raw_bad = _ordinal504_stub_failed(ad)
        m = length(ad.θ)
        @test all(isfinite, raw_bad.θ)
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end (stub failures): n_converged counts only converged replicates ($route)" for (route, fit, ad, direct, pack) in routes
        m = length(ad.θ)
        bad_marker = fill(-1, size(Y))   # a sentinel draw, never passed to a fitter
        ad_mixed = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                rng -> rand(rng) < 0.5 ? Y : bad_marker,
                                Yb -> Yb === bad_marker ? _ordinal504_stub_failed(ad) : ad.refit(Yb))
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, ad, direct, pack = routes[1]   # shared cutpoints, the fastest route
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
