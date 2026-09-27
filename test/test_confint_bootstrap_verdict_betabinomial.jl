# #542 (part of #504): the beta-binomial bootstrap `refit` closures in
# `src/confint_family.jl` (`_family_ci` for BetaBinomialFit,
# BetaBinomialGroupedFit and BetaBinomialGroupedCovFit) used to return a bare
# parameter vector, so `_bootstrap_refit_ok` (#508) could only check `isfinite`
# on it. A replicate whose refit reported converged = false was still counted as
# a good draw: the fitter's `1e12` failure sentinel, and (since #522) a Beta
# precision at the 1e6 boundary, where the loglik and every parameter, including
# log φ ≈ 29, are finite. This follows the Poisson migration (#516,
# test_confint_bootstrap_verdict_poisson.jl): the closures now return
# `(θ = ..., converged = ..., loglik = ...)`, and such replicates are excluded
# and not counted in `n_converged` (maintainer choice, option 1 on #542).
#
# Every assertion is a relation (the adapter against a direct refit, the new
# contract against the old bare-vector one, a count strictly between 0 and
# n_boot) rather than one seed's fitted number. The data are literal and
# hash-verified: #522's fixture for healthy fits and
# fixtures/beta_binomial_boot_boundary_542.toml for the boundary case.
using GLLVModels, Test, Random, TOML, SHA
const GM = GLLVModels

const BB542_HEALTHY = joinpath(@__DIR__, "fixtures", "beta_binomial_verdict_515.toml")
const BB542_BOUNDARY = joinpath(@__DIR__, "fixtures", "beta_binomial_boot_boundary_542.toml")

function _bb542_case(path, key)
    fx = TOML.parsefile(path)
    p, n = fx["p"], fx["n"]
    c = fx[key]
    y = Int64.(c["Y_column_major"]); nn = Int64.(c["N_column_major"])
    @test bytes2hex(sha256(reinterpret(UInt8, y))) == c["data_sha256"]
    @test bytes2hex(sha256(reinterpret(UInt8, nn))) == c["N_sha256"]
    return reshape(y, p, n), reshape(nn, p, n)
end

# Deterministic literal covariate (no RNG), as in test_beta_binomial_grouped_verdict_515.jl.
_bb542_X(p, n) = reshape(Float64[((7s + 3t) % 11) / 5 - 1 for t in 1:p, s in 1:n], p, n, 1)

# One base fit per fit type on the healthy dataset, its CI adapter, and a direct
# refit function with exactly the adapter's settings.
function _bb542_routes(Y, N)
    p, n = size(Y)
    X = _bb542_X(p, n)
    one_ = ones(Int, p)
    f0 = GM.fit_beta_binomial_gllvm(Y; K = 2, N = N)
    f1 = GM.fit_beta_binomial_gllvm_grouped(Y; K = 2, N = N, group = one_)
    f2 = GM.fit_beta_binomial_gllvm_grouped_cov(Y; X = X, K = 2, N = N, group = one_)
    return (
        (:ungrouped, f0, GM._family_ci(f0, Y; N = N),
         Yb -> GM.fit_beta_binomial_gllvm(Yb; K = 2, link = f0.link, N = N),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.φ))),
        (:grouped, f1, GM._family_ci(f1, Y; N = N),
         Yb -> GM.fit_beta_binomial_gllvm_grouped(Yb; K = 2, N = N, group = one_, link = f1.link),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log.(fb.φ))),
        (:grouped_cov, f2, GM._family_ci(f2, Y; N = N, X = X),
         Yb -> GM.fit_beta_binomial_gllvm_grouped_cov(Yb; X = X, K = 2, N = N, group = one_,
                                                      link = f2.link, γ_fixed = f2.γ_fixed),
         fb -> vcat(fb.β, fb.γ[findall(!, f2.γ_fixed)], GM.pack_lambda(fb.Λ), log.(fb.φ))),
    )
end

# A draw no fit can evaluate: one count above its trial number. Every θ the
# optimiser tries gives a log-pmf of -Inf, so each fitter ends on its failure
# sentinel and reports converged = false, loglik = -Inf, at iteration 0. It is
# data-driven, not RNG-driven, so it is the same on every Julia version.
function _bb542_bad(Y, N)
    B = copy(Y)
    B[1, 1] = N[1, 1] + 5
    return B
end

@testset "BetaBinomial bootstrap refit reports the fitter's own verdict (#542)" begin
    Y, N = _bb542_case(BB542_HEALTHY, "healthy_seed_9001")
    routes = _bb542_routes(Y, N)

    @testset "adapter parity with a direct refit ($route)" for (route, fit, ad, direct, pack) in routes
        @test fit.converged   # sanity: the base fit itself is healthy
        Yb = ad.simulate(MersenneTwister(542))
        raw = ad.refit(Yb)
        # The migrated contract: a named tuple, not a bare vector. A bare vector
        # has no `.converged` / `.loglik`, which is why this fails on main.
        @test raw isa NamedTuple
        @test raw.converged isa Bool
        # Each fitter's warm start is a deterministic function of the data, so
        # an independent refit on the same draw reproduces the adapter's fit.
        fb = direct(Yb)
        @test raw.converged == fb.converged
        @test raw.loglik == fb.loglik
        @test raw.θ == pack(fb)
    end

    @testset "a failed refit is rejected ($route)" for (route, fit, ad, direct, pack) in routes
        Y_bad = _bb542_bad(Y, N)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        @test raw_bad isa NamedTuple
        @test raw_bad.converged == false
        @test !GM._bootstrap_refit_ok(raw_bad, length(ad.θ))[2]
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, ad, direct, pack) in routes
        # A stubbed `simulate` alternates the healthy original `Y` (whose refit
        # converges; deterministic warm start) with the failing draw, so
        # n_converged must land strictly between 0 and n_boot.
        Y_bad = _bb542_bad(Y, N)
        m = length(ad.θ)
        n_boot = 6
        ad_mixed = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, n_boot, 7, false)
        @test 0 < result.n_converged < n_boot
    end

    @testset "a phi-boundary replicate is rejected, though every value in it is finite" begin
        # On these literal Binomial datasets fit_beta_binomial_gllvm runs φ past
        # 1e6 (seed 3: φ ≈ 5.8e12, seed 6: φ ≈ 1.1e8 on Julia 1.10.12) and
        # reports converged = false with a finite loglik (#522). The adapter is
        # built on that fit, so its refit of the same data is the same boundary
        # fit: the bootstrap replicate #542 is about.
        for key in ("binomial_seed_3", "binomial_seed_6")
            Yk, Nk = _bb542_case(BB542_BOUNDARY, key)
            fk = GM.fit_beta_binomial_gllvm(Yk; K = 2, N = Nk)
            @test fk.φ >= GM._BB_PHI_STABLE && !fk.converged   # the recorded state
            ad = GM._family_ci(fk, Yk; N = Nk)
            m = length(ad.θ)
            raw = ad.refit(Yk)
            @test raw.converged == false
            @test isfinite(raw.loglik) && raw.loglik < 0
            @test all(isfinite, raw.θ)
            # The old bare-vector contract would have counted it ...
            @test GM._bootstrap_refit_ok(raw.θ, m)[2]
            # ... the migrated contract does not.
            @test !GM._bootstrap_refit_ok(raw, m)[2]
        end
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        # When every replicate converges, the migration must not move any
        # endpoint. Rebuild the old behaviour by wrapping the same refit so it
        # returns only θ, and compare the two bootstrap results exactly.
        route, fit, ad, direct, pack = routes[2]   # grouped: the default public route
        m = length(ad.θ)
        ad_bare = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds, ad.simulate,
                               Yb -> (r = ad.refit(Yb); r === nothing ? nothing : r.θ))
        # `_family_bootstrap` reports NaN bounds below 10 usable replicates, so
        # n_boot must be at least 10 or this comparison would pass on NaN == NaN.
        n_boot = 10
        new = GM._family_bootstrap(ad, collect(1:m), 0.95, n_boot, 11, false)
        old = GM._family_bootstrap(ad_bare, collect(1:m), 0.95, n_boot, 11, false)
        @test new.n_converged == n_boot   # precondition: every replicate converged
        @test old.n_converged == n_boot
        @test all(isfinite, new.lower) && all(isfinite, new.upper)   # non-vacuous
        @test isequal(new.lower, old.lower)
        @test isequal(new.upper, old.upper)
        @test isequal(new.estimate, old.estimate)
    end
end
