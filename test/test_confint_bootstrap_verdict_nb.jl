# #504, NB2 migration (after Poisson #516 and beta-binomial #542): the bootstrap
# `refit` closures in `src/confint_family.jl` (`_family_ci` for NBFit,
# NBGroupedFit and NBGroupedCovFit) returned a bare parameter vector, so
# `_bootstrap_refit_ok` (#508) could only check `isfinite`. They now return
# `(θ, converged, loglik, upper_boundary)`: a non-converged refit is excluded,
# and `upper_boundary` flags each `log r` past 1e6 (the Poisson limit; the same
# test `_dispersion_group_boundary` applies to the grouped point fit). Following
# option 3 on #542, `_family_bootstrap` reports `Inf` as the upper bound of an
# `r` whose flagged share of usable replicates exceeds `(1 - level)/2`.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), never one platform's fitted number. Data are
# literal and hash-verified: fixtures/nb_boot_boundary_504.toml.
using GLLVModels, Test, Random, TOML, SHA
const GM = GLLVModels

const NB504_FIXTURE = joinpath(@__DIR__, "fixtures", "nb_boot_boundary_504.toml")

function _nb504_case(key)
    fx = TOML.parsefile(NB504_FIXTURE)
    y = Int64.(fx[key]["Y_column_major"])
    @test bytes2hex(sha256(reinterpret(UInt8, y))) == fx[key]["data_sha256"]
    return reshape(y, fx["p"], fx["n"])
end

# Deterministic literal covariate (no RNG), as in the beta-binomial verdict tests.
_nb504_X(p, n) = reshape(Float64[((7s + 3t) % 11) / 5 - 1 for t in 1:p, s in 1:n], p, n, 1)

function _nb504_routes(Y)
    p, n = size(Y)
    X = _nb504_X(p, n)
    one_ = ones(Int, p)
    f0 = GM.fit_nb_gllvm(Y; K = 2)
    f1 = GM.fit_nb_gllvm_grouped(Y; K = 2, group = one_)
    f2 = GM.fit_nb_gllvm_grouped_cov(Y; X = X, K = 2, group = one_)
    return (
        (:shared, f0, GM._family_ci(f0, Y),
         Yb -> GM.fit_nb_gllvm(Yb; K = 2, link = f0.link, hessian = f0.hessian),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.r))),
        (:grouped, f1, GM._family_ci(f1, Y),
         Yb -> GM.fit_nb_gllvm_grouped(Yb; K = 2, group = one_, link = f1.link, hessian = f1.hessian),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log.(fb.r_group))),
        (:grouped_cov, f2, GM._family_ci(f2, Y; X = X),
         Yb -> GM.fit_nb_gllvm_grouped_cov(Yb; X = X, K = 2, group = one_, link = f2.link,
                                           γ_fixed = f2.γ_fixed, hessian = f2.hessian),
         fb -> vcat(fb.β, fb.γ[findall(!, f2.γ_fixed)], GM.pack_lambda(fb.Λ), log.(fb.r_group))),
    )
end

# A draw no fit can evaluate: one negative count. Each fitter ends on its failure
# sentinel (converged = false, loglik = -Inf). Data-driven, not RNG-driven.
function _nb504_bad(Y)
    B = copy(Y)
    B[1, 1] = -5
    return B
end

@testset "NB2 bootstrap refit reports the fitter's own verdict (#504)" begin
    Y = _nb504_case("nb_r3_seed_1")
    routes = _nb504_routes(Y)

    @testset "adapter parity with a direct refit ($route)" for (route, fit, ad, direct, pack) in routes
        @test fit.converged   # sanity: NB data with r = 3, a healthy fit
        Yb = ad.simulate(MersenneTwister(504))
        raw = ad.refit(Yb)
        @test raw isa NamedTuple   # a bare vector on main: this is where main fails
        fb = direct(Yb)
        @test raw.converged == fb.converged
        @test raw.loglik == fb.loglik
        @test raw.θ == pack(fb)
        @test raw.upper_boundary == GM._nb_r_upper_boundary(pack(fb), route === :shared ? 1 : length(fb.r_group))
        @test !any(raw.upper_boundary)   # r ≈ 3: nowhere near the boundary
    end

    @testset "a failed refit is rejected ($route)" for (route, fit, ad, direct, pack) in routes
        Y_bad = _nb504_bad(Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test !GM._bootstrap_refit_ok(raw_bad, length(ad.θ))[2]
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, ad, direct, pack) in routes
        Y_bad = _nb504_bad(Y)
        m = length(ad.θ)
        ad_mixed = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
    end

    @testset "shared-r fit at the Poisson limit: flagged, and the r bound is Inf" begin
        # On poisson_seed_1 (Poisson data) fit_nb_gllvm runs r past 1e6 and still
        # reports converged = true (7.66e6 on Julia 1.10.12): the shared-r fitter
        # has no boundary verdict. The old contract counts every such replicate
        # with a finite log r; the flag excludes it, and when all replicates are
        # at the boundary the upper bound for r is Inf.
        Yp = _nb504_case("poisson_seed_1")
        fp = GM.fit_nb_gllvm(Yp; K = 2)
        @test fp.r > 1e6   # the recorded state
        ad = GM._family_ci(fp, Yp)
        m = length(ad.θ)
        raw = ad.refit(Yp)
        @test all(isfinite, raw.θ) && isfinite(raw.loglik)
        @test raw.upper_boundary == [i == m for i in 1:m]
        @test GM._bootstrap_refit_ok(raw.θ, m)[2]   # the old contract accepts it
        n_boot = 10
        ad_p = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds, rng -> Yp, ad.refit)
        new = GM._family_bootstrap(ad_p, collect(1:m), 0.95, n_boot, 3, false)
        @test new.n_converged == 0
        @test new.upper[m] == Inf
        ad_bare = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds, rng -> Yp,
                               Yb -> (r = ad.refit(Yb); r === nothing ? nothing : r.θ))
        old = GM._family_bootstrap(ad_bare, collect(1:m), 0.95, n_boot, 3, false)
        @test old.n_converged == n_boot && isfinite(old.upper[m])   # what main reported
    end

    @testset "per-species grouped refit flags exactly the species past 1e6" begin
        # poisson_seed_2, group = 1:p: several species' r run far past 1e6 (up to
        # 1.0e48 on Julia 1.10.12); the grouped fitter reports converged = false.
        Yp = _nb504_case("poisson_seed_2")
        p = size(Yp, 1)
        fk = GM.fit_nb_gllvm_grouped(Yp; K = 2, group = collect(1:p))
        @test any(>(1e6), fk.r_group) && !fk.converged   # the recorded state
        ad = GM._family_ci(fk, Yp)
        m = length(ad.θ)
        raw = ad.refit(Yp)
        @test raw.upper_boundary == vcat(fill(false, m - p), fk.r_group .> 1e6)
        @test 1 <= count(raw.upper_boundary) < p
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        route, fit, ad, direct, pack = routes[2]   # grouped, one group
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
        @test isequal(new.estimate, old.estimate)
    end
end
