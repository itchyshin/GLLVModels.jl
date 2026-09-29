# #504, Gamma migration (after Poisson #516): the bootstrap `refit` closures in
# `src/confint_family.jl` (`_family_ci` for GammaFit, GammaGroupedFit and
# GammaGroupedCovFit) returned a bare parameter vector, so `_bootstrap_refit_ok`
# (#508) could only check `isfinite`. A refit that ended on the fitter's failure
# sentinel (θ is the finite warm start) was counted as a good replicate. They now
# report `(θ = ..., converged = ..., loglik = ...)`.
#
# No `upper_boundary` flag (the #542 option-3 field) is set for the Gamma shape
# α: unlike NB2 `r` or the beta-binomial `φ`, a large α is not a flat-likelihood
# limit. On near-deterministic data (true α = 1e8) every route estimates α close
# to the truth, so flagging α > 1e6 would drop every replicate.
#
# Assertions are relations (adapter against a direct refit, new contract against
# the old bare-vector one, counts), so the seeded draw differing across Julia
# versions does not matter.
using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

function _sim_gamma_verdict(p::Integer, K::Integer, n::Integer, α::Real; seed::Integer)
    rng = MersenneTwister(seed)
    β = 0.5 .* randn(rng, p) .+ 1.0
    Λ = 0.5 .* randn(rng, p, K)
    z = randn(rng, K, n)
    return [rand(rng, Gamma(α, exp(β[t] + dot(Λ[t, :], z[:, s])) / α)) for t in 1:p, s in 1:n]
end

_gamma504_X(p, n) = reshape(Float64[((7s + 3t) % 11) / 5 - 1 for t in 1:p, s in 1:n], p, n, 1)

function _gamma504_routes(Y)
    p, n = size(Y)
    X = _gamma504_X(p, n)
    one_ = ones(Int, p)
    f0 = GM.fit_gamma_gllvm(Y; K = 1)
    f1 = GM.fit_gamma_gllvm_grouped(Y; K = 1, group = one_)
    f2 = GM.fit_gamma_gllvm_grouped_cov(Y; X = X, K = 1, group = one_)
    return (
        (:shared, f0, GM._family_ci(f0, Y),
         Yb -> GM.fit_gamma_gllvm(Yb; K = 1, link = f0.link, hessian = f0.hessian),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log(fb.α))),
        (:grouped, f1, GM._family_ci(f1, Y),
         Yb -> GM.fit_gamma_gllvm_grouped(Yb; K = 1, group = one_, link = f1.link, hessian = f1.hessian),
         fb -> vcat(fb.β, GM.pack_lambda(fb.Λ), log.(fb.α))),
        (:grouped_cov, f2, GM._family_ci(f2, Y; X = X),
         Yb -> GM.fit_gamma_gllvm_grouped_cov(Yb; X = X, K = 1, group = one_, link = f2.link,
                                              γ_fixed = f2.γ_fixed, hessian = f2.hessian),
         fb -> vcat(fb.β, fb.γ[findall(!, f2.γ_fixed)], GM.pack_lambda(fb.Λ), log.(fb.α))),
    )
end

# A draw no Gamma fit can evaluate: one zero cell (log-density -Inf), so each
# fitter ends on its failure sentinel. Data-driven, not RNG-driven.
function _gamma504_bad(Y)
    B = copy(Y)
    B[1, 1] = 0.0
    return B
end

@testset "Gamma bootstrap refit reports the fitter's own verdict (#504)" begin
    Y = _sim_gamma_verdict(5, 1, 120, 2.0; seed = 61)
    routes = _gamma504_routes(Y)

    @testset "adapter parity with a direct refit ($route)" for (route, fit, ad, direct, pack) in routes
        @test fit.converged   # sanity: α = 2, a healthy fit
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
        Y_bad = _gamma504_bad(Y)
        fb_bad = direct(Y_bad)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        m = length(ad.θ)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]    # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]     # the migrated one does not
    end

    @testset "end to end: n_converged counts only converged replicates ($route)" for (route, fit, ad, direct, pack) in routes
        Y_bad = _gamma504_bad(Y)
        m = length(ad.θ)
        ad_mixed = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, 6, 7, false)
        @test 0 < result.n_converged < 6
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
    end
end
