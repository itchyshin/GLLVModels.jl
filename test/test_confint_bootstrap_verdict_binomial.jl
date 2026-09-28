# #504, Binomial migration (after Poisson #516): the Laplace-route Binomial
# bootstrap `refit` closure in `src/confint_family.jl` (`_family_ci(fit::BinomialFit,
# ...)`) returned a bare parameter vector, so `_bootstrap_refit_ok` (#508) could
# only check `isfinite` on it. A refit that ended on the fitter's `1e12` failure
# sentinel (θ is the finite warm start) was counted as a good replicate. The
# closure now reports `(θ = ..., converged = ..., loglik = ...)`. The AGHQ route
# already checked `fb.converged` and is untouched.
#
# Every assertion is a relation (the adapter against an independent direct refit,
# the new contract against the old bare-vector one, a count strictly between 0
# and n_boot), so it holds on every Julia version and platform even though the
# seeded simulation draws different data there.
using GLLVModels, Test, Random, Distributions
const GM = GLLVModels

function _sim_binomial_verdict(p::Integer, K::Integer, n::Integer, Ntr::Integer; seed::Integer)
    rng = MersenneTwister(seed)
    β = 0.5 .* randn(rng, p)
    Λ = 0.5 .* randn(rng, p, K)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        η = β .+ Λ * randn(rng, K)
        for t in 1:p
            Y[t, s] = rand(rng, Binomial(Ntr, 1 / (1 + exp(-η[t]))))
        end
    end
    return Y, fill(Ntr, p, n)
end

# A draw no fit can evaluate: one count above its trial number (log-pmf -Inf for
# every θ), so the fitter ends on its failure sentinel. Data-driven, not RNG-driven.
function _binom_bad(Y, N)
    B = copy(Y)
    B[1, 1] = N[1, 1] + 3
    return B
end

@testset "Binomial bootstrap refit reports the fitter's own verdict (#504)" begin
    @testset "adapter parity: converged, loglik and θ match a direct refit" begin
        for (p, K, n, Ntr, seed) in ((4, 1, 120, 8, 51), (5, 1, 150, 10, 52), (3, 1, 90, 6, 53))
            Y, N = _sim_binomial_verdict(p, K, n, Ntr; seed = seed)
            fit = fit_binomial_gllvm(Y; K = K, N = N)
            @test fit.converged   # sanity: the base fit itself is healthy
            ad = GM._family_ci(fit, Y; N = N)
            Yb = ad.simulate(MersenneTwister(seed + 1000))
            raw = ad.refit(Yb)
            @test raw isa NamedTuple   # a bare vector on main: this is where main fails
            @test raw.converged isa Bool
            fb = fit_binomial_gllvm(Yb; K = K, link = fit.link, N = N, hessian = fit.hessian)
            @test raw.converged == fb.converged
            @test raw.θ == vcat(fb.β, GM.pack_lambda(fb.Λ))
            @test raw.loglik == fb.loglik
        end
    end

    @testset "a failed refit is rejected, and n_converged drops" begin
        Y, N = _sim_binomial_verdict(4, 1, 120, 8; seed = 54)
        fit = fit_binomial_gllvm(Y; K = 1, N = N)
        @test fit.converged
        ad = GM._family_ci(fit, Y; N = N)
        m = length(ad.θ)
        Y_bad = _binom_bad(Y, N)
        fb_bad = fit_binomial_gllvm(Y_bad; K = 1, N = N)
        @test !fb_bad.converged && fb_bad.loglik == -Inf   # a genuine failure
        raw_bad = ad.refit(Y_bad)
        @test raw_bad isa NamedTuple && raw_bad.converged == false
        @test all(isfinite, raw_bad.θ)                    # why main counted it
        @test GM._bootstrap_refit_ok(raw_bad.θ, m)[2]     # the old contract accepts it
        @test !GM._bootstrap_refit_ok(raw_bad, m)[2]      # the migrated one does not

        # End to end: a stubbed `simulate` alternates the healthy `Y` with `Y_bad`.
        n_boot = 20
        ad_mixed = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                rng -> rand(rng) < 0.5 ? Y : Y_bad, ad.refit)
        result = GM._family_bootstrap(ad_mixed, collect(1:m), 0.95, n_boot, 5, false)
        @test 0 < result.n_converged < n_boot
    end

    @testset "all-converged bootstrap: endpoints identical to the bare-vector contract" begin
        Y, N = _sim_binomial_verdict(4, 1, 120, 8; seed = 55)
        fit = fit_binomial_gllvm(Y; K = 1, N = N)
        ad = GM._family_ci(fit, Y; N = N)
        m = length(ad.θ)
        ad_bare = GM._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds, ad.simulate,
                               Yb -> (r = ad.refit(Yb); r === nothing ? nothing : r.θ))
        # `_family_bootstrap` reports NaN bounds below 10 usable replicates.
        n_boot = 12
        new = GM._family_bootstrap(ad, collect(1:m), 0.95, n_boot, 9, false)
        old = GM._family_bootstrap(ad_bare, collect(1:m), 0.95, n_boot, 9, false)
        @test new.n_converged == old.n_converged
        @test new.n_converged >= 10
        @test all(isfinite, new.lower) && all(isfinite, new.upper)   # non-vacuous
        @test isequal(new.lower, old.lower) && isequal(new.upper, old.upper)
    end
end
