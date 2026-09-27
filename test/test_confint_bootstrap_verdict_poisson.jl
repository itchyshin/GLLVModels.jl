# #504: the Laplace-route Poisson bootstrap `refit` closure in
# `src/confint_family.jl` (`_family_ci(fit::PoissonFit, ...)`) used to return a
# bare parameter vector, so `_bootstrap_refit_ok` (introduced in #508) could
# only check `isfinite` on it — a replicate whose refit landed on the
# fitter's own `1e12` failure sentinel, or otherwise failed to converge, was
# still counted as a successful bootstrap draw. This file is RED on
# `origin/main` (the bare vector has no `.converged` field to read) and GREEN
# once the closure reports `(θ = ..., converged = ..., loglik = ...)`.
#
# Every assertion here is a RELATION (the adapter's own verdict against an
# independent direct refit, or a count strictly between 0 and n_boot) rather
# than one seed's fitted number, so it is stable across the Julia 1.10 / 1.13
# CI matrix and across platforms.
using GLLVModels, Test, Random, Distributions

# Self-contained simulator (deliberately not shared with `_sim_poisson` in
# test_confint_family.jl — this file must stand alone).
function _sim_poisson_verdict(p::Integer, K::Integer, n::Integer; seed::Integer)
    rng = MersenneTwister(seed)
    β = 0.5 .* randn(rng, p) .+ 1.0
    Λ = 0.5 .* randn(rng, p, K)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        η = β .+ Λ * randn(rng, K)
        for t in 1:p
            Y[t, s] = rand(rng, Poisson(exp(η[t])))
        end
    end
    return Y
end

@testset "Poisson bootstrap refit reports the fitter's own convergence verdict (#504)" begin
    @testset "adapter parity: converged and θ match an independent direct refit" begin
        # Several simulated datasets, not one seed's numbers — only relations.
        for (p, K, n, seed) in ((4, 1, 120, 41), (5, 1, 150, 42), (3, 1, 90, 43))
            Y = _sim_poisson_verdict(p, K, n; seed = seed)
            fit = fit_poisson_gllvm(Y; K = K)
            @test fit.converged   # sanity: the base fit itself is healthy

            ad = GLLVModels._family_ci(fit, Y)
            Yb = ad.simulate(MersenneTwister(seed + 1000))
            raw = ad.refit(Yb)

            # The migrated contract: a named tuple, not a bare vector — a bare
            # vector has no `.converged` / `.loglik` field at all, which is
            # exactly why this @testset fails outright on `origin/main`.
            @test raw isa NamedTuple
            @test raw.converged isa Bool

            # Independent direct refit on the identical bootstrap draw. The
            # fitter's warm start is a deterministic function of the data (no
            # RNG), so this reproduces the adapter's internal fit exactly.
            fb = fit_poisson_gllvm(Yb; K = K, link = fit.link, hessian = fit.hessian)
            @test raw.converged == fb.converged
            @test raw.θ == vcat(fb.β, GLLVModels.pack_lambda(fb.Λ))
            @test raw.loglik == fb.loglik
        end
    end

    @testset "a non-converged refit is excluded from the bootstrap (n_converged drops)" begin
        Y = _sim_poisson_verdict(4, 1, 120; seed = 44)
        fit = fit_poisson_gllvm(Y; K = 1)
        @test fit.converged
        ad = GLLVModels._family_ci(fit, Y)

        # Deliberately pathological bootstrap draw: one wildly invalid count.
        # Every θ the optimiser tries (including the warm start) throws inside
        # the fitter's own try/catch and lands on the `1e12` failure sentinel,
        # so `_fit_verdict` reports `converged = false` regardless of what
        # `Optim.converged` claims — data-driven, not RNG-driven, so it is
        # stable across the Julia 1.10 / 1.13 CI matrix.
        p, n = size(Y)
        Y_bad = fill(5, p, n)
        Y_bad[1, 1] = -100_000
        fb_bad = fit_poisson_gllvm(Y_bad; K = 1)
        @test !fb_bad.converged     # the fixture really is a genuine non-convergence

        raw_bad = ad.refit(Y_bad)
        @test raw_bad isa NamedTuple
        @test raw_bad.converged == false

        m = length(ad.θ)
        @test !GLLVModels._bootstrap_refit_ok(raw_bad, m)[2]

        # End-to-end: a stubbed `simulate` alternates the healthy original `Y`
        # (whose refit always converges — deterministic warm start) with the
        # pathological `Y_bad` above, so `n_converged` must land strictly
        # between 0 and n_boot: the non-converged replicates were rejected,
        # not all of them, and not none of them.
        n_boot = 20
        ad_mixed = GLLVModels._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds,
                                        rng -> rand(rng) < 0.5 ? Y : Y_bad,
                                        ad.refit)
        result = GLLVModels._family_bootstrap(ad_mixed, collect(1:m), 0.95, n_boot, 5, false)
        @test 0 < result.n_converged < n_boot
    end
end
