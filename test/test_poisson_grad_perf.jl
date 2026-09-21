# Workflow Q gate for the Poisson perf repair (core070, poisson-perf-repair-notes.md).
#
# (a) FD-vs-analytic gradient check on a seeded p=10, n=100, K=2 fixture (≤1e-6).
# (b) Fitted logLik on a seeded p=20, n=500 fixture must match the PRE-REPAIR
#     baseline to ≤1e-8 — see provenance comment below.
# (c)/(d) full-suite green and honest before/after timing are recorded in
#     docs/dev-log/core070/poisson-perf-repair-notes.md, not asserted here (wall
#     time is not a reproducible CI assertion).

using GLLVModels, Test, Random, LinearAlgebra
using Distributions: Poisson as _PoissonDist

@testset "Poisson perf repair — gradient + logLik gates (core070)" begin
    @testset "FD gradient check, p=10, n=100, K=2" begin
        Random.seed!(20260901)
        p, K, n = 10, 2, 100
        β = randn(p) .* 0.3
        Λ = randn(p, K) .* 0.4
        Y = rand(0:8, p, n)

        rr = GLLVModels.rr_theta_len(p, K)
        θ = vcat(β, GLLVModels.pack_lambda(Λ))

        f = function (θv)
            b = θv[1:p]
            L = GLLVModels.unpack_lambda(θv[(p + 1):(p + rr)], p, K)
            return GLLVModels.poisson_marginal_loglik_laplace(Y, L, b, LogLink();
                                                         maxiter = 200, tol = 1e-12)
        end

        m = length(θ)
        g_fd = similar(θ)
        h = 1e-5
        for i in 1:m
            θp = copy(θ); θp[i] += h
            θm = copy(θ); θm[i] -= h
            g_fd[i] = (f(θp) - f(θm)) / (2h)
        end

        g_an = GLLVModels.poisson_laplace_grad(Y, Λ, β)

        @test length(g_an) == m
        @test all(isfinite, g_an)
        maxrel = maximum(abs.(g_an .- g_fd) ./ max.(abs.(g_fd), 1.0))
        @test maxrel <= 1e-6
    end

    @testset "Fitted logLik regression, p=20, n=500, K=2" begin
        # PROVENANCE (Julia < 1.12): baseline captured on the pre-repair commit
        # (before R2/R3/R4), using this exact fixture (MersenneTwister(20260901),
        # Poisson via Distributions.jl), via `fit_poisson_gllvm(Y; K=2)` with default
        # settings (no optimizer tolerance changes made anywhere in the repair).
        # Captured 2026-09-01, GLLVModels.jl branch codex/core070-aghq-20260830,
        # commit b1e704e4. If this test ever needs to change, the cause must be a
        # genuine numerical fix, never a repair-induced drift: re-derive the number
        # by hand, do not copy the post-repair value back in.
        #
        # WHY THERE ARE TWO (D-275, 2026-09-21). The constant above pins a Julia
        # version as well as a number, and did not say so. MersenneTwister's stream
        # changed after 1.10, so `Lam`, `bet` and `Y` below are DIFFERENT DATA on a
        # newer Julia and a log-likelihood pinned to the 1.10 dataset cannot hold on
        # them. Reproduce in one line, no package loaded:
        #     rand(MersenneTwister(20260901), Int, 2)
        #   1.10.0  -> [-2482383529823488692,  6476472913113760869]
        #   1.12.6  -> [-6819447421300569731, -8678045115267176261]
        #   1.13.0  -> [-6819447421300569731, -8678045115267176261]
        # 1.12.6 and 1.13.0 agree, 1.10.0 differs, so the split is placed at 1.12.
        # 1.11 was not tested (not installed here) and CI runs only 1.10 and latest.
        #
        # PROVENANCE (Julia >= 1.12): DERIVED, not copied from a failing CI log.
        # Measured on origin/main 69a69b0a0 under Julia 1.13.0 (pre-S6, pre-S8: the
        # same baseline every identity in this arc gates against), then confirmed
        # bit-identical on this branch. macOS gives ...449 and Linux CI ...444, a
        # 5e-12 absolute difference that atol 1e-8 covers. Re-derive it the same way
        # if it ever moves.
        #
        # THE REAL FIX is to make the fixture stream-stable with StableRNGs, which
        # ten other files in this suite already use. That is its own arc, because it
        # retires the 2026-09-01 pre-repair guarantee rather than carrying it.
        BASELINE_LOGLIK = VERSION < v"1.12" ? -14604.017303313138 :
                                              -13707.693675285449

        rng = Random.MersenneTwister(20260901)
        p, K, n = 20, 2, 500
        Λ = randn(rng, p, K) .* 0.5
        β = randn(rng, p) .* 0.3
        Z = randn(rng, K, n)
        η = β .+ Λ * Z
        μ = exp.(clamp.(η, -5, 5))
        Y = [rand(rng, _PoissonDist(μ[t, s])) for t in 1:p, s in 1:n]

        fit = GLLVModels.fit_poisson_gllvm(Y; K = 2)
        @test fit.converged
        @test isapprox(fit.loglik, BASELINE_LOGLIK; atol = 1e-8, rtol = 0)
    end
end
