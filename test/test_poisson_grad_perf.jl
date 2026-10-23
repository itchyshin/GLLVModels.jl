# Workflow Q gate for the Poisson perf repair (core070, poisson-perf-repair-notes.md).
#
# (a) FD-vs-analytic gradient check on a seeded p=10, n=100, K=2 fixture (≤1e-6).
# (b) Fitted logLik on a StableRNG-seeded p=20, n=500 fixture must match the
#     pinned baseline to ≤1e-8 — see provenance comment below.
# (c)/(d) full-suite green and honest before/after timing are recorded in
#     docs/dev-log/core070/poisson-perf-repair-notes.md, not asserted here (wall
#     time is not a reproducible CI assertion).
#
# Standalone run: StableRNGs is a test-only dependency, so `julia --project=.`
# cannot load this file; run it through the test environment instead, e.g.
#   julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path=".");
#             Pkg.add(["StableRNGs", "Distributions"]);
#             include("test/test_poisson_grad_perf.jl")'

using GLLVModels, Test, Random, LinearAlgebra
using StableRNGs
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
        # PROVENANCE: FRESH BASELINE, captured 2026-09-21 (D-275). The fixture
        # draws from StableRNGs.StableRNG(20260901), whose stream is fixed by the
        # package and not by the Julia release, so one constant holds on every
        # Julia version. Derived by running this exact fixture through
        # `fit_poisson_gllvm(Y; K=2)` with default settings, on branch
        # claude/poisson-pin-stablerng-20260921 (off claude/lane-speed78-20260919
        # at eb48adcc2), macOS arm64:
        #   Julia 1.10.0 -> -14741.53968547299   (converged)
        #   Julia 1.13.0 -> -14741.539685473015  (converged)
        # The data are byte-identical on both (SHA-256 of Λ, β and Y match;
        # sum(Y) = 17592); the 2.5e-11 gap is BLAS reduction-order noise that
        # atol 1e-8 covers. If this number ever needs to change, the cause must be
        # a genuine numerical fix, never drift: re-derive it by hand on two Julia
        # versions and do not copy a failing CI value back in.
        #
        # WHAT THIS RETIRED. Until 2026-09-21 the constant was -14604.017303313138,
        # captured on the PRE-REPAIR commit b1e704e4 (2026-09-01, branch
        # codex/core070-aghq-20260830) from MersenneTwister(20260901), so that a
        # repair-induced drift (R2/R3/R4) could not pass unnoticed. MersenneTwister's
        # stream changed after Julia 1.10 (`rand(MersenneTwister(20260901), Int, 2)`
        # differs on 1.12+), which made that pin a Julia-version pin as well and
        # forced a version-keyed pair for one day (9db03e8e9). Changing the RNG
        # changes the dataset, so the 2026-09-01 pre-repair guarantee cannot be
        # carried across and is retired here. It was honoured to the end: the
        # MersenneTwister pin still passed on Julia 1.10.0 at the moment of the
        # switch. From here the constant guards drift relative to THIS capture.
        BASELINE_LOGLIK = -14741.53968547299

        rng = StableRNG(20260901)
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
