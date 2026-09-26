using GLLVModels, Test, Random, LinearAlgebra, Distributions

# _fit_verdict(res) (fit_verdict.jl) reported converged = true whenever Optim.converged(res)
# was true. Optim 1.13.3's `converged = x_converged || f_converged || g_converged`, with
# x_abstol = x_reltol = f_abstol = f_reltol = 0.0 by default, so a single zero-length
# line-search step trips x_converged/f_converged trivially even when the gradient residual
# is nowhere near g_tol. Audit evidence (docs/dev-log/core070/class-audit-20260924/
# fit_verdict_classB_probe.jl, _fit_verdict_run1.log, itself unmerged): fit_nb1_gllvm_grouped
# on 3/3 seeded no-X datasets (seeds 101-103) reported converged = true with Optim gradient
# residuals of 5.68, 2.14e2 and 1.79e1 against g_tol = 1e-5 (scaled threshold ~1e-2).
#
# Fix: `_fit_verdict(res)` now also requires the gradient criterion
# gres <= max(g_tol, g_tol * |nll|), the scale-aware rule `_tweedie_verdict`
# (families/tweedie.jl) and the Beta grouped verdict (families/grouped_dispersion.jl,
# `_beta_grouped_g_met`, #483/#480) already use.
#
# Data generation matches the audit's `route_nb1` exactly (seed 101, defaults
# p=5, n=80, K=2, sd=0.7, φ=1.0, βlo=0.5, βhi=1.5) so this reproduces the real defect
# rather than a fabricated Optim result.

function _nb1_verdict_lowtri(rng, p, K, sd)
    Λ = sd .* randn(rng, p, K)
    for j in 1:K, i in 1:(j - 1)
        Λ[i, j] = 0.0
    end
    for j in 1:K
        Λ[j, j] = abs(Λ[j, j]) + 0.3 * sd
    end
    return Λ
end

function _nb1_verdict_fixture(seed; p = 5, n = 80, K = 2, sd = 0.7, φ = 1.0, βlo = 0.5, βhi = 1.5)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p)
    Λ = _nb1_verdict_lowtri(rng, p, K, sd)
    Z = randn(rng, K, n)
    Y = [begin
            μ = exp(clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30))
            rand(rng, NegativeBinomial(μ / φ, 1 / (1 + φ)))
         end for t in 1:p, i in 1:n]
    return Y
end

@testset "_fit_verdict requires the gradient criterion (#485)" begin
    @testset "NB1 grouped: honesty correction on a real stalled fit (seed 101)" begin
        Y = _nb1_verdict_fixture(101)
        fit = GLLVModels.fit_nb1_gllvm_grouped(Y; K = 2, group = collect(1:5))
        # Before the fix: converged = true, loglik ≈ -1038.5504, on a zero-length
        # line-search step (Optim g residual ≈ 5.68 against g_tol = 1e-5).
        @test fit.loglik ≈ -1038.5503750549412 atol = 1e-6
        @test !fit.converged
    end

    @testset "NB1 grouped: a genuinely stationary smoke fit stays converged" begin
        # Same generator/route as test_grouped_dispersion_tweedie_nb1.jl's "per-species
        # smoke" case (seed 703, tiny p=4,n=50) — guards against the fix over-correcting
        # a fit whose gradient really is small.
        rng = Random.MersenneTwister(703)
        p, K, n = 4, 1, 50
        β = 0.3 .* randn(rng, p)
        Λ = 0.3 .* randn(rng, p, K)
        φtrue = 1.0
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            η = β .+ Λ * randn(rng, K)
            μ = exp.(η)
            for t in 1:p
                Y[t, s] = rand(rng, NegativeBinomial(μ[t] / φtrue, 1 / (1 + φtrue)))
            end
        end
        fit = GLLVModels.fit_nb1_gllvm_grouped(Y; K = K, group = collect(1:p))
        @test isfinite(fit.loglik)
        @test fit.converged
    end
end
