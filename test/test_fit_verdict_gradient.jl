using GLLVModels, Test, Random, LinearAlgebra, Distributions

# _fit_verdict(res) (fit_verdict.jl) is the shared, family-agnostic helper used across
# ~90 fitters: it screens the failure-sentinel plateau (nll >= _NLL_FAIL_THRESHOLD) but
# otherwise trusts Optim.converged(res) as given, unchanged from origin/main.
#
# #485's false-converged report is scoped to `fit_nb1_gllvm_grouped`
# (families/grouped_dispersion.jl): Optim's `converged = x_converged || f_converged ||
# g_converged`, and this fitter's `Optim.Options(g_tol = g_tol, iterations = iterations)`
# leaves `x_abstol = x_reltol = f_abstol = f_reltol = 0.0` (Optim 1.13.3 defaults), so a
# single zero-length line-search step trips x_converged/f_converged trivially even while
# the gradient residual sits orders of magnitude above `g_tol`. Audit evidence (docs/
# dev-log/core070/class-audit-20260924/fit_verdict_classB_probe.jl,
# _fit_verdict_run1.log, itself unmerged): fit_nb1_gllvm_grouped on 3/3 seeded no-X
# datasets (seeds 101-103) reported converged = true with Optim gradient residuals of
# 5.68, 2.14e2 and 1.79e1 against g_tol = 1e-5.
#
# MAINTAINER DECISION (2026-09-26, PR #502 comment): option (d), per-family verdicts,
# not a blanket change to the shared `_fit_verdict`. An earlier attempt (PR #502) widened
# the gradient criterion into `_fit_verdict` itself and broke five unrelated suites
# (test_twolevel.jl, and the four test_phylo_{poisson,beta,binomial,gamma}_xlv.jl files)
# whose fitters take a legitimate x/f-converged exit with a small but nonzero configured
# tolerance (fit_gaussian_gllvm in src/fit.jl and aghq_gaussian_fit.jl both set nonzero
# x/f tolerances) at a gradient this blanket rule rejected; it also flagged genuine
# optima in fit_gamma_gllvm where the objective has small jumps. `_nb1_grouped_g_met`
# (families/grouped_dispersion.jl), applied only inside `fit_nb1_gllvm_grouped`, follows
# the existing `_tweedie_verdict` (families/tweedie.jl) / `_beta_grouped_g_met`
# (families/grouped_dispersion.jl, #480/#483) precedent instead: `gres <= max(g_tol,
# g_tol * |nll|)`, scale-aware so a caller's g_tol set below the finite-difference noise
# floor does not turn a genuine stationary point into a false negative.
#
# PLATFORM ROBUSTNESS. Optim's finite-difference L-BFGS path is not bit-reproducible
# across BLAS/LAPACK builds: the same seed can take a different number of steps, stall
# at a different zero-length step, or land in a different local optimum on Linux x86_64
# vs macOS ARM64. None of these tests pin an exact loglik or an exact gradient value for
# that reason: (a) asserts the qualitative flip this fix exists for (Optim's own flag
# says converged, the gradient plainly is not small, so the fit must report
# not-converged); (b) checks a relation — whenever `fit.converged`, the gradient is
# small — across several seeds, which must hold regardless of which optimiser path any
# given platform takes; (c) checks a fit that is actually at a stationary point stays
# converged, again without pinning the numeric optimum.

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

# Largest central-difference gradient of the NB1 grouped negative log-likelihood at the
# fitted point, computed from the public marginal (`nb1_grouped_marginal_loglik_laplace`)
# and not from Optim's own bookkeeping — an independent check that a "converged" fit is
# actually stationary, matching test_beta_grouped_convergence.jl's `_beta480_max_grad`.
function _nb1_verdict_max_grad(Y, β, Λ, φg, gidx)
    p, K = size(Λ)
    rr = GLLVModels.rr_theta_len(p, K)
    G = length(φg)
    θ = vcat(β, GLLVModels.pack_lambda(Λ), log.(φg))
    function f(θ)
        β_ = θ[1:p]
        Λ_ = GLLVModels.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φg_ = exp.(θ[(p + rr + 1):(p + rr + G)])
        φvec = [φg_[gidx[t]] for t in 1:p]
        return -GLLVModels.nb1_grouped_marginal_loglik_laplace(Y, Λ_, β_, φvec)
    end
    h = 1e-5
    return maximum(eachindex(θ)) do j
        e = zeros(length(θ)); e[j] = h
        abs(f(θ .+ e) - f(θ .- e)) / (2h)
    end
end

@testset "_nb1_grouped_g_met scopes the gradient criterion to fit_nb1_gllvm_grouped (#485)" begin
    @testset "seed 101: the zero-length-step stall no longer reports converged" begin
        # Data generation matches the audit's `route_nb1` exactly (seed 101, defaults
        # p=5, n=80, K=2, sd=0.7, φ=1.0, βlo=0.5, βhi=1.5). Before this fix (both
        # origin/main and PR #502's blanket `_fit_verdict` change reverted here):
        # fit.converged == true although the fit sits on a zero-length line-search
        # step with a large gradient. No loglik pin — the optimiser path (and hence
        # which non-stationary point it stalls at) is not bit-reproducible across
        # platforms.
        Y = _nb1_verdict_fixture(101)
        fit = GLLVModels.fit_nb1_gllvm_grouped(Y; K = 2, group = collect(1:5))
        @test isfinite(fit.loglik)
        gmax = _nb1_verdict_max_grad(Y, fit.β, fit.Λ, fit.φ, fit.group)
        # Confirms this really is the non-stationary stall the fix targets, not a
        # coincidentally-small gradient at a point Optim happened to converge on.
        @test gmax > 1e-3
        @test !fit.converged
    end

    @testset "relation over several seeds: fit.converged implies a small gradient" begin
        # Whatever optimiser path a given platform takes on a given seed, a fit this
        # reports as converged must actually be close to stationary. Unlike the seed
        # 101 case above, this does not require any particular seed to stall — it
        # only requires the implication to hold wherever `converged` is asserted.
        for seed in 101:110
            Y = _nb1_verdict_fixture(seed)
            fit = GLLVModels.fit_nb1_gllvm_grouped(Y; K = 2, group = collect(1:5))
            if fit.converged
                @test _nb1_verdict_max_grad(Y, fit.β, fit.Λ, fit.φ, fit.group) < 1e-2
            end
        end
    end

    @testset "seed 703: a genuinely stationary smoke fit stays converged" begin
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
        @test _nb1_verdict_max_grad(Y, fit.β, fit.Λ, fit.φ, fit.group) < 1e-2
    end
end
