# test_truncated_nbinom2_parity.jl — zero-truncated NB2 GLLVModels logLik vs gllvmTMB
# (twin fid 11)
#
# Included by runparity.jl. NEVER included by test/runtests.jl.
#
# Identity lock: docs/dev-log/decisions/2026-08-15-truncated-nbinom2-identity.md.
#   · Dispersion is PER-TRAIT: r_t ≡ twin exp(log_phi_truncnb2[t])
#     (`src/gllvmTMB.cpp:1187-1190`; map `R/fit-multi.R:5354-5355` leaves one free
#     entry per truncNB2 trait). So this cell pairs with
#     `fit_truncated_nbinom2_gllvm_pertrait`, NEVER the shared-scalar
#     `fit_truncated_nbinom2_gllvm` — that would be a second, silent mismatch.
#   · η is on the UNTRUNCATED NB2 mean μ = exp(η); truncation enters as −log(1−p₀).
#   · Log link only (`R/fit-multi.R:844-845`); support y ≥ 1 (`R/fit-multi.R:3119`).
#
# WHY THIS CELL COULD NOT BE PAID BEFORE 2026-08-24, and what changed:
# both Julia truncNB2 routes built their Laplace log-det from the FISHER (expected
# information) weight, with no way to select otherwise, while TMB always uses the
# OBSERVED joint Hessian. For NB2-class likelihoods those differ pointwise, because
# the curvature is y-dependent through −(y+r)·log(μ+r) — unlike truncated Poisson
# (fid 10), where y enters η linearly, observed ≡ Fisher, and the Fisher-core cell
# paid legitimately at ~2.7e-9. Comparing them would have been the same class of
# artifact as the NB1 defect found the same day (0.115 of log-likelihood), except
# with no keyword to flip. `_truncnb2_observed_weight` now supplies the observed
# curvature (derived analytically, verified against ForwardDiff to 1.8e-13 across
# 125 (μ, r, y) cells), and `hessian = :observed` is the default.
#
# Twin Δ rule: no number is quoted unless this cell runs live under
# GLLVM_PARITY_TESTS=1 against a real gllvmTMB. Never invent, never carry over.

using GLLVModels, RCall, Test, Random
import Distributions
include(joinpath(@__DIR__, "truncnb2_policy.jl"))

# parity_helpers.jl is included once by runparity.jl

# Seed pre-registered BEFORE any run (42–49, 420–431, 52–57 already taken).
const _TNB2_SEED = 58

@testset "truncated_nbinom2 GLLVModels parity: GLLVModels.jl vs gllvmTMB (twin fid 11)" begin

    Random.seed!(_TNB2_SEED)
    p, K, n = 5, 1, 120
    # Intercepts chosen so μ ≳ 3: keeps p₀ small (cheap rejection sampling), the
    # truncation correction well conditioned, and the Laplace modes stable.
    β = log.([4.0, 5.0, 3.5, 4.5, 4.0])
    r_true = 4.0
    Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -3.0, 3.5))
        while true                      # zero-truncated draw by rejection
            v = rand(Distributions.NegativeBinomial(r_true, r_true / (r_true + μ)))
            if v >= 1
                Y[t, s] = v
                break
            end
        end
    end

    @testset "support is {1,2,…} (zero-truncated)" begin
        @test all(>=(1), Y)
    end

    # Per-trait dispersion — the pairing the Identity requires.
    jl_fit = fit_truncated_nbinom2_gllvm_pertrait(Y; K = K)
    @test jl_fit.converged
    @test isfinite(jl_fit.loglik)
    @test length(jl_fit.r) == p          # genuinely per-trait, not shared
    jl_logL = jl_fit.loglik

    # Explicit policy, not a silent change to the default oracle helper.
    r = parity_truncnb2_public_bfgs(Y, K, jl_fit)
    @testset "explicit public R continuation and complete fit health" begin
        d=r.policy
        @test d["policy"]=="truncnb2_default_then_public_bfgs_v1"
        @test d["same_data_map"]
        @test d["native_nfree"]==d["r_nfree"]==15
        @test d["r_gradient_max"]<=1e-4
        @test d["native_gradient_max"]<=1e-4
        @test d["fd_stability"]<=1e-4
        @test d["native_objective_delta"]<=1e-8
        @test abs(d["samepoint_delta"])<=1e-6
        core070_record_values!("objective at the R optimum"; julia = d["samepoint_native_nll"],
                               r = d["r_objective"], atol = 1e-6,
                               test = "@test abs(d[\"samepoint_delta\"])<=1e-6")
        @test abs(d["r_loglik"]+d["r_objective"])<=1e-8
        @test all(isfinite,d["native_parameters"]) && all(isfinite,d["r_parameters"])
    end
    @test r.converged
    @test isfinite(r.logLik)

    print_parity_loglik(
        "truncated_nbinom2 logLik oracle (seed=$(_TNB2_SEED), p=$p, K=$K, n=$n, per-trait r; twin fid 11)";
        jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
    )

    core070_record_values!("logLik"; julia = jl_logL, r = r.logLik, rtol = 1e-6,
                           test = "@test jl_logL ≈ r.logLik rtol = 1e-6")
    @testset "log-likelihood agreement (rtol=1e-6)" begin
        @test jl_logL ≈ r.logLik rtol = 1e-6
        @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
    end

    # Regression guard for the 2026-08-24 observed-curvature fix. `:fisher` is a
    # legitimate expected-information objective and stays reachable; it was only ever
    # wrong as the *only* option. Asserting it is a DIFFERENT objective pins the
    # direction of the fix — if these two ever coincide, either the keyword stopped
    # threading through or the observed weight silently reverted to Fisher.
    @testset "`hessian=:fisher` remains reachable and is a different objective" begin
        f_fisher = fit_truncated_nbinom2_gllvm_pertrait(Y; K = K, hessian = :fisher)
        @test f_fisher.converged
        @test !isapprox(f_fisher.loglik, jl_logL; rtol = 1e-6)
    end

    # An invalid symbol must fail loudly. The objective wraps its body in a try/catch
    # that turns any throw into 1e12, so without an up-front check a typo would return
    # a converged-looking garbage fit rather than an error.
    @testset "invalid hessian symbol is rejected up front" begin
        @test_throws ArgumentError fit_truncated_nbinom2_gllvm_pertrait(
            Y; K = K, hessian = :bogus)
    end
end
