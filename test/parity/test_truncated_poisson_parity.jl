# test_truncated_poisson_parity.jl — zero-truncated Poisson GLLVModels logLik vs gllvmTMB
# (twin fid 10)
#
# Included by runparity.jl. NEVER included by test/runtests.jl.
#
# Identity lock: docs/dev-log/decisions/2026-08-15-truncated-poisson-identity.md
# (ACCEPTED).
#   · η is on the UNTRUNCATED Poisson mean scale: μ = exp(η)  (TMB `lambda_t`).
#   · Observation law is Poisson truncated at zero; support is {1,2,…}.
#   · Log link only.
#   · The twin's `linkinv` returns the TRUNCATED mean λ/(1−e^{−λ}) for GLM display.
#     That is a display transform, NOT the likelihood parameterisation — so this
#     cell compares the log-likelihood ONLY and never a mean-scale quantity.
#   · No dispersion parameter (contrast with truncated_nbinom2, fid 11).
#
# Comparability: Laplace on BOTH sides (Julia `fit_truncated_poisson_gllvm` is
# Laplace + LBFGS; twin fid 10 is TMB Laplace). This is approximation-vs-
# approximation, so triage a mismatch in this order: (a) inner-mode convergence,
# (b) outer convergence, (c) identity.
#
# Twin Δ rule: no number is quoted unless this cell runs live under
# GLLVM_PARITY_TESTS=1 against a real gllvmTMB. Never invent, never carry over.

using GLLVModels, RCall, Test, Random, LinearAlgebra

# parity_helpers.jl is included once by runparity.jl

# Seed pre-registered BEFORE any run. The plan first named 46; seeds 42–49 are
# already taken by existing cells (46 = ordinal-probit), so this moved to 53 for
# receipt uniqueness — a collision fix made before execution, not a re-roll.
const _TP_SEED = 53

# Knuth sampler (local name — the Poisson cell defines its own `_rand_poisson`).
function _tp_rand_poisson(λ::Float64)
    λ = clamp(λ, 0.0, 1e6)
    L = exp(-λ)
    k = 0
    prod = 1.0
    while true
        k += 1
        prod *= rand()
        prod <= L && return k - 1
    end
end

# Zero-truncated draw by rejection. β is chosen so λ ≳ 2, which keeps the
# rejection loop cheap (P(0) = e^{−λ} is small) and the Laplace modes stable.
function _tp_rand_trunc_poisson(λ::Float64)
    while true
        k = _tp_rand_poisson(λ)
        k >= 1 && return k
    end
end

@testset "truncated_poisson GLLVModels parity: GLLVModels.jl vs gllvmTMB (twin fid 10)" begin

    Random.seed!(_TP_SEED)
    p, K, n = 5, 2, 60
    β = log.([3.0, 5.0, 2.0, 4.0, 3.5])
    Λ = 0.45 .* parity_loadings_p5k2()          # milder loadings keep modes stable
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = [_tp_rand_trunc_poisson(exp(clamp(η[t, s], -8.0, 8.0))) for t in 1:p, s in 1:n]

    # ── Identity gate: support matches the lock ──────────────────────────────
    @testset "support is {1,2,…} (zero-truncated)" begin
        @test all(>=(1), Y)
        @test all(y -> y == round(y), Y)
    end

    # Do NOT centre Y — Julia estimates per-trait β and the twin fits `0 + trait`.
    jl_fit = fit_truncated_poisson_gllvm(Y; K = K)
    @test jl_fit.converged
    @test isfinite(jl_fit.loglik)
    jl_logL = jl_fit.loglik

    r = fit_gllvmtmb_parity_loglik(Y, K; family = :truncated_poisson)
    @test r.converged
    @test isfinite(r.logLik)

    print_parity_loglik(
        "truncated_poisson logLik oracle (seed=$(_TP_SEED), p=$p, K=$K, n=$n; twin fid 10)";
        jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
    )

    core070_record_values!("logLik"; julia = jl_logL, r = r.logLik, rtol = 1e-6,
                           test = "@test jl_logL ≈ r.logLik rtol = 1e-6")
    @testset "log-likelihood agreement (rtol=1e-6)" begin
        @test jl_logL ≈ r.logLik rtol = 1e-6
        @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
    end
end
