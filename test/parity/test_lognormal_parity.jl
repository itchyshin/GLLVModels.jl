# test_lognormal_parity.jl — one-part lognormal GLLVModels logLik vs gllvmTMB (twin fid 3)
#
# Included by runparity.jl. NEVER included by test/runtests.jl.
#
# Identity lock: docs/dev-log/decisions/2026-08-15-lognormal-identity.md (ACCEPTED).
#   · η is the mean of log y  (E[log y] = η); exp(η) is the MEDIAN, not the mean.
#   · Dispersion is a SHARED SCALAR σ (twin `sigma_eps`), NOT per-trait. This is the
#     one no-X family where pairing with a grouped/per-trait fitter would be WRONG.
#   · The reported log-likelihood is on the **y scale** and MUST include the
#     change-of-variables Jacobian −Σ log y. A dropped Jacobian would show up as a
#     data-dependent constant offset that can masquerade as "small disagreement".
#
# Comparability: log y is exactly Gaussian, so BOTH sides evaluate an exact marginal
# (Julia closed-form; twin TMB). This is exact-vs-exact — the strongest case in the
# suite, and a mismatch here is an identity bug, not an approximation gap.
#
# Twin Δ rule: no number is quoted unless this cell runs live under
# GLLVM_PARITY_TESTS=1 against a real gllvmTMB. Never invent, never carry over.

using GLLVModels, RCall, Test, Random, LinearAlgebra

# parity_helpers.jl is included once by runparity.jl

# Seed pre-registered BEFORE any run (forecloses seed-shopping). The plan first
# named 45; seeds 42–49 are already taken by existing cells (45 = NB2 and Beta), so
# this was moved to 52 for receipt uniqueness — a collision fix made before the cell
# had ever been executed, NOT a re-roll after seeing a Δ.
const _LN_SEED = 52

function _lognormal_sim(p, K, n; seed = _LN_SEED)
    Random.seed!(seed)
    β = log.([3.0, 5.0, 2.0, 4.0, 3.5])[1:p]
    Λ = 0.45 .* parity_loadings_p5k2()[1:p, 1:K]
    σ = 0.5
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = exp.(η .+ σ .* randn(p, n))     # strictly positive by construction
    return Y
end

@testset "lognormal GLLVModels parity: GLLVModels.jl vs gllvmTMB (twin fid 3)" begin

    p, K, n = 5, 2, 60
    Y = _lognormal_sim(p, K, n)
    @test all(>(0), Y)

    # NOTE: do NOT centre Y. Julia estimates per-trait β internally (row means of
    # log Y) and the twin fits `0 + trait`; both carry per-trait intercepts, so the
    # Gaussian cell's manual centring does not apply here.
    jl_fit = fit_lognormal_gllvm(Y; K = K)
    @test jl_fit.converged
    @test isfinite(jl_fit.loglik)
    jl_logL = jl_fit.loglik

    # ── Sanity only: σ is finite and positive ────────────────────────────────
    # NOTE (Rose audit 2026-08-24): this is NOT a check that the pairing is
    # shared-scalar. `LognormalFit.σ::Float64` is statically scalar
    # (src/families/lognormal.jl), so any `isa Real` assertion here is a tautology
    # and was removed rather than left to look like evidence.
    #
    # What actually proves the shared-σ pairing is the Δ below: had the twin fitted
    # p free `sigma_eps` against Julia's single σ, the twin's log-likelihood would be
    # strictly and materially higher (an extra p−1 free parameters), and a Δ of ~1e-8
    # would be impossible. The Δ is the evidence; this block is a smoke check.
    @testset "σ finite and positive (smoke; see Δ for the pairing evidence)" begin
        @test isfinite(jl_fit.σ)
        @test jl_fit.σ > 0
    end

    # ── Identity gate 2: Jacobian present (structural reconstruction) ────────
    # Reproduce the documented decomposition: y-scale loglik == Gaussian marginal of
    # the centred log-responses MINUS Σ log y.
    @testset "reported loglik includes the −Σ log y Jacobian (structural)" begin
        Z = log.(Y)
        β̂ = vec(sum(Z; dims = 2)) ./ n
        Rc = Z .- β̂
        gfit = fit_gaussian_gllvm(Rc; K = K)
        @test isapprox(jl_logL, gfit.logLik - sum(Z); atol = 1e-8)
    end

    # ── Identity gate 3: Jacobian has the right FORM (behavioural) ───────────
    # Under y → c·y the centred log-residuals are UNCHANGED (the per-trait intercepts
    # absorb log c), so an exact y-scale log-likelihood must shift by exactly
    # −p·n·log c on BOTH sides, leaving Δ fixed.
    #
    # What this adds over the Δ test (Rose audit 2026-08-24 — the earlier comment
    # here overstated it): a ONE-SIDED dropped Jacobian is already caught by the Δ
    # test, since the offset Σ log y ≈ 375 against a loglik of ≈ −594.67 is a
    # relative error of ~0.6, thousands of times the locked rtol 1e-6. What the Δ
    # test canNOT see is a BOTH-SIDES drop — a shared convention error where the two
    # engines omit the term together and therefore still agree with each other. That
    # is the failure mode this gate exists for, and it also pins the Jacobian's
    # functional form (coefficient p·n, sign negative), not merely its presence.
    @testset "scale-shift test: loglik shifts by −p·n·log c on both sides" begin
        c = 2.0
        expected_shift = -p * n * log(c)

        jl_scaled = fit_lognormal_gllvm(c .* Y; K = K)
        @test jl_scaled.converged
        @test isapprox(jl_scaled.loglik - jl_logL, expected_shift; atol = 1e-6)

        r_base   = fit_gllvmtmb_parity_loglik(Y, K; family = :lognormal)
        r_scaled = fit_gllvmtmb_parity_loglik(c .* Y, K; family = :lognormal)
        @test r_base.converged && r_scaled.converged
        @test isapprox(r_scaled.logLik - r_base.logLik, expected_shift; rtol = 1e-6)

        # Δ is invariant under the rescaling — the property that matters.
        @test isapprox(jl_scaled.loglik - r_scaled.logLik,
                       jl_logL - r_base.logLik; atol = 1e-6)
    end

    # ── Twin oracle + Δ ──────────────────────────────────────────────────────
    r = fit_gllvmtmb_parity_loglik(Y, K; family = :lognormal)
    @test r.converged
    @test isfinite(r.logLik)

    print_parity_loglik(
        "lognormal logLik oracle (seed=$(_LN_SEED), p=$p, K=$K, n=$n; twin fid 3)";
        jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
    )

    core070_record_values!("logLik"; julia = jl_logL, r = r.logLik, rtol = 1e-6,
                           test = "@test jl_logL ≈ r.logLik rtol = 1e-6")
    @testset "log-likelihood agreement (rtol=1e-6)" begin
        @test jl_logL ≈ r.logLik rtol = 1e-6
        @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
    end
end
