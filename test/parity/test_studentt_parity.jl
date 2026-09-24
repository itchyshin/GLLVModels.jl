# test_studentt_parity.jl — Student-t GLLVModels logLik vs gllvmTMB
# (twin fid 9)
#
# Included by runparity.jl. NEVER included by test/runtests.jl.
#
# Parameterisation note:
# docs/dev-log/decisions/2026-08-28-studentt-parameterisation.md (twin's df
# profile-CI off-by-one bug; irrelevant here — logLik only, no CI).
# docs/dev-log/decisions/2026-08-28-per-trait-dispersion-synthesis.md (why the
# cause below was PREDICTED before being measured).
#
# The twin's default estimates `df` per trait (`student(df = NULL)`), and
# Julia's public `nu = nothing, disp_group = :species` route estimates the
# matching per-trait `ν` and scale in `src/families/studentt.jl`. The fixed-ν
# call below is a control: BOTH sides fix `df = ν_true`
# (`gllvmTMB::student(df = ν_true)`), isolating the one remaining structural
# difference under the shared Julia scale: the twin's `log_sigma_student` is a
# PER-TRAIT `n_traits`-length TMB parameter (`gllvmTMB.cpp:1184`; no
# shared/pinned mode is exposed through the family constructor), while
# Julia's `fit_studentt_gllvm` defaults to a single SHARED scalar `σ`
# (`disp_group = :shared`). Under a shared-σ DGP the twin has `p−1` more free
# dispersion parameters, so its logLik is generically >= Julia's shared-σ
# fit even under a correct fit on both sides — exactly the delta-cell
# pattern (`test_delta_lognormal_parity.jl`, commit 6c471352).
#
# `disp_group = :species` (2026-08-28) closes that scale-grouping gap. With
# `nu = nothing`, it also estimates per-trait ν; that public estimated-ν target
# remains separate from the fixed-ν control. Near the Gaussian limit ν can be
# weakly identified, so cross-evaluating ν is diagnostic only, never an
# equality requirement.
#
# Twin Δ rule: no number is quoted unless this cell runs live under
# GLLVM_PARITY_TESTS=1 against a real gllvmTMB. Never invent, never carry over.

using GLLVModels, RCall, Test, Random, Distributions

# parity_helpers.jl is included once by runparity.jl

# Seed pre-registered BEFORE any run (42–49, 52–58, 61-62, 420–431 already
# taken; see test_delta_lognormal_parity.jl / test_delta_gamma_parity.jl).
const _ST_SEED = 71

@testset "Student-t GLLVModels parity: GLLVModels.jl vs gllvmTMB (twin fid 9)" begin

    Random.seed!(_ST_SEED)
    p, K, n = 5, 1, 130
    β_true = [0.2, -0.1, 0.3, 0.0, -0.2]
    Λ_true = 0.5 .* parity_loadings_p5k2()[:, 1:K]
    σ_true = 0.7
    ν_true = 4.0

    Z = randn(K, n)
    η = β_true .+ Λ_true * Z          # p×n
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = η[t, s] + σ_true * rand(TDist(ν_true))
    end

    r = fit_gllvmtmb_parity_student(Y, K; df_fixed = ν_true)
    @test r.converged
    @test r.optimizer_code == 0
    @test isfinite(r.logLik)
    @test all(≈(ν_true; atol = 1e-8), r.df_vec)   # df genuinely pinned, not estimated

    @testset "shared σ (Julia default) — measured baseline mismatch" begin
        jl_fit = fit_studentt_gllvm(Y; K = K, nu = ν_true)
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        @test jl_fit.disp_group === :shared

        print_parity_loglik(
            "student logLik oracle, disp_group=:shared (seed=$(_ST_SEED), p=$p, " *
            "K=$K, n=$n, df fixed = $ν_true on both sides; twin fid 9)";
            jl_logL = jl_fit.loglik, r_logL = r.logLik, r_obj = r.objective,
        )
        println("  Julia β (shared σ=$(round(jl_fit.σ; sigdigits=4))) = ",
                round.(jl_fit.β; sigdigits = 5))
        println("  gllvmTMB b_fix (per-trait σ)  = ", round.(r.b_fix; sigdigits = 5))
        println("  gllvmTMB per-trait σ vector   = ", round.(r.sigma_vec; sigdigits = 5))
        println()

        # NOT asserted at rtol=1e-6 — the twin's extra p−1 free dispersion
        # parameters generically make r.logLik >= jl_logL under a shared-σ DGP;
        # report the actual numbers instead of a pass/fail gate here.
        @test r.logLik >= jl_fit.loglik - 1e-6   # twin should never do WORSE than Julia
        @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
    end

    @testset "per-trait σ (disp_group = :species) — closes the mismatch" begin
        jl_fit = fit_studentt_gllvm(Y; K = K, nu = ν_true, disp_group = :species,
                                    iterations = 400)
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        @test jl_fit.disp_group === :species
        @test jl_fit.σ isa Vector{Float64}

        print_parity_loglik(
            "student logLik oracle, disp_group=:species (seed=$(_ST_SEED), p=$p, " *
            "K=$K, n=$n, df fixed = $ν_true on both sides; twin fid 9)";
            jl_logL = jl_fit.loglik, r_logL = r.logLik, r_obj = r.objective,
        )
        println("  Julia per-trait σ  = ", round.(jl_fit.σ; sigdigits = 5))
        println("  gllvmTMB per-trait σ vector = ", round.(r.sigma_vec; sigdigits = 5))
        println()

        @testset "log-likelihood agreement (fixed-ν control, absolute Δ ≤ 0.001)" begin
            @test abs(r.logLik - jl_fit.loglik) ≤ 0.001
        end
    end

    @testset "per-trait σ + per-trait estimated ν (twin default) — Parity Cell 9" begin
        r_est = fit_gllvmtmb_parity_student(Y, K; df_fixed = nothing)
        # Read right after the R fit, while `fit_r` is still this cell's fit.
        # Recorded, not gated: docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-pins.md.
        r_gradient_max = rcopy(Float64, R"max(abs(as.numeric(fit_r$tmb_obj$gr(fit_r$opt$par))))")
        @test r_est.converged
        @test r_est.optimizer_code == 0
        @test isfinite(r_est.logLik)

        jl_est = fit_studentt_gllvm(Y; K = K, nu = nothing, disp_group = :species,
                                    iterations = 400)
        @test jl_est.converged
        @test isfinite(jl_est.loglik)
        @test jl_est.disp_group === :species
        @test jl_est.σ isa Vector{Float64}
        @test jl_est.ν isa Vector{Float64}
        @test all(>(1.0), jl_est.ν)

        print_parity_loglik(
            "student logLik oracle, disp_group=:species, estimated ν (seed=$(_ST_SEED), p=$p, " *
            "K=$K, n=$n; twin fid 9)";
            jl_logL = jl_est.loglik, r_logL = r_est.logLik, r_obj = r_est.objective,
        )
        println("  Julia per-trait σ  = ", round.(jl_est.σ; sigdigits = 5))
        println("  gllvmTMB per-trait σ = ", round.(r_est.sigma_vec; sigdigits = 5))
        println("  Julia per-trait ν  = ", round.(jl_est.ν; sigdigits = 5))
        println("  gllvmTMB per-trait ν = ", round.(r_est.df_vec; sigdigits = 5))
        println("  gllvmTMB optimizer code/message/iterations = ",
                (r_est.optimizer_code, r_est.optimizer_message, r_est.optimizer_iterations))
        println("  gllvmTMB r_gradient_max = ", r_gradient_max, " (recorded, not a gate)")
        flat_boundary = any(>(1e6), jl_est.ν) || any(>(1e6), r_est.df_vec)
        println("  flat Gaussian-limit boundary diagnosed = ", flat_boundary)
        println()

        # A large ν on either engine is a flat-boundary diagnosis, not a
        # parameter-equality target. The original fixture's admissible result
        # is instead the absolute likelihood difference and both fit-health
        # checks above.
        @testset "log-likelihood agreement (public estimated-ν target, absolute Δ ≤ 0.001)" begin
            @test abs(jl_est.loglik - r_est.logLik) ≤ 0.001
        end
    end

    @testset "well-identified estimated-ν diagnostic" begin
        Random.seed!(72)
        p_diag, K_diag, n_diag = 3, 1, 400
        β_diag = [0.2, -0.1, 0.3]
        Λ_diag = reshape([0.45, -0.35, 0.25], p_diag, K_diag)
        σ_diag = [0.6, 0.8, 0.7]
        ν_diag = 4.0
        η_diag = β_diag .+ Λ_diag * randn(K_diag, n_diag)
        Y_diag = [η_diag[t, s] + σ_diag[t] * rand(TDist(ν_diag))
                  for t in 1:p_diag, s in 1:n_diag]

        r_diag = fit_gllvmtmb_parity_student(Y_diag, K_diag; df_fixed = nothing)
        jl_diag = fit_studentt_gllvm(Y_diag; K = K_diag, nu = nothing,
                                     disp_group = :species, g_tol = 1e-7,
                                     iterations = 800)
        @test r_diag.converged
        @test r_diag.optimizer_code == 0
        @test jl_diag.converged
        @test isfinite(r_diag.logLik) && isfinite(jl_diag.loglik)
        # This is an independent diagnosis and regression guard. It supports,
        # but never substitutes for, the original seed-71 target above.
        @test abs(jl_diag.loglik - r_diag.logLik) ≤ 0.001
    end

    @testset "near-Gaussian estimated-ν diagnostic" begin
        Random.seed!(73)
        p_diag, K_diag, n_diag = 3, 1, 400
        β_diag = [0.2, -0.1, 0.3]
        Λ_diag = reshape([0.45, -0.35, 0.25], p_diag, K_diag)
        σ_diag = [0.6, 0.8, 0.7]
        ν_diag = 1.0e6
        η_diag = β_diag .+ Λ_diag * randn(K_diag, n_diag)
        Y_diag = [η_diag[t, s] + σ_diag[t] * rand(TDist(ν_diag))
                  for t in 1:p_diag, s in 1:n_diag]

        r_diag = fit_gllvmtmb_parity_student(Y_diag, K_diag; df_fixed = nothing)
        jl_diag = fit_studentt_gllvm(Y_diag; K = K_diag, nu = nothing,
                                     disp_group = :species, g_tol = 1e-7,
                                     iterations = 800)
        @test r_diag.converged
        @test r_diag.optimizer_code == 0
        @test jl_diag.converged
        @test isfinite(r_diag.logLik) && isfinite(jl_diag.loglik)
        # ν is inherently weakly identified near the Gaussian limit. Do not
        # force its equality or turn cross-evaluation into a substitute for
        # the original seed-71 gate. Retain the difference as a diagnostic.
        println("  near-Gaussian diagnostic Δ logLik (jl − r) = ",
                jl_diag.loglik - r_diag.logLik)
        println("  near-Gaussian diagnostic ν (Julia, R) = ",
                (jl_diag.ν, r_diag.df_vec))
    end
end
