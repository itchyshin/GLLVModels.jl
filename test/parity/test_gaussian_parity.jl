# test_gaussian_parity.jl — Gaussian GLLVModels parity: GLLVModels.jl vs R gllvmTMB
#
# Included by runparity.jl after the env gate and RCall load succeed.
# NEVER included by test/runtests.jl.
#
# Call-shape source of truth (2026-08-01 recon):
#   docs/dev-log/plans/scratch/2026-08-01-gaussian-rcall-shape.md
#
# ── Why rotation-invariant quantities only? ──────────────────────────────────
# The loading matrix Λ has two non-identifiable symmetries under a Gaussian
# GLLVModels:
#
#   1. Rotation invariance: for any orthogonal Q, ΛQ gives the same marginal
#      covariance ΛΛᵀ and thus the same likelihood.
#   2. Column-sign flip.
#
# Compare only: marginal log-likelihood, Σ_y = ΛΛᵀ + σ²_eps I, σ_eps.

using GLLVModels, RCall, Test, Random, LinearAlgebra, Statistics

@testset "Gaussian GLLVModels parity: GLLVModels.jl vs gllvmTMB" begin

    # ── 1. Simulate data (tiny fixed seed) ───────────────────────────────────
    Random.seed!(42)
    p, K, n = 5, 2, 80   # small: p traits, K latent factors, n sites

    Λ_true = [
        0.8   0.0;   # lower-triangular canonical form (top K×K block)
        0.5   0.6;
        0.3  -0.4;
       -0.2   0.5;
        0.1   0.3
    ]
    σ_true = 0.7

    η = randn(K, n)                       # K × n latent scores
    y = Λ_true * η + σ_true * randn(p, n) # p × n (traits × sites)

    # Centre per trait so gllvmTMB `0+trait` intercepts match Julia's zero-mean
    # J1 model (see scratch/2026-08-01-gaussian-rcall-shape.md §2).
    y .-= mean(y; dims = 2)

    # ── 2. Julia fit via GLLVModels.jl ────────────────────────────────────────────
    jl_fit = fit_gaussian_gllvm(y; K = K)

    @test jl_fit.converged
    @test isfinite(jl_fit.logLik)

    jl_logL  = jl_fit.logLik
    jl_Λ     = jl_fit.pars.Λ
    jl_σ_eps = jl_fit.pars.σ_eps
    jl_Σ_y   = jl_Λ * jl_Λ' + jl_σ_eps^2 * I(p)

    # ── 3. R fit via gllvmTMB (primary twin — not CRAN gllvm) ────────────────
    # Model alignment:
    #   latent(..., unique = FALSE) → no Ψ; Σ = ΛΛᵀ + σ²I like Julia
    #   per-trait-centred Y         → intercepts do not shift the objective
    # parity_helpers.jl is included once by runparity.jl
    _parity_require_gllvmtmb!()
    @rput y K p n

    # Assign into R global env so extractors are unambiguous; R""" return
    # value alone is an RObject and is NOT named `r_result` on the R side.
    R"""
        # y arrives as p × n (traits × sites). Build long data for explicit
        # traits(...,) / latent formula (byte-equivalent to wide form).
        trait_names <- paste0("t", seq_len(p))
        df_long <- data.frame(
            site  = factor(rep(seq_len(n), each = p)),
            trait = factor(rep(trait_names, times = n), levels = trait_names),
            value = as.vector(y)   # column-major on p×n ⇒ site blocks
        )

        fit_r <- gllvmTMB(
            value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
            data = df_long,
            unit = "site",
            trait = "trait",
            family = gaussian(),
            control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
        )

        r_logL  <- as.numeric(stats::logLik(fit_r))
        r_sigma <- as.numeric(fit_r$report$sigma_eps)
        Sig_sh  <- extract_Sigma(fit_r, level = "unit", part = "shared")$Sigma
        Sigma_y_r <- Sig_sh + diag(r_sigma^2, p)

        .gllvm_parity_gauss <<- list(
            logL      = r_logL,
            sigma     = r_sigma,
            Sigma_y   = Sigma_y_r,
            objective = as.numeric(fit_r$opt$objective),
            converged = identical(as.integer(fit_r$opt$convergence), 0L)
        )
        invisible(NULL)
    """

    r_logL   = rcopy(Float64, R".gllvm_parity_gauss$logL")
    r_sigma  = rcopy(Float64, R".gllvm_parity_gauss$sigma")
    r_Σ_y    = rcopy(Matrix{Float64}, R".gllvm_parity_gauss$Sigma_y")
    r_obj    = rcopy(Float64, R".gllvm_parity_gauss$objective")
    r_conv   = rcopy(Bool, R".gllvm_parity_gauss$converged")
    @test r_conv

    # Always print the numbers — verify by log, not exit code alone.
    println()
    println("── Gaussian logLik oracle (seed=42, p=$p, K=$K, n=$n, centred) ──")
    println("  Julia logLik          = ", jl_logL)
    println("  gllvmTMB logLik       = ", r_logL)
    println("  gllvmTMB -objective   = ", -r_obj)
    println("  Δ logLik (jl − r)     = ", jl_logL - r_logL)
    println("  Julia σ_eps           = ", jl_σ_eps)
    println("  gllvmTMB σ_eps        = ", r_sigma)
    println("  Δ σ_eps               = ", jl_σ_eps - r_sigma)
    println()

    # ── 4. Parity assertions (logLik first; then rotation-invariant extras) ──
    @testset "log-likelihood agreement (rtol=1e-6)" begin
        @test jl_logL ≈ r_logL rtol=1e-6
        @test r_logL ≈ -r_obj rtol=0 atol=1e-10   # logLik == -objective
    end

    @testset "residual SD σ_eps agreement (rtol=1e-4)" begin
        @test jl_σ_eps ≈ r_sigma rtol=1e-4
    end

    @testset "fitted covariance Σ_y agreement (atol=1e-4)" begin
        for i in 1:p, j in 1:p
            @test jl_Σ_y[i, j] ≈ r_Σ_y[i, j] atol=1e-4
        end
    end

end  # @testset
