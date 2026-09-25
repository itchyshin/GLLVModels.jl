# test_ordinal_probit_parity.jl — ordinal-probit GLLVModels logLik vs gllvmTMB
#
# Included by runparity.jl. NEVER included by test/runtests.jl.
# Same-model bar: cumulative-probit link, per-trait intercepts, τ₁ = 0,
# and latent unique=FALSE (no Ψ).

using GLLVModels, RCall, Test, Random

@testset "Ordinal-probit GLLVModels diagnostic: GLLVModels.jl vs gllvmTMB" begin
    Random.seed!(46)
    p, K, n, C = 5, 1, 60, 3
    β = [0.30, -0.20, 0.15, -0.10, 0.05]
    Λ = reshape([0.8, 0.5, 0.3, -0.2, 0.1], p, K)
    τ = [0.0, 0.75]
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n, t in 1:p
        u = rand()
        Y[t, s] = u < GLLVModels._ord_F(τ[1] - η[t, s], ProbitLink()) ? 1 :
                  u < GLLVModels._ord_F(τ[2] - η[t, s], ProbitLink()) ? 2 : 3
    end

    jl_fit = fit_ordinal_gllvm_pertrait(Y; K = K, link = ProbitLink(),
                                        g_tol = 1e-7, iterations = 1_000)
    @test jl_fit.converged
    @test isfinite(jl_fit.loglik)

    # parity_helpers.jl is included once by runparity.jl
    _parity_require_gllvmtmb!()
    @rput Y K p n
    R"""
    trait_names <- paste0("t", seq_len(p))
    df_long <- data.frame(
        site  = factor(rep(seq_len(n), each = p)),
        trait = factor(rep(trait_names, times = n), levels = trait_names),
        value = as.vector(Y)
    )
    fit_r <- gllvmTMB(
        value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
        data = df_long,
        unit = "site",
        trait = "trait",
        family = ordinal_probit(),
        control = gllvmTMBcontrol(n_init = 1L, se = FALSE)
    )
    .ordinal_probit_parity_last <<- list(
        logL = as.numeric(stats::logLik(fit_r)),
        objective = as.numeric(fit_r$opt$objective),
        converged = identical(as.integer(fit_r$opt$convergence), 0L)
    )
    """
    r_logL = rcopy(Float64, R".ordinal_probit_parity_last$logL")
    r_obj = rcopy(Float64, R".ordinal_probit_parity_last$objective")
    r_converged = rcopy(Bool, R".ordinal_probit_parity_last$converged")

    @test r_converged
    println("Ordinal-probit diagnostic: Julia = $(jl_fit.loglik), ",
            "R = $r_logL, Δ = $(jl_fit.loglik - r_logL)")
    @test r_logL ≈ -r_obj rtol = 0 atol = 1e-10
    @test jl_fit.loglik ≈ r_logL rtol = 1e-6
end
