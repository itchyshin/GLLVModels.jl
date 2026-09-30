# test_binomial_parity.jl — Binomial (Bernoulli) GLLVModels logLik vs gllvmTMB
#
# Included by runparity.jl. NEVER included by test/runtests.jl.
# Same-model bar: per-trait intercepts + latent unique=FALSE (no Ψ).
# Inventory: docs/dev-log/plans/scratch/2026-08-01-correctness-inventory.md

using GLLVModels, RCall, Test, Random, LinearAlgebra

# parity_helpers.jl is included once by runparity.jl

@testset "Binomial GLLVModels parity: GLLVModels.jl vs gllvmTMB" begin
    Random.seed!(43)
    p, K, n = 5, 2, 60
    β = [-0.5, 0.0, 0.5, -0.2, 0.3]
    Λ = parity_loadings_p5k2()
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = [rand() < 1 / (1 + exp(-η[t, s])) ? 1 : 0 for t in 1:p, s in 1:n]

    jl_fit = fit_binomial_gllvm(Y; K = K)
    @test jl_fit.converged
    @test isfinite(jl_fit.loglik)
    jl_logL = jl_fit.loglik

    r = fit_gllvmtmb_parity_loglik(Y, K; family = :binomial)
    @test r.converged
    @test isfinite(r.logLik)

    print_parity_loglik(
        "Binomial logLik oracle (seed=43, p=$p, K=$K, n=$n, Bernoulli)";
        jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
    )

    core070_record_values!("logLik"; julia = jl_logL, r = r.logLik, rtol = 1e-6,
                           test = "@test jl_logL ≈ r.logLik rtol = 1e-6")
    @testset "log-likelihood agreement (rtol=1e-6)" begin
        @test jl_logL ≈ r.logLik rtol = 1e-6
        @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
    end
end
