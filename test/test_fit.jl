using GLLVModels, Test, Random, LinearAlgebra

@testset "fit (smoke)" begin
    @testset "function exists and returns the expected struct" begin
        # We can't run a full recovery test until J1-B-1 + J1-B-2 land
        # together (they each test their own piece in isolation). Here
        # we just check the function signature and struct shape.
        Random.seed!(0)
        p, K, n = 4, 2, 60
        Λ_true = [0.7 0; 0.5 0.4; 0.3 -0.2; -0.1 0.3]
        σ_true = 1.0
        η      = randn(K, n)
        y      = Λ_true * η + σ_true * randn(p, n)

        fit = fit_gaussian_gllvm(y; K = K)
        @test isa(fit, GllvmFit)
        @test isa(fit.model, GllvmModel)
        @test fit.model.p == p
        @test fit.model.K == K
        @test size(fit.pars.Λ) == (p, K)
        @test fit.pars.σ_eps > 0
        @test isfinite(fit.logLik)
        @test fit.cputime > 0
        @test fit.converged
    end

    @testset "recovery on a clean fixture" begin
        # Light recovery test: σ_eps within 10%, Λ frobenius within ~2× sd.
        # We don't enforce per-entry recovery because rotation is not pinned
        # (lower-triangular doesn't fully pin in this MVP; see scope).
        Random.seed!(1)
        p, K, n = 5, 1, 200
        Λ_true = reshape([0.6, 0.5, 0.4, -0.3, 0.2], p, K)
        σ_true = 0.5
        η      = randn(K, n)
        y      = Λ_true * η + σ_true * randn(p, n)

        fit = fit_gaussian_gllvm(y; K = K)
        @test fit.converged
        @test fit.pars.σ_eps ≈ σ_true rtol=0.10
        # Σ_y recovery (rotation-invariant): compare ΛΛᵀ + σ²I
        Σ_true = Λ_true * Λ_true' + σ_true^2 * I
        Σ_hat  = fit.pars.Λ * fit.pars.Λ' + fit.pars.σ_eps^2 * I
        @test norm(Σ_true - Σ_hat) / norm(Σ_true) < 0.10
    end
end

@testset "fit_gaussian_gllvm default stops at the optimum, not on f_tol" begin
    # Regression: the default relative f_tol = 1e-10 stopped iterative fits
    # (phylo-unique, covariates) after ~29 LBFGS steps with a gradient near
    # 3e-3, while reporting converged = true; the Wald SEs there were up to
    # 0.25% off those at the optimum. The default now leaves stopping to g_tol.
    using Optim: g_residual
    Random.seed!(30)
    tree = GLLVModels.augmented_phy("(((((A:0.2,B:0.2):0.2,C:0.4):0.2,(D:0.3,E:0.3):0.3):0.1," *
                                    "((F:0.2,G:0.2):0.3,H:0.5):0.1):0.1,(I:0.4,J:0.4):0.2);")
    Σ = GLLVModels.sigma_phy_dense(tree; σ²_phy = 1.0)
    Λ = reshape([0.7, 0.5, -0.4, 0.3, 0.6, -0.5, 0.4, 0.2, 0.8, -0.3], 10, 1)
    y = Λ * randn(1, 500) .+ reshape(0.8 .* (cholesky(Symmetric(Σ)).L * randn(10)), 10, 1) .+
        0.5 .* randn(10, 500)
    fit = fit_gaussian_gllvm(y; K = 1, has_phy_unique = true, Σ_phy = Σ)
    @test fit.converged
    @test g_residual(fit.optim_result) < 1e-5
    ref = fit_gaussian_gllvm(y; K = 1, has_phy_unique = true, Σ_phy = Σ,
                             f_tol = 0.0, x_tol = 0.0, g_tol = 1e-9, iterations = 5_000)
    se, se_ref = confint(fit; y = y, Σ_phy = Σ).se, confint(ref; y = y, Σ_phy = Σ).se
    @test maximum(abs.(se .- se_ref) ./ abs.(se_ref)) < 1e-4
end
