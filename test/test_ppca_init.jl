using GLLVModels, Test, Random, LinearAlgebra, Distributions

@testset "PPCA closed-form initialisation" begin
    @testset "exact recovery on pure PPCA fixture (K=1)" begin
        Random.seed!(0)
        # n bumped from 2000 → 5000 so that sampling error in the
        # sample covariance (∝ sqrt(p/n)) is comfortably below the 5%
        # tolerance the test enforces against the population covariance.
        p, K, n = 5, 1, 5000
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3, 0.2], p, K)
        σ_true = 0.5
        η = randn(K, n)
        y = Λ_true * η + σ_true * randn(p, n)

        λ_init, σ_init = GLLVModels.ppca_init(y, K)
        # Σ_y recovery (rotation-invariant)
        Σ_true = Λ_true * Λ_true' + σ_true^2 * I
        Σ_init = λ_init * λ_init' + σ_init^2 * I
        @test norm(Σ_true - Σ_init) / norm(Σ_true) < 0.05
        @test σ_init ≈ σ_true rtol = 0.05
    end

    @testset "exact recovery on pure PPCA fixture (K=2)" begin
        Random.seed!(1)
        p, K, n = 6, 2, 3000
        Λ_true = [0.8 0;
                  0.5 0.6;
                  0.4 0.3;
                 -0.3 -0.2;
                  0.2 0.4;
                 -0.1 0.5]
        σ_true = 0.5
        η = randn(K, n)
        y = Λ_true * η + σ_true * randn(p, n)

        λ_init, σ_init = GLLVModels.ppca_init(y, K)
        Σ_true = Λ_true * Λ_true' + σ_true^2 * I
        Σ_init = λ_init * λ_init' + σ_init^2 * I
        @test norm(Σ_true - Σ_init) / norm(Σ_true) < 0.05
        @test σ_init ≈ σ_true rtol = 0.05
    end

    @testset "lower-triangular structure preserved" begin
        Random.seed!(2)
        p, K, n = 8, 3, 1000
        Λ_true = randn(p, K); for i in 1:K, k in 1:K; if i < k; Λ_true[i, k] = 0; end; end
        for k in 1:K; Λ_true[k, k] = abs(Λ_true[k, k]) + 0.5; end
        y = Λ_true * randn(K, n) + 0.5 * randn(p, n)
        λ_init, _ = GLLVModels.ppca_init(y, K)
        # Check strict-upper is zero (after rotation)
        for i in 1:K, k in (i+1):K
            @test abs(λ_init[i, k]) < 1e-10
        end
        # Diagonals positive
        for k in 1:K
            @test λ_init[k, k] > 0
        end
    end

    @testset "warm-start makes fit converge in fewer iterations" begin
        # Compare LBFGS iteration count: default init vs PPCA init
        Random.seed!(3)
        p, K, n = 5, 1, 1000
        Λ_true = reshape([0.7, 0.5, 0.4, -0.3, 0.2], p, K)
        η = randn(K, n)
        y = Λ_true * η + 0.5 * randn(p, n)

        fit_default = fit_gaussian_gllvm(y; K = K)
        λ_init, σ_init = GLLVModels.ppca_init(y, K)
        fit_warm = fit_gaussian_gllvm(y; K = K, λ_init = λ_init, σ_eps_init = σ_init)

        # Same answer (Σ_y recovery)
        Σ_default = fit_default.pars.Λ * fit_default.pars.Λ' + fit_default.pars.σ_eps^2 * I
        Σ_warm = fit_warm.pars.Λ * fit_warm.pars.Λ' + fit_warm.pars.σ_eps^2 * I
        @test norm(Σ_default - Σ_warm) / norm(Σ_default) < 1e-3

        # Warm-start needs at most as many iterations as default; usually fewer
        @test fit_warm.n_iter ≤ fit_default.n_iter
        @info "PPCA warm-start iterations: default=$(fit_default.n_iter), warm=$(fit_warm.n_iter)"
    end
end
