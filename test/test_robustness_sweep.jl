using GLLVModels, Test, Random, LinearAlgebra, Distributions

# Robustness sweep (issues #138 #143 #145 #146 #147 #158; #161 is comment-only).

@testset "robustness sweep" begin

    @testset "#138 _bridge_scores does not mask real failures" begin
        # Genuine failure inside getLV must surface, not become a 0x0 matrix.
        @test_throws ErrorException GLLVModels._bridge_scores(() -> error("non-PD residual"))
        @test_throws LinearAlgebra.PosDefException GLLVModels._bridge_scores(
            () -> throw(LinearAlgebra.PosDefException(1)))
        # Success passes through unchanged.
        @test GLLVModels._bridge_scores(() -> [1.0 2.0; 3.0 4.0]) == [1.0 2.0; 3.0 4.0]
        # Documented degradation: a getLV signature that does not apply
        # (MethodError) still yields an empty matrix.
        @test size(GLLVModels._bridge_scores(() -> sin("not a number"))) == (0, 0)
    end

    @testset "#143 transformed-Wald Sigma rejects an indefinite Hessian" begin
        Hpd  = [2.0 0.3; 0.3 1.0]
        Hind = [2.0 0.0; 0.0 -1.0]       # invertible but indefinite
        Σ, pd = GLLVModels._tw_sigma_from_hessian_matrix(Hpd)
        @test pd
        @test Σ ≈ inv(Hpd)
        Σ2, pd2 = GLLVModels._tw_sigma_from_hessian_matrix(Hind)
        @test !pd2
        @test Σ2 === nothing
        Σ3, pd3 = GLLVModels._tw_sigma_from_hessian_matrix([1.0 2.0; 2.0 4.0])  # singular
        @test !pd3 && Σ3 === nothing
    end

    @testset "#145 em_fa default init is reproducible; rng keyword" begin
        Random.seed!(3)
        p, K, n = 6, 2, 300
        Λt = [0.8 0; 0.5 0.4; 0.3 -0.2; -0.1 0.3; 0.2 0.1; -0.3 0.4]
        y = rand(MvNormal(zeros(p), Symmetric(Λt * Λt' + 0.3I)), n)
        Random.seed!(1)
        a = GLLVModels.em_fa(y, K; max_iter = 3, tol = 0.0)
        Random.seed!(2)
        b = GLLVModels.em_fa(y, K; max_iter = 3, tol = 0.0)
        @test a[1] == b[1]                       # independent of global RNG state
        c = GLLVModels.em_fa(y, K; max_iter = 3, tol = 0.0, rng = MersenneTwister(99))
        d = GLLVModels.em_fa(y, K; max_iter = 3, tol = 0.0, rng = MersenneTwister(99))
        @test c[1] == d[1]
        @test c[1] != a[1]
    end

    @testset "#146 SQUAREM premature-stop fallback never returns a worse point" begin
        Random.seed!(17)
        tree = augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
        p = tree.n_leaves
        Σ = GLLVModels.sigma_phy_dense(tree; σ²_phy = 1.0)
        Λ = reshape([0.8, 0.6, 0.4, -0.3, 0.5, -0.2], p, 1)
        y = Λ * randn(1, 300) .+ 0.9 .* (cholesky(Symmetric(Σ)).L * randn(p)) .+ 0.5 .* randn(p, 300)
        # Loose tol forces a premature stop; the loose-tol plain-EM fallback
        # stops early too, so it can be worse than the polished point.
        kw = (tol = 0.5, max_iter = 200)
        raw = em_fit_phylo_squarem(y, 1, Σ; kw..., safety_check = false)
        chk = em_fit_phylo_squarem(y, 1, Σ; kw..., safety_polish_iters = 200,
                                   safety_warmstart = false)
        # Reference: the polished log-lik reachable from the SQUAREM point.
        polished = em_fit_phylo_squarem(y, 1, Σ; tol = 1e-9, max_iter = 2000,
                                        safety_check = false)
        @test chk.logLik ≥ raw.logLik - 1e-9
        @test raw.logLik < polished.logLik - 1e-3      # premise: stop was premature
        @test chk.logLik > raw.logLik + 1e-3           # polish point used, not θ_sq
    end

    @testset "#147 Beta density clamps y away from 0 and 1" begin
        f = GLLVModels.Beta(10.0, 1.0)
        for y in (0.0, 1.0)
            @test isfinite(GLLVModels._glm_logpdf(f, 0.5, 1, y))
            @test isfinite(GLLVModels._glm_score(f, 0.5, 1, 1.0, y))
        end
        # clamp matches gllvmTMB (1e-12): boundary == clamped value
        @test GLLVModels._glm_logpdf(f, 0.5, 1, 0.0) ==
              GLLVModels._glm_logpdf(f, 0.5, 1, 1e-12)
        @test GLLVModels._glm_logpdf(f, 0.5, 1, 1.0) ==
              GLLVModels._glm_logpdf(f, 0.5, 1, 1 - 1e-12)
        # interior unchanged
        for y in (1e-6, 0.3, 0.9)
            @test GLLVModels._glm_logpdf(f, 0.4, 1, y) ==
                  logpdf(Distributions.Beta(0.4 * 10.0, 0.6 * 10.0), y)
        end
    end

    @testset "#158 contrasts no-loadings == dense with ones(p) phylo loading" begin
        Random.seed!(5)
        tree = augmented_phy("(((A:0.1,B:0.1):0.1,C:0.2):0.1,(D:0.2,E:0.2):0.1);")
        p = tree.n_leaves
        Λ = reshape([0.7, 0.5, 0.4, -0.3, 0.2], p, 1)
        y = randn(p, 40)
        s2 = 1.7
        Σ = GLLVModels.sigma_phy_dense(tree; σ²_phy = s2)
        ll_c = gaussian_marginal_loglik_contrasts(y, Λ, 0.6; tree = tree, σ²_phy = s2)
        ll_h = GLLVModels.gaussian_marginal_loglik(y, Λ, 0.6; σ_phy = ones(p), Σ_phy = Σ)
        ll_n = GLLVModels.gaussian_marginal_loglik(y, Λ, 0.6; Σ_phy = Σ)   # no loadings: no phylo block
        @test ll_c ≈ ll_h rtol = 1e-10
        @test !isapprox(ll_c, ll_n; rtol = 1e-6)
    end
end
