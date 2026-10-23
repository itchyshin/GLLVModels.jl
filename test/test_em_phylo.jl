using GLLVModels, Test, Random, LinearAlgebra, Distributions, SparseArrays, Statistics

# Gradient-free EM for the Gaussian phylo_unique GLLVModels. NEW FILES not wired
# into the GLLVModels module on this branch (matching the PERF+++++ hard constraint
# "do NOT modify any existing src/test/Project.toml file"). Pull them in
# directly for the test. `em_phylo.jl` calls `GLLVModels.gaussian_marginal_loglik`
# and `GLLVModels.ppca_init` via the loaded module, and consumes the
# `AugmentedPhy` from the GLLVModels-exported `augmented_phy`.
# em_phylo.jl is included in GLLVModels.jl

# Helper: simulate phylo_unique data with K_B site factors plus one shared
# per-trait phylo random effect z = diag(σ_phy) φ, φ ~ N(0, Σ_phy).
function _sim_phylo_unique(tree, Λ_B, σ_phy, σ_eps, n; seed = 0)
    Random.seed!(seed)
    Σ_phy = GLLVModels.sigma_phy_dense(tree; σ²_phy = 1.0)
    p, K_B = size(Λ_B)
    η_B = randn(K_B, n)
    φ   = cholesky(Symmetric(Σ_phy)).L * randn(p)
    z   = σ_phy .* φ
    y   = Λ_B * η_B .+ reshape(z, p, 1) .+ σ_eps .* randn(p, n)
    return y, Σ_phy
end

@testset "EM phylo (phylo_unique, fast sparse solves)" begin

    # -- Shared K_B = 1 fixture (seed 22: interior all-positive optimum) --------
    tree1   = augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p1      = tree1.n_leaves
    Λ_B1    = reshape([0.8, 0.6, 0.4, -0.3, 0.5, -0.2], p1, 1)
    σ_phy1  = fill(0.9, p1)
    n1      = 400
    y1, Σ1  = _sim_phylo_unique(tree1, Λ_B1, σ_phy1, 0.5, n1; seed = 22)

    # Fit once with assert_monotone=true and reuse across the fixture-1 tests
    fit1 = fit_gaussian_gllvm(y1; K = 1, has_phy_unique = true, Σ_phy = Σ1)
    emf1 = em_fit_phylo(y1, 1, Σ1; tol = 1e-10, max_iter = 10_000, assert_monotone = true)

    @testset "CORRECTNESS GATE: EM matches dense MLE (K_B = 1)" begin
        @test fit1.converged
        @test emf1.converged

        # logLik agrees to ~1e-4 (the primary gate). Observed ≈ 1e-9.
        @test abs(emf1.logLik - fit1.logLik) < 1e-4

        # Loadings agree up to sign/rotation. K_B = 1 ⇒ Λ_B Λ_B' is the
        # rotation-invariant quantity; observed max abs diff ≈ 6e-7.
        @test maximum(abs.(fit1.pars.Λ * fit1.pars.Λ' .- emf1.Λ_B * emf1.Λ_B')) < 1e-2
        # σ_eps and σ_phy agree to ~1e-2 (observed ≈ 3e-8 and ≈ 5e-5).
        @test abs(fit1.pars.σ_eps - emf1.σ_eps) < 1e-2
        @test maximum(abs.(fit1.pars.σ_phy .- emf1.σ_phy)) < 1e-2
        # This fixture's optimum is interior to σ_phy > 0.
        @test all(emf1.σ_phy .> 0)
    end

    @testset "EM log-lik is monotone non-decreasing (EM invariant)" begin
        # assert_monotone = true (default) would error on a decrease; here we
        # also check the trace directly. Tiny round-off slack of 1e-9.
        incs = diff(emf1.loglik_trace)
        @test minimum(incs) ≥ -1e-9
        @test length(emf1.loglik_trace) ≥ 2
    end

    @testset "CORRECTNESS GATE: EM matches dense MLE (K_B = 2)" begin
        # Larger balanced tree (p = 10), K_B = 2, interior all-positive optimum.
        tree2 = augmented_phy("((((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2," *
                              "((E:0.3,F:0.3):0.2,(G:0.3,H:0.3):0.2):0.2):0.2," *
                              "(I:0.5,J:0.5):0.2);")
        p2 = tree2.n_leaves
        Random.seed!(14)
        Λ_B2 = randn(p2, 2)
        for k in 1:2, i in 1:(k - 1)
            Λ_B2[i, k] = 0.0
        end
        σ_phy2 = fill(0.9, p2)
        n2 = 600
        y2, Σ2 = _sim_phylo_unique(tree2, Λ_B2, σ_phy2, 0.5, n2; seed = 14)

        fit = fit_gaussian_gllvm(y2; K = 2, has_phy_unique = true, Σ_phy = Σ2)
        @test fit.converged
        emf = em_fit_phylo(y2, 2, Σ2; λ_init = fit.pars.Λ, σ_eps_init = fit.pars.σ_eps,
                           σ_phy_init = fit.pars.σ_phy, tol = 1e-10, max_iter = 10_000)
        @test emf.converged

        @test abs(emf.logLik - fit.logLik) < 1e-4              # observed ≈ 2e-6
        # K_B = 2: compare the rotation-invariant Λ_B Λ_B' (observed ≈ 7e-5).
        @test maximum(abs.(fit.pars.Λ * fit.pars.Λ' .- emf.Λ_B * emf.Λ_B')) < 1e-2
        @test abs(fit.pars.σ_eps - emf.σ_eps) < 1e-2
        @test maximum(abs.(fit.pars.σ_phy .- emf.σ_phy)) < 1e-2
    end

    @testset "fast sparse (A + n B)⁻¹ solve == dense to machine precision" begin
        # The E-step's dominant linear solve is (A + n B)⁻¹ applied to vectors.
        # `solve_AnB` reuses the augmented-state saddle point of
        # `likelihood_sparse_phy.jl` and must equal the dense Cholesky solve.
        tree_aug = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
        Σ        = GLLVModels.sigma_phy_dense(tree_aug; σ²_phy = 1.0)
        p        = tree_aug.n_leaves
        Random.seed!(7)
        Λ_B   = randn(p, 2)
        σ_eps = 0.5
        σ_phy = abs.(randn(p)) .+ 0.2
        n     = 50
        A   = Λ_B * Λ_B' + σ_eps^2 * I
        Bm  = (σ_phy * σ_phy') .* Σ
        AnB = A + n .* Bm

        solver = build_AnB_sparse(Λ_B, σ_eps, σ_phy, tree_aug, n)
        for _ in 1:5
            rhs = randn(p)
            v_sparse = solve_AnB(solver, rhs)
            v_dense  = AnB \ rhs
            @test v_sparse ≈ v_dense rtol = 1e-9
        end
    end

    @testset "ancestral-state BLUP extraction (dense == sparse == EM)" begin
        # μ_z = n B (A + n B)⁻¹ m is the BLUP of the phylo effect on the data
        # scale. The EM returns it from the final E-step; the sparse path
        # computes it without forming Σ_phy. All three must agree.
        # Dense BLUP formula evaluated at the EM parameters.
        A   = emf1.Λ_B * emf1.Λ_B' + emf1.σ_eps^2 * I
        Bm  = (emf1.σ_phy * emf1.σ_phy') .* Σ1
        m   = vec(mean(y1, dims = 2))
        μ_z_dense = n1 .* Bm * ((A + n1 .* Bm) \ m)

        # EM's stored BLUP (from the final E-step, dense solves).
        @test emf1.blup_phy ≈ μ_z_dense rtol = 1e-6 atol = 1e-8
        @test length(emf1.blup_phy) == p1

        # Sparse augmented-Q BLUP (never forms Σ_phy).
        tree_aug   = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
        μ_z_sparse = blup_phylo_sparse(y1, emf1.Λ_B, emf1.σ_eps, emf1.σ_phy, tree_aug)
        @test μ_z_sparse ≈ μ_z_dense rtol = 1e-8 atol = 1e-9
    end

    @testset "EM reproduces dense log-lik when evaluated at EM params" begin
        # The EM trajectory is measured with the SAME dense closed form the
        # gradient fit uses, so the reported logLik equals a fresh evaluation.
        ll  = GLLVModels.gaussian_marginal_loglik(y1, emf1.Λ_B, emf1.σ_eps;
                                             σ_phy = emf1.σ_phy, Σ_phy = Σ1)
        @test emf1.logLik ≈ ll rtol = 1e-12
    end

    @testset "HONEST NOTE: EM reaches a mixed-sign σ_phy optimum, no worse than the dense fit" begin
        # `fit_gaussian_gllvm` fits a signed σ_phy (identity link, with a
        # single-flip sign-pattern search and a global sign anchor); it no
        # longer restricts σ_phy = exp(log_σ_phy) > 0, which is the case this
        # test was first written for. The EM per-trait WLS is also
        # unconstrained in the sign of σ_phy. Seed 17 is a fixture whose
        # optimum over signed σ_phy uses a negative coupling: EM reaches a
        # mixed-sign σ_phy, and its likelihood is at least the dense fit's
        # (to 1e-6). EM still monotonically increases its own likelihood.
        tree = augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
        p    = tree.n_leaves
        Λ_B  = reshape([0.8, 0.6, 0.4, -0.3, 0.5, -0.2], p, 1)
        y, Σ = _sim_phylo_unique(tree, Λ_B, fill(0.9, p), 0.5, 400; seed = 17)

        fit = fit_gaussian_gllvm(y; K = 1, has_phy_unique = true, Σ_phy = Σ)
        emf = em_fit_phylo(y, 1, Σ; tol = 1e-10, max_iter = 10_000)
        # EM monotone (its own invariant holds regardless of the dense fit).
        @test minimum(diff(emf.loglik_trace)) ≥ -1e-9
        # Here EM finds a mixed-sign σ_phy optimum.
        @test !all(emf.σ_phy .> 0)
        # EM's likelihood is ≥ the dense fit's (both optimise signed σ_phy).
        @test emf.logLik ≥ fit.logLik - 1e-6
    end
end
