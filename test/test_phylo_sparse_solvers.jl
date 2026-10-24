using GLLVModels, Test, Random, LinearAlgebra, SparseArrays, Statistics

# Direct coverage for the exported sparse phylo solver surface:
#   * AnBSparseSolver / build_AnB_sparse / solve_AnB / blup_phylo_sparse (src/em_phylo.jl)
#   * node_dσ_phy_only, NodePerSpecies, node_blups (src/node_gradient.jl)
#   * the fit_em_phylo / fit_phylo_squarem aliases
# Every sparse result is compared with plain dense linear algebra built from the
# matrix the docstring defines: A = Λ_B Λ_B' + σ²_eps I, B = diag(σ_phy) Σ_phy diag(σ_phy).

# Fixture builder: same shape as `make_em_fixture` in test_sparse_phy_identities.jl,
# with `n` observations per species.
function _pss_fixture(p; K_B = 1, n = 3, σ_phy_scale = 0.9, σ_eps = 0.5, seed = 30)
    Random.seed!(seed)
    phy   = GLLVModels.random_balanced_tree(p; branch_length = 0.1)
    Σ_phy = GLLVModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    Λ_B   = randn(p, K_B)
    for k in 1:K_B, i in 1:(k - 1)
        Λ_B[i, k] = 0.0
    end
    σ_phy = σ_phy_scale .* (0.8 .+ 0.4 .* rand(p))   # non-constant per-species SDs
    η_B   = randn(K_B, n)
    φ     = cholesky(Symmetric(Σ_phy)).L * randn(p)
    y     = Λ_B * η_B .+ reshape(σ_phy .* φ, p, 1) .+ σ_eps .* randn(p, n)
    return (; phy, Σ_phy, Λ_B, σ_phy, σ_eps, y, n, p)
end

# Dense (A + n B) with Σ_phy scaled by σ²_phy.
function _pss_dense_AnB(fx, σ²_phy)
    A = fx.Λ_B * fx.Λ_B' + fx.σ_eps^2 * I
    Σ = GLLVModels.sigma_phy_dense(fx.phy; σ²_phy = σ²_phy)
    B = Diagonal(fx.σ_phy) * Σ * Diagonal(fx.σ_phy)
    return A, B, A + fx.n * B
end

@testset "phylo sparse solvers" begin

    @testset "solve_AnB matches dense solve (p=$p, K_B=$K_B)" for p in (8, 24), K_B in (1, 2)
        fx = _pss_fixture(p; K_B = K_B, seed = 100 + p + K_B)
        σ²_phy = 1.3
        s = build_AnB_sparse(fx.Λ_B, fx.σ_eps, fx.σ_phy, fx.phy, fx.n; σ²_phy = σ²_phy)
        @test s isa AnBSparseSolver
        _, _, M = _pss_dense_AnB(fx, σ²_phy)

        rhs1 = randn(p)
        @test maximum(abs.(solve_AnB(s, rhs1) .- M \ rhs1)) < 1e-9

        # The factorisation is reusable: a second rhs, then the first again,
        # must give the same answers (no state is consumed by a solve).
        rhs2 = randn(p)
        @test maximum(abs.(solve_AnB(s, rhs2) .- M \ rhs2)) < 1e-9
        @test maximum(abs.(solve_AnB(s, rhs1) .- M \ rhs1)) < 1e-9

        # chol_cache path factorises to the same operator.
        s2 = build_AnB_sparse(fx.Λ_B, fx.σ_eps, fx.σ_phy, fx.phy, fx.n;
                              σ²_phy = σ²_phy, chol_cache = Ref{Any}(nothing))
        @test maximum(abs.(solve_AnB(s2, rhs1) .- M \ rhs1)) < 1e-9
    end

    @testset "blup_phylo_sparse matches dense BLUP (p=$p, K_B=$K_B)" for p in (8, 24), K_B in (1, 2)
        fx = _pss_fixture(p; K_B = K_B, seed = 200 + p + K_B)
        σ²_phy = 0.8
        _, B, M = _pss_dense_AnB(fx, σ²_phy)
        Σ = GLLVModels.sigma_phy_dense(fx.phy; σ²_phy = σ²_phy)
        Bm = Diagonal(fx.σ_phy) * Σ * Diagonal(fx.σ_phy)
        m = vec(mean(fx.y, dims = 2))
        μ_ref = fx.n .* (Bm * (M \ m))
        μ_z = blup_phylo_sparse(fx.y, fx.Λ_B, fx.σ_eps, fx.σ_phy, fx.phy; σ²_phy = σ²_phy)
        @test length(μ_z) == p
        @test maximum(abs.(μ_z .- μ_ref)) < 1e-9
    end

    @testset "node_dσ_phy_only equals central FD of the sparse loglik (p=$p, K_B=$K_B)" for p in (8, 24), K_B in (1, 2)
        fx = _pss_fixture(p; K_B = K_B, seed = 300 + p + K_B)
        st = GLLVModels.build_sparse_phy_state(fx.y, fx.Λ_B, fx.σ_eps;
                                               σ_phy = fx.σ_phy, phy = fx.phy)
        g = node_dσ_phy_only(st)
        @test length(g) == p
        f(sp) = GLLVModels.sparse_phy_value(
            GLLVModels.build_sparse_phy_state(fx.y, fx.Λ_B, fx.σ_eps; σ_phy = sp, phy = fx.phy))
        gfd = similar(g)
        h = 1e-6
        for t in 1:p
            sp⁺ = copy(fx.σ_phy); sp⁺[t] += h
            sp⁻ = copy(fx.σ_phy); sp⁻[t] -= h
            gfd[t] = (f(sp⁺) - f(sp⁻)) / (2h)
        end
        @test maximum(abs.(g .- gfd)) / max(1.0, maximum(abs.(gfd))) < 1e-6
        # Same block as the full node gradient.
        @test g ≈ node_grad(st).dσ_phy
    end

    @testset "NodePerSpecies / node_blups match dense phylogenetic BLUP (p=$p)" for p in (8, 24)
        fx = _pss_fixture(p; seed = 400 + p)
        σ²_eps = 0.45
        μ = 0.2
        y = fx.y[:, 1]
        stn = build_node_perspecies(fx.phy, fx.σ_phy, σ²_eps)
        @test stn isa NodePerSpecies
        û, ẑ_tip = node_blups(stn, y, μ)
        # Dense: ẑ = B (B + σ²_eps I)⁻¹ (y − μ), B = Λ_φ Σ_phy Λ_φ with σ²_phy folded in.
        Σ = GLLVModels.sigma_phy_dense(fx.phy; σ²_phy = 1.0)
        B = Diagonal(fx.σ_phy) * Σ * Diagonal(fx.σ_phy)
        z_ref = B * ((B + σ²_eps * I) \ (y .- μ))
        @test length(ẑ_tip) == p
        @test maximum(abs.(ẑ_tip .- z_ref)) < 1e-9
        @test length(û) == stn.nb
    end

    @testset "fit_em_phylo / fit_phylo_squarem aliases" begin
        @test fit_em_phylo === em_fit_phylo
        @test fit_phylo_squarem === em_fit_phylo_squarem

        fx = _pss_fixture(8; n = 40, seed = 500)
        kw = (; max_iter = 1, assert_monotone = false)
        a = fit_em_phylo(fx.y, 1, fx.Σ_phy; kw...)
        b = em_fit_phylo(fx.y, 1, fx.Σ_phy; kw...)
        @test a isa EMPhyloFit
        @test a.n_iter == b.n_iter
        @test a.Λ_B == b.Λ_B && a.σ_eps == b.σ_eps && a.σ_phy == b.σ_phy
        @test a.logLik == b.logLik

        c = fit_phylo_squarem(fx.y, 1, fx.Σ_phy; max_iter = 1, assert_monotone = false)
        d = em_fit_phylo_squarem(fx.y, 1, fx.Σ_phy; max_iter = 1, assert_monotone = false)
        @test c isa EMPhyloFit
        @test c.n_iter == d.n_iter
        @test c.Λ_B == d.Λ_B && c.σ_eps == d.σ_eps && c.σ_phy == d.σ_phy
        @test c.logLik == d.logLik
    end
end
