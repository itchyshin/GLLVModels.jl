# Lane cursor-cloud-1125y14 — src/precision_multivariate.jl, src/precision_fit_admission.jl, src/phylo_dep.jl
using GLLVModels, Test, LinearAlgebra, SparseArrays, Random

const G = GLLVModels

function _cov14_precision_fixture()
    q_base = [4.0 -1.0 -1.0  0.0;
              -1.0  3.0  0.0 -1.0;
              -1.0  0.0  3.0  0.0;
               0.0 -1.0  0.0  3.0]
    q = sparse(2.5 .* q_base)
    ii, jj, xx = findnz(q)
    pp = PrecisionPhy(ii, jj, xx, 4, 2,
        ["ancestor_1", "ancestor_2", "species_A", "species_B"],
        logdet(cholesky(Symmetric(Matrix(q)))), 2.5, [3, 4])
    y = [0.4 -0.7; -0.3 0.1]
    loading = [0.8 0.0; -0.2 0.5]
    psi = [0.6, 0.5]
    return pp, y, loading, psi
end

@testset "cov 1125y14 phylo dep and precision kernel" begin
    rng = MersenneTwister(11251407)
    phy, y, loading, psi = _cov14_precision_fixture()

    @testset "multivariate_phylo_precision_loglik valid" begin
        ll = G.multivariate_phylo_precision_loglik(
            y, phy, loading, psi; species_id = [1, 2], sigma2_phy = 0.7)
        @test isfinite(ll)
        @test ll < 0
    end

    @testset "multivariate_phylo_precision guards" begin
        @test_throws ArgumentError G.multivariate_phylo_precision_loglik(
            y, phy, loading, psi; species_id = [1, 2], sigma2_phy = -1.0)
        @test_throws ArgumentError G.multivariate_phylo_precision_loglik(
            y, phy, loading[1:1, :], psi; species_id = [1, 2])
        @test_throws ArgumentError G.multivariate_phylo_precision_loglik(
            y, phy, zeros(2, 0), psi; species_id = [1, 2])
        badpsi = [0.0, 0.5]
        @test_throws ArgumentError G.multivariate_phylo_precision_loglik(
            y, phy, loading, badpsi; species_id = [1, 2])
        @test_throws ArgumentError G.multivariate_phylo_precision_loglik(
            y, phy, loading, psi; species_id = [1, 99])
    end

    @testset "_validate_precision_fit_input" begin
        admitted = G._validate_precision_fit_input(phy)
        @test admitted.Q !== phy.Q
        @test Matrix(admitted.Q) ≈ Matrix(phy.Q)
    end

    @testset "phylo_dep extra guards" begin
        tree = augmented_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        p = tree.n_leaves
        Y = randn(rng, p, 10)
        @test_throws ArgumentError fit_phylo_dep_gllvm(Y, tree; K = 2)
        fit = fit_phylo_dep_gllvm(Y, tree; iterations = 15, g_tol = 1e-3)
        @test fit isa GllvmFit
        @test fit.model.K_phy == p
    end
end