# Unit tests for the small statistics / tree helpers exported next to the
# branch random-effects fitters (src/phylo_branch_re.jl). These helpers are
# otherwise exercised only indirectly; the checks below avoid the optimiser
# wherever possible. Convergence flags of `fit_branch_re` are out of scope.

using GLLVModels, Test, Random, LinearAlgebra, SparseArrays, Statistics

@testset "branch-RE helpers" begin

    @testset "welch_t" begin
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 6.0, 8.0, 10.0]
        na, nb = length(a), length(b)
        ma, mb = sum(a) / na, sum(b) / nb
        va = sum((a .- ma) .^ 2) / (na - 1)
        vb = sum((b .- mb) .^ 2) / (nb - 1)
        t_ref = (ma - mb) / sqrt(va / na + vb / nb)
        sp = sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        t, dm, d = welch_t(a, b)
        @test t ≈ t_ref rtol = 1e-12
        @test dm ≈ ma - mb
        @test d ≈ (ma - mb) / sp rtol = 1e-12
        @test t < 0
        t2, dm2, d2 = welch_t(b, a)
        @test t2 ≈ -t
        @test dm2 ≈ -dm
        @test d2 ≈ -d
    end

    @testset "rank_sum_z" begin
        # ties: pooled sorted 1,2,2,2,3,3,4,5; midranks 1,3,3,3,5.5,5.5,7,8
        a = [1.0, 2.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0, 5.0]
        Ra = 1 + 3 + 3 + 5.5
        U = Ra - 4 * 5 / 2
        z_ref = (U - 8) / sqrt(4 * 4 * 9 / 12)
        @test rank_sum_z(a, b) ≈ z_ref rtol = 1e-12
        @test rank_sum_z(a, b) < 0
        @test rank_sum_z(b, a) ≈ -z_ref rtol = 1e-12
        # no overlap: Ra = 1+2+3 = 6, U = 0, mean 6, sd sqrt(3*4*8/12)
        lo, hi = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]
        @test rank_sum_z(lo, hi) ≈ -6 / sqrt(8) rtol = 1e-12
        @test rank_sum_z(hi, lo) ≈ 6 / sqrt(8) rtol = 1e-12
        # all-equal pooled sample: U equals its mean exactly
        @test rank_sum_z([1.0, 1.0], [1.0, 1.0]) == 0.0
    end

    @testset "excess_kurtosis" begin
        rng = MersenneTwister(20240607)
        n = 200_000
        k = excess_kurtosis(randn(rng, n))
        @test abs(k) < 4 * sqrt(24 / n)        # SE of sample kurtosis ≈ √(24/n)
        @test excess_kurtosis([-1.0, 1.0]) ≈ -2 atol = 1e-12
        @test excess_kurtosis([0.0, 0.0, 5.0, 5.0, 5.0, 0.0]) ≈ -2 atol = 1e-12
    end

    @testset "qq_max_dev" begin
        rng = MersenneTwister(11)
        n = 2000
        gauss = randn(rng, n)
        heavy = randn(rng, n) ./ sqrt.(rand(rng, n))      # scale mixture, fat tails
        dg, dh = qq_max_dev(gauss), qq_max_dev(heavy)
        @test dg < 0.5
        @test dh > 2 * dg
        # location / scale invariance
        @test qq_max_dev(3 .* gauss .+ 7) ≈ dg rtol = 1e-8
    end

    # 6 leaves, distinct branch lengths so edges are identified by length:
    #   root -> n2 (0.31) -> [n1 (0.21) -> [A .11, B .12], C .13]
    #        -> n4 (0.32) -> [n3 (0.22) -> [D .14, E .15], F .16]
    nwk = "(((A:0.11,B:0.12):0.21,C:0.13):0.31,((D:0.14,E:0.15):0.22,F:0.16):0.32);"
    phy = edge_phy(nwk)
    node_with_edge_len(ℓ) =
        only(u for u in 1:phy.n_nodes if phy.node_edge[u] != 0 &&
             phy.branch_lengths[phy.node_edge[u]] ≈ ℓ)
    lens(es) = sort(round.(phy.branch_lengths[es]; digits = 10))

    @testset "clade_edges" begin
        @test lens(clade_edges(phy, node_with_edge_len(0.31))) ==
              [0.11, 0.12, 0.13, 0.21, 0.31]
        @test lens(clade_edges(phy, node_with_edge_len(0.21))) == [0.11, 0.12, 0.21]
        @test lens(clade_edges(phy, node_with_edge_len(0.13))) == [0.13]
        all_edges = clade_edges(phy, phy.root_index)
        @test sort(all_edges) == 1:phy.n_edges
        @test allunique(clade_edges(phy, node_with_edge_len(0.32)))
    end

    @testset "find_clade_root" begin
        nleaves(u) = isempty(phy.node_children[u]) ? 1 :
                     sum(nleaves, phy.node_children[u])
        for target in (2, 3)
            u = find_clade_root(phy; target_leaves = target)
            @test u != phy.root_index
            @test !isempty(phy.node_children[u])
            @test nleaves(u) == target
        end
        # No clade of size 5 exists. The helper's documented contract is
        # "nearest leaf count", so it returns an internal non-root node.
        u = find_clade_root(phy; target_leaves = 5)
        @test u != phy.root_index
        @test !isempty(phy.node_children[u])
        @test nleaves(u) == 3
    end

    @testset "fit_branch_re_dense vs sparse" begin
        phy12 = edge_phy("((((A:0.1,B:0.2):0.1,(C:0.15,D:0.25):0.2):0.1," *
                         "((E:0.3,F:0.1):0.1,(G:0.2,H:0.2):0.15):0.1):0.05," *
                         "((I:0.2,J:0.3):0.1,(K:0.1,L:0.4):0.2):0.1);")
        @test phy12.n_leaves == 12
        rng = MersenneTwister(5)
        y = 1.0 .+ 0.8 .* randn(rng, 12)
        cache = branch_re_cache(phy12)
        @test cache isa BranchRECache
        @test cache.p == 12 && cache.E == phy12.n_edges
        fd = fit_branch_re_dense(phy12, y)
        # Fixed parameters: the sparse likelihood at the dense fit's variances
        # must reproduce the dense fit's own negative log-likelihood.
        nll_sparse, μ_sparse = branch_re_profile_negll(cache, y, fd.σ², fd.σ²_eps)
        @test nll_sparse ≈ fd.negll atol = 1e-8
        @test μ_sparse ≈ fd.μ atol = 1e-8
        # Both fitters optimise the same objective: optimised likelihoods agree
        # (loose tolerance; the optimum itself is not compared).
        fs = fit_branch_re(phy12, y)
        @test fs.negll ≈ fd.negll atol = 1e-4
    end
end
