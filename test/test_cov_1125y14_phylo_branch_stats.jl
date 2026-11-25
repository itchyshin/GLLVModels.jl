# Lane cursor-cloud-1125y14 — src/phylo_branch_re.jl
using GLLVModels, Test, Random, Statistics

const G = GLLVModels

@testset "cov 1125y14 phylo branch RE stats" begin
    phy = G.edge_phy("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.1);")
    rng = MersenneTwister(11251405)

    @testset "path_membership and clade helpers" begin
        P = G.path_membership(phy)
        @test size(P) == (phy.n_leaves, phy.n_edges)
        @test all(P .∈ Ref((0.0, 1.0)))
        root = G.find_clade_root(phy; target_leaves = 2)
        @test 1 <= root <= phy.n_nodes
        edges = G.clade_edges(phy, phy.root_index)
        @test all(1 .<= edges .<= phy.n_edges)
        order = G._postorder(phy)
        @test length(order) == phy.n_nodes
        @test order[end] == phy.root_index
    end

    @testset "welch_t rank_sum_z diagnostics" begin
        a = randn(rng, 8)
        b = randn(rng, 10) .+ 0.5
        t, dmean, d = G.welch_t(a, b)
        @test isfinite(t)
        @test dmean ≈ mean(a) - mean(b)
        z = G.rank_sum_z(a, b)
        @test isfinite(z)
        ek = G.excess_kurtosis(vcat(a, b))
        @test isfinite(ek)
        qq = G.qq_max_dev(vcat(a, b))
        @test qq >= 0
        r = G._tiedrank_re([1.0, 1.0, 2.0, 3.0])
        @test r ≈ [1.5, 1.5, 3.0, 4.0]
    end

    @testset "_branch_re_check_y" begin
        @test_throws ArgumentError G._branch_re_check_y([1.0, 2.0], 3; caller = "t")
        @test_throws ArgumentError G._branch_re_check_y([NaN], 1; caller = "t")
        G._branch_re_check_y([1.0, 2.0], 2; caller = "t")
    end

    @testset "tiny fit_branch_re" begin
        y = randn(rng, phy.n_leaves)
        fit = fit_branch_re(phy, y; max_iter = 80)
        @test isfinite(fit.negll)
        @test fit.σ² >= 0
        cache = G.branch_re_cache(phy)
        nll, μ̂ = G.branch_re_profile_negll(cache, y, fit.σ², fit.σ²_eps)
        @test isfinite(nll)
        ẑ, std_incr, rate_e = G.branch_blups(cache, y, fit.σ², fit.σ²_eps, μ̂)
        @test length(ẑ) == phy.n_edges
        @test length(rate_e) == phy.n_edges
        @test all(isfinite, std_incr)
    end

    @testset "_nelder_mead_1d" begin
        f = x -> (x - 0.3)^2
        xmin, fbest, iters, conv = G._nelder_mead_1d(f, 1.0)
        @test abs(xmin - 0.3) < 0.05
        @test fbest < 1e-4
        @test iters > 0
    end
end