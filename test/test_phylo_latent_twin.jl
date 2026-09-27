# Red-first Julia twins of gllvmTMB P1 (9539352f6) `phylo_latent()` test
# blocks, per the twin map of docs/design/phylo-latent-port-spec.md section
# 4.1. Each testset names the R block it twins. Fixtures are Julia-drawn
# (StableRNGs) ultrametric trees; the paired R receipts live in
# test_phylo_latent_paired_p1.jl.
using Test, LinearAlgebra, SparseArrays, Random, StableRNGs, GLLVModels

# Random coalescent-style ultrametric tree as a Newick string: merge two
# random lineages at increasing times, so every tip sits at the root height.
function _pl_coalescent_newick(n::Integer, rng; labels = ["sp$(i)" for i in 1:n])
    lineages = [(String(labels[i]), 0.0) for i in 1:n]
    t = 0.0
    while length(lineages) > 1
        # Only `rand(rng)` and `rand(rng, range)` draws: their StableRNGs
        # streams are version-stable (`randperm` / `randexp` are not).
        k = length(lineages)
        t += -log(rand(rng)) / k
        i = rand(rng, 1:k)
        j = rand(rng, 1:(k - 1))
        j >= i && (j += 1)
        i, j = minmax(i, j)
        (a, ha), (b, hb) = lineages[i], lineages[j]
        node = ("($(a):$(t - ha),$(b):$(t - hb))", t)
        deleteat!(lineages, j); deleteat!(lineages, i)
        push!(lineages, node)
    end
    return first(lineages)[1] * ";"
end

# Tip correlation matrix of a Newick tree from the native AugmentedPhy path
# (independent of the twin's precision builder).
function _pl_tip_corr(newick)
    phy = augmented_phy(newick; correlation = true)
    C = GLLVModels.sigma_phy_dense(phy)
    o = sortperm(phy.leaf_names)
    return C[o, o], phy.leaf_names[o]
end

# Traits x observations response from the phylo_latent model.
function _pl_simulate(newick, n_traits, reps, loading; sd_eps = 0.3, seed = 1,
        beta = collect(range(-0.5, 0.5; length = n_traits)))
    rng = StableRNG(seed)
    C, tips = _pl_tip_corr(newick)
    p = length(tips)
    L = cholesky(Symmetric(C + 1e-12I)).L
    K = size(loading, 2)
    g = reduce(hcat, [L * randn(rng, p) for _ in 1:K])
    obs = repeat(1:p; inner = reps)
    m = length(obs)
    Y = beta .+ loading * g[obs, :]' .+ sd_eps .* randn(rng, n_traits, m)
    return Y, tips[obs], C, tips
end

const _PL_A14_NEWICK = "(((s1:2,s2:2):1,(s3:1,s4:1):2):1,((s5:1.5,s6:1.5):1,(s7:1,s8:1):1.5):1.5);"

@testset "phylo_latent twin (gllvmTMB P1)" begin
    newick50 = _pl_coalescent_newick(50, StableRNG(7))
    Y50, sp50, C50, tips50 = _pl_simulate(newick50, 4, 1, [0.7 0.0; 0.3 0.5; -0.4 0.2; 0.5 -0.3]; seed = 7)

    @testset "test-phylo-hadfield.R:53 tree route gives a sparse augmented precision" begin
        phy, labels = GLLVModels._phylo_latent_tree_precision(newick50)
        @test phy.n_aug == 2 * 50 - 2
        @test phy.n_aug > 50
        @test nnz(phy.Q) < 0.1 * phy.n_aug^2
        @test nnz(phy.Q) < 600
        @test sort(labels) == tips50
    end

    @testset "test-phylo-hadfield.R:70 vcv route stores a dense tip precision" begin
        phy = GLLVModels._phylo_latent_dense_precision(C50, tips50, tips50)
        @test phy.n_aug == 50
        @test nnz(phy.Q) > 5 * 50
    end

    @testset "test-phylo-hadfield.R:85 tree and vcv routes give the same MLE" begin
        fh = fit_phylo_latent_gllvm(Y50, sp50; d = 2, tree = newick50)
        fd = fit_phylo_latent_gllvm(Y50, sp50; d = 2, vcv = C50, tip_labels = tips50)
        @test fh.converged
        @test fd.converged
        @test isapprox(fh.loglik, fd.loglik; rtol = 1e-4)
        @test isapprox(extract_Sigma(fh; level = :phy, part = :shared).Sigma,
            extract_Sigma(fd; level = :phy, part = :shared).Sigma; rtol = 1e-4)
        @test isapprox(fh.beta, fd.beta; rtol = 1e-4)
    end

    @testset "test-phylo-hadfield.R:114 tip map points inside the augmented precision" begin
        newick30 = _pl_coalescent_newick(30, StableRNG(8))
        phy, _ = GLLVModels._phylo_latent_tree_precision(newick30)
        @test phy.n_aug == 30 + 29 - 1
        @test all(i -> 1 <= i <= phy.n_aug, phy.species_aug_id)
        @test length(unique(phy.species_aug_id)) == 30
    end

    @testset "test-phylo-hadfield.R:134 and test-gllvmTMB-args.R:202 non-phylogeny tree refuses" begin
        err = try
            fit_phylo_latent_gllvm(Y50, sp50; tree = C50)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("tree must be a phylogeny object.", err.msg)
        @test occursin("GJL-GATE-PHYLO-LATENT-TREE", err.msg)
        @test_throws r"tree must be a phylogeny object" fit_phylo_latent_gllvm(Y50, sp50; tree = "not a tree")
    end

    @testset "test-phylo-tree-precision.R:6 precision inverts to the tip correlation" begin
        phy, labels = GLLVModels._phylo_latent_tree_precision(_PL_A14_NEWICK)
        C, tips = _pl_tip_corr(_PL_A14_NEWICK)
        @test labels == tips
        @test phy.scale == 4.0
        Sigma = inv(Matrix(phy.Q))
        @test isapprox(Sigma[phy.species_aug_id, phy.species_aug_id], C; atol = 1e-10)
        # R's log-det rule: n_aug * log(height) - sum(log(edge_length)).
        edges = [1, 1, 2, 2, 2, 1, 1, 1.5, 1, 1.5, 1.5, 1.5, 1, 1]
        @test isapprox(phy.log_det, 14 * log(4) - sum(log, edges); atol = 1e-12)
        recomputed, shipped, diff = precision_logdet_check(phy)
        @test diff <= 1e-8
    end

    @testset "test-phylo-tree-precision.R:42 species-to-node map by label" begin
        phy, labels = GLLVModels._phylo_latent_tree_precision(_PL_A14_NEWICK)
        @test phy.node_labels[phy.species_aug_id] == labels
        @test phy.species_aug_id == collect((phy.n_aug - 7):phy.n_aug)   # tips last
    end

    newick20 = _pl_coalescent_newick(20, StableRNG(11))
    Y20, sp20, C20, tips20 = _pl_simulate(newick20, 4, 3, [0.8 0.0; 0.4 0.5; -0.5 0.3; 0.3 -0.4]; seed = 11)
    fit20 = fit_phylo_latent_gllvm(Y20, sp20; d = 2, vcv = C20, tip_labels = tips20)

    @testset "test-stage35-phylo-rr.R:35 fit and shapes" begin
        @test fit20.converged
        @test fit20.rank == 2
        @test size(fit20.loading) == (4, 2)
        S = extract_Sigma(fit20; level = :phy, part = :shared).Sigma
        @test size(S) == (4, 4)
        @test all(>(0), diag(S))
        @test fit20.loading[1, 2] == 0
        @test fit20.residual_mode === :shared
        @test fit20.species_labels == sp20
    end

    @testset "test-stage35-phylo-rr.R:51 a source is required" begin
        @test_throws r"found in formula but phylo_vcv \(or phylo_tree\) is NULL" fit_phylo_latent_gllvm(Y20, sp20; d = 2)
        @test_throws r"GJL-GATE-PHYLO-LATENT-SOURCE" fit_phylo_latent_gllvm(Y20, sp20; d = 2)
    end

    @testset "test-phylo-vcv-A-aliases.R:72 A = and vcv = are byte-identical (phylo_latent arm)" begin
        fa = fit_phylo_latent_gllvm(Y20, sp20; d = 2, A = C20, tip_labels = tips20)
        @test fa.parameters == fit20.parameters
        @test fa.loglik == fit20.loglik
    end

    @testset "test-phylo-vcv-A-aliases.R:107,119 two sources refuse" begin
        @test_throws r"phylo_latent\(\) got both A and vcv" fit_phylo_latent_gllvm(Y20, sp20; vcv = C20, A = C20, tip_labels = tips20)
        @test_throws r"phylo_latent\(\) got both Ainv and vcv" fit_phylo_latent_gllvm(Y20, sp20; vcv = C20, Ainv = inv(C20), tip_labels = tips20)
        @test_throws r"Supply one of tree, vcv, or A / Ainv" fit_phylo_latent_gllvm(Y20, sp20; tree = newick20, vcv = C20, tip_labels = tips20)
    end

    @testset "test-phylo-latent-unique-fold.R:176 unique = false is loadings-only" begin
        @test fit20.phylo_unique_variance === nothing
        @test all(iszero, extract_Sigma(fit20; level = :phy, part = :unique).s)
        @test fit20.mode === :barelowrank
    end

    @testset "test-phylo-signal-categorical.R:117 no advisory for a Gaussian fit; H2 = 1" begin
        ps = @test_logs extract_phylo_signal(fit20)
        @test ps.H2 == ones(4)
        @test ps.C2_non == zeros(4)
        @test ps.Psi == zeros(4)
        @test ps.V_eta ≈ diag(fit20.loading * fit20.loading')
        @test_throws r"GJL-GATE-PHYLO-LATENT-SIGNAL-CI" extract_phylo_signal(fit20; ci = true)
    end

    @testset "test-latent-rank-guard.R:128 d = T + 1 refuses" begin
        @test_throws r"phylo_latent\(d = 5\) exceeds the number of traits \(4\); the latent rank must satisfy d <= n_traits" fit_phylo_latent_gllvm(Y20, sp20; d = 5, vcv = C20, tip_labels = tips20)
    end

    @testset "test-latent-rank-guard.R:162 d == T fits" begin
        Y3, sp3, C3, tips3 = _pl_simulate(_PL_A14_NEWICK, 3, 4, [0.9 0.0 0.0; 0.5 0.6 0.0; -0.4 0.3 0.5]; seed = 3)
        f3 = fit_phylo_latent_gllvm(Y3, sp3; d = 3, tree = _PL_A14_NEWICK)
        @test f3.converged
        @test size(f3.loading) == (3, 3)
    end

    @testset "test-species-unused-levels-guard.R:47,62 an unused declared level names droplevels()" begin
        keep = sp20 .!= "sp1"
        Yk, spk = Y20[:, keep], sp20[keep]
        rest = findall(!=("sp1"), tips20)
        tree19 = _pl_coalescent_newick(19, StableRNG(12); labels = tips20[rest])
        levels20 = tips20
        @test_throws r"phylo_tree tip labels do not cover all species levels.*droplevels" fit_phylo_latent_gllvm(Yk, spk; tree = tree19, species_levels = levels20)
        err = try
            fit_phylo_latent_gllvm(Yk, spk; vcv = C20[rest, rest], tip_labels = tips20[rest], species_levels = levels20)
        catch e
            e
        end
        @test occursin("phylo_vcv rownames do not cover all species levels.", err.msg)
        @test occursin("droplevels()", err.msg)
        @test occursin("GJL-GATE-PHYLO-LATENT-COVERAGE", err.msg)
    end

    @testset "test-species-unused-levels-guard.R:116 an observed species missing from the tree is a genuine mismatch" begin
        keep = sp20 .!= "sp1"
        rest = findall(t -> !(t in ("sp1", "sp2")), tips20)
        err = try
            fit_phylo_latent_gllvm(Y20[:, keep], sp20[keep]; vcv = C20[rest, rest],
                tip_labels = tips20[rest], species_levels = tips20)
        catch e
            e
        end
        @test occursin("droplevels", err.msg)
        @test occursin("genuine mismatch", err.msg)
        @test occursin("sp2", err.msg)
    end

    @testset "test-species-unused-levels-guard.R:143 a clean fit is unaffected" begin
        f = fit_phylo_latent_gllvm(Y20, sp20; d = 1, tree = newick20)
        @test f.converged
        @test sort(f.tip_labels) == tips20
        @test f.species_labels == sp20
    end

    @testset "test-gllvmTMB-args.R:175 vcv without labels refuses" begin
        @test_throws r"phylo_vcv must have rownames matching levels of species" fit_phylo_latent_gllvm(Y20, sp20; vcv = C20)
    end

    @testset "test-gllvmTMB-args.R:187 vcv rows that do not cover the species refuse" begin
        rest = findall(!=("sp1"), tips20)
        @test_throws r"phylo_vcv rownames do not cover all species levels" fit_phylo_latent_gllvm(Y20, sp20; vcv = C20[rest, rest], tip_labels = tips20[rest])
    end

    @testset "Julia scope fences: rho and Ainv" begin
        @test_throws r"GJL-GATE-PHYLO-LATENT-RHO" fit_phylo_latent_gllvm(Y20, sp20; vcv = C20, tip_labels = tips20, rho = 0.5)
        @test_throws r"GJL-GATE-PHYLO-LATENT-AINV" fit_phylo_latent_gllvm(Y20, sp20; Ainv = inv(C20), tip_labels = tips20)
    end

    @testset "Julia-only: non-ultrametric tree refuses with the R sentence" begin
        @test_throws r"tree must be ultrametric.*GJL-GATE-PHYLO-NONULTRAMETRIC" fit_phylo_latent_gllvm(Y20[:, 1:4], ["a", "a", "b", "c"]; tree = "((a:1,b:2):1,c:2);")
    end

    @testset "Julia-only: polytomies are admitted as R admits them" begin
        poly = "((a:1,b:1,c:1):1,(d:1.5,e:1.5):0.5);"
        phy, labels = GLLVModels._phylo_latent_tree_precision(poly)
        @test phy.n_aug == 5 + 3 - 1          # n_tip + Nnode - 1
        @test labels == ["a", "b", "c", "d", "e"]
        Sigma = inv(Matrix(phy.Q))[phy.species_aug_id, phy.species_aug_id]
        # Path-sum covariance / height (height 2): shared paths.
        Cpoly = [2 1 1 0 0; 1 2 1 0 0; 1 1 2 0 0; 0 0 0 2 0.5; 0 0 0 0.5 2] ./ 2
        @test isapprox(Sigma, Cpoly; atol = 1e-12)
        @test precision_logdet_check(phy)[3] <= 1e-8
        # Same route through an AugmentedPhy of a bifurcating tree.
        phyA, labelsA = GLLVModels._phylo_latent_tree_precision(augmented_phy(_PL_A14_NEWICK))
        phyN, _ = GLLVModels._phylo_latent_tree_precision(_PL_A14_NEWICK)
        @test labelsA == ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
        @test isapprox(phyA.log_det, phyN.log_det; atol = 1e-12)
        @test isapprox(inv(Matrix(phyA.Q))[phyA.species_aug_id, phyA.species_aug_id],
            inv(Matrix(phyN.Q))[phyN.species_aug_id, phyN.species_aug_id]; atol = 1e-12)
    end

    @testset "Julia-only: dense route replicates R's 1e-8 ridge" begin
        phy = GLLVModels._phylo_latent_dense_precision(C20, tips20, tips20)
        @test isapprox(phy.log_det, -logdet(C20 + 1e-8I); atol = 1e-10)
        @test isapprox(Matrix(phy.Q), inv(C20 + 1e-8I); rtol = 1e-10)
        @test issymmetric(phy.Q)
    end

    @testset "Julia-only: d = 1 recovers a planted Sigma_phy (StableRNGs)" begin
        newick = _pl_coalescent_newick(60, StableRNG(21))
        loading = reshape([0.9, 0.6, -0.5], 3, 1)
        Y, sp, _, _ = _pl_simulate(newick, 3, 5, loading; sd_eps = 0.3, seed = 21)
        f = fit_phylo_latent_gllvm(Y, sp; d = 1, tree = newick)
        @test f.converged
        S = extract_Sigma(f; level = :phy, part = :shared).Sigma
        @test isapprox(S, loading * loading'; atol = 0.35)
        @test isapprox(sqrt(f.residual_variance[1]), 0.3; rtol = 0.15)
    end
end
