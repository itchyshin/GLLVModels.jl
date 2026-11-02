using GLLVModels, Test, Random, LinearAlgebra, SparseArrays, Statistics

# Part of the relaxed-clock coverage gap: `estep_edge_moments` and
# `clade_detection` are exported but were not exercised by
# test_relaxed_clock.jl (which covers `fit_relaxed_clock` only).

# Dense brute-force Gaussian posterior for the conditioned (root-dropped)
# node system, built independently of `estep_edge_moments`.
function _brute_estep(phy, σ²_e, σ²_eps, y)
    p, n_rep = size(y)
    keep = [i for i in 1:phy.n_nodes if i != phy.root_index]
    B = Matrix(phy.B)
    w = 1.0 ./ (σ²_e .* phy.branch_lengths)
    Qc = (B * Diagonal(w) * B')[keep, keep]
    nc = length(keep)
    S = zeros(p, nc)
    for t in 1:p
        S[t, findfirst(==(phy.leaf_indices[t]), keep)] = 1.0
    end
    P = Qc + S' * S / σ²_eps
    Pinv = inv(Symmetric(P))
    Mu = P \ (S' * y / σ²_eps)
    μ_nodes = zeros(phy.n_nodes, n_rep)
    μ_nodes[keep, :] = Mu
    Δ = B' * μ_nodes
    Bc = B[keep, :]
    s_e = vec(sum(Δ .^ 2; dims = 2)) .+ n_rep .* diag(Bc' * Pinv * Bc)
    resid = y - S * Mu
    recon = sum(abs2, resid) + n_rep * tr(S * Pinv * S')
    return s_e, recon, μ_nodes
end

@testset "estep_edge_moments vs dense brute-force posterior" begin
    phy = edge_phy("(((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.1):0.2,((E:0.3,F:0.15):0.25,(G:0.2,H:0.35):0.4):0.1);")
    cm = GLLVModels._cond_map(phy)
    rng = MersenneTwister(20261019)
    p = phy.n_leaves
    σ²_e = collect(range(0.4, 2.2; length = phy.n_edges))
    σ²_eps = 0.35
    y = randn(rng, p, 4)

    s_e, recon, μ_nodes = estep_edge_moments(phy, σ²_e, σ²_eps, y, cm)
    s_ref, recon_ref, μ_ref = _brute_estep(phy, σ²_e, σ²_eps, y)

    @test size(μ_nodes) == (phy.n_nodes, size(y, 2))
    @test μ_nodes ≈ μ_ref rtol = 1e-8 atol = 1e-10
    @test all(μ_nodes[phy.root_index, :] .== 0)
    @test length(s_e) == phy.n_edges
    @test s_e ≈ s_ref rtol = 1e-8
    @test recon ≈ recon_ref rtol = 1e-8
    @test all(s_e .>= 0)
    @test recon >= 0

    @testset "duplicating replicate columns doubles the mean part" begin
        y2 = hcat(y, y)
        s2, recon2, μ2 = estep_edge_moments(phy, σ²_e, σ²_eps, y2, cm)
        # Posterior means are per-replicate, so duplicated columns repeat.
        @test μ2[:, 1:4] ≈ μ_nodes
        @test μ2[:, 5:8] ≈ μ_nodes
        # s_e = Σ_r δ̄² + n_rep var_e: both pieces scale linearly in n_rep.
        @test s2 ≈ 2 .* s_e rtol = 1e-8
        @test recon2 ≈ 2 * recon rtol = 1e-8
        # Isolate the mean part: s_e - (posterior-variance part) doubles.
        Δ = phy.B' * μ_nodes
        mean_part = vec(sum(Δ .^ 2; dims = 2))
        var_part = s_e .- mean_part
        @test all(var_part .>= -1e-10)
        @test (s2 .- 8 .* (var_part ./ 4)) ≈ 2 .* mean_part rtol = 1e-8
    end
end

@testset "clade_detection" begin
    est = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25]
    fast = [3, 4]
    slow = [1, 5, 6]
    r = clade_detection(est, fast, slow)
    @test r.mean_fast == 6.0
    @test r.mean_slow ≈ (1.0 + 0.5 + 0.25) / 3
    @test r.ratio ≈ 6.0 / ((1.0 + 0.5 + 0.25) / 3)
    lf = log.(est[fast]); ls = log.(est[slow])
    @test r.log_gap ≈ mean(lf) - mean(ls)
    se = sqrt(var(lf) / 2 + var(ls) / 3)
    @test r.t ≈ (mean(lf) - mean(ls)) / se
    @test r.t > 0

    # Swapping the groups flips the sign of t and log_gap.
    rs = clade_detection(est, slow, fast)
    @test rs.t ≈ -r.t
    @test rs.log_gap ≈ -r.log_gap
    @test rs.ratio ≈ 1 / r.ratio

    # Zero spread in both groups: se == 0 takes the Inf branch.
    r0 = clade_detection([2.0, 2.0, 0.5, 0.5], [1, 2], [3, 4])
    @test r0.t == Inf
    @test r0.ratio == 4.0
    @test r0.log_gap ≈ log(4.0)

    # A single-element group: var of one element is NaN, so se is NaN and
    # `se > 0` is false. Pinned as currently coded: t falls into the Inf
    # branch (not NaN), whatever the sign of the gap.
    r1 = clade_detection([1.0, 3.0, 2.0, 2.5], [1], [2, 3, 4])
    @test r1.mean_fast == 1.0
    @test !isnan(r1.t)
    @test r1.t == Inf
    @test r1.log_gap < 0
end
