using GLLVModels, Test, Random, LinearAlgebra, SparseArrays, ForwardDiff

# Coverage for the exported `solve_Q` (src/edge_incidence.jl) and the
# per-branch helpers `edge_W_diag` / `Q_perbranch` (src/relaxed_clock.jl).

const _SQ_NEWICK = "(((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.1):0.2,(E:0.25,F:0.35):0.15);"

# Dense root-conditioned precision built from phy.B and branch_lengths only.
function _sq_Qcond(phy, s2)
    W = Diagonal(1 ./ (s2 .* phy.branch_lengths))
    Q = Matrix(phy.B) * W * Matrix(phy.B)'
    keep = [i for i in 1:phy.n_nodes if i != phy.root_index]
    return Q[keep, keep], keep
end

@testset "edge-incidence solve_Q / edge_W_diag / Q_perbranch" begin
    phy = edge_phy(_SQ_NEWICK)
    n_cond = phy.n_nodes - 1
    rng = MersenneTwister(11)
    b = randn(rng, n_cond)
    s2 = 0.7

    @testset "Q_cond * solve_Q(b) ≈ b" begin
        Qc, _ = _sq_Qcond(phy, s2)
        x = solve_Q(phy, s2, b)
        @test length(x) == n_cond
        @test Qc * x ≈ b atol = 1e-10
    end

    @testset "leaf block of Q_cond⁻¹ equals sigma_phy_dense_edge" begin
        _, keep = _sq_Qcond(phy, s2)
        # position of each leaf among the kept (non-root) nodes
        pos = [findfirst(==(l), keep) for l in phy.leaf_indices]
        Σ = sigma_phy_dense_edge(phy, s2)
        p = phy.n_leaves
        Σsolve = zeros(p, p)
        for j in 1:p
            e = zeros(n_cond); e[pos[j]] = 1.0
            Σsolve[:, j] = solve_Q(phy, s2, e)[pos]
        end
        @test Σsolve ≈ Σ atol = 1e-10
    end

    @testset "scaling: Q ∝ 1/σ²" begin
        x1 = solve_Q(phy, s2, b)
        for c in (0.5, 3.0, 10.0)
            @test solve_Q(phy, c * s2, b) ≈ c .* x1 rtol = 1e-10
        end
    end

    @testset "length mismatch throws ArgumentError" begin
        for bad in (n_cond - 1, n_cond + 1, 0)
            err = try
                solve_Q(phy, s2, zeros(bad)); nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("must equal n_nodes", sprint(showerror, err))
        end
    end

    @testset "ForwardDiff duals propagate" begin
        x = solve_Q(phy, s2, b)
        # σ² is a Dual: x(s2) = s2 · x(1), so dx/ds2 = x / s2
        dx = ForwardDiff.derivative(s -> solve_Q(phy, s, b), s2)
        @test dx ≈ x ./ s2 rtol = 1e-8
        # b is Dual: linear in b, Jacobian = Q_cond⁻¹
        Qc, _ = _sq_Qcond(phy, s2)
        J = ForwardDiff.jacobian(bb -> solve_Q(phy, s2, bb), b)
        @test J ≈ inv(Qc) atol = 1e-9
        # both at once
        v = ForwardDiff.derivative(s -> sum(solve_Q(phy, s, b)), s2)
        @test v ≈ sum(x) / s2 rtol = 1e-8
    end

    @testset "edge_W_diag" begin
        v = 0.3 .+ rand(rng, phy.n_edges)
        @test edge_W_diag(phy, v) ≈ 1 ./ (v .* phy.branch_lengths)
        @test_throws ArgumentError edge_W_diag(phy, ones(phy.n_edges + 1))
    end

    @testset "Q_perbranch" begin
        v = 0.3 .+ rand(rng, phy.n_edges)
        Q = Q_perbranch(phy, v)
        @test Q isa SparseMatrixCSC
        @test size(Q) == (phy.n_nodes, phy.n_nodes)
        @test Matrix(Q) ≈ Matrix(Q)' atol = 1e-12
        # rows sum to zero (B' * 1 = 0)
        @test all(abs.(Matrix(Q) * ones(phy.n_nodes)) .< 1e-10)
        # constant v: same matrix solve_Q inverts, restricted to non-root nodes
        Qc, keep = _sq_Qcond(phy, s2)
        Qconst = Q_perbranch(phy, fill(s2, phy.n_edges))
        @test Matrix(Qconst)[keep, keep] ≈ Qc atol = 1e-10
        @test Matrix(Qconst)[keep, keep] * solve_Q(phy, s2, b) ≈ b atol = 1e-10
        # scaling: Q_perbranch(c*v) = Q_perbranch(v)/c
        @test Matrix(Q_perbranch(phy, 2 .* v)) ≈ Matrix(Q) ./ 2 atol = 1e-12
    end
end
