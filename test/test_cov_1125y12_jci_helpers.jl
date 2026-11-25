# Lane cursor-cloud-1125y12 — joint_covariance_identification.jl helpers (extra branches)
using GLLVModels, Test, LinearAlgebra, SparseArrays

const _JCI = GLLVModels

function _phy_Q(Q::AbstractMatrix)
    sparseQ = sparse(Float64.(Q))
    n = size(sparseQ, 1)
    return PrecisionPhy(findnz(sparseQ)..., n, n, ["t$i" for i in 1:n],
                        logdet(cholesky(Symmetric(Matrix(sparseQ)))), 1.0, collect(1:n))
end

@testset "cov 1125y12 joint CI helper paths" begin
    @testset "_jci_kind" begin
        a = (kind = :phylo,)
        b = (kind = :ordinary,)
        c = (kind = :residual,)
        @test _JCI._jci_kind(a, b) === :phylo_vs_ordinary
        @test _JCI._jci_kind(b, c) === :ordinary_vs_residual
        @test _JCI._jci_kind(b, b) === :duplicate_ordinary
        @test _JCI._jci_kind(a, c) === :phylo_vs_residual
        @test _JCI._jci_kind(c, c) === :within_component
    end

    @testset "_jci_trace and frobenius" begin
        phy = _phy_Q([2.0 -0.3; -0.3 2.0])
        mapped = [1, 2, 1]
        tr = _JCI._jci_trace_phylo_kernel(cholesky(Symmetric(phy.Q)), mapped, phy.n_aug)
        fro = _JCI._jci_phylo_kernel_frobenius2(cholesky(Symmetric(phy.Q)), mapped, phy.n_aug)
        @test tr > 0 && fro > 0
        @test fro >= tr^2 / length(mapped)^2 - 1e-10
    end

    @testset "_jci_validate_inputs throws" begin
        phy = _phy_Q(Matrix(I, 3, 3))
        L = reshape([0.4], 1, 1)
        @test_throws ArgumentError _JCI._jci_validate_inputs(
            phy, Int[], GroupingTerm[], SparseMatrixCSC{Float64, Int}[], L, nothing)
        @test_throws DimensionMismatch _JCI._jci_validate_inputs(
            phy, [1, 2, 3], GroupingTerm[], SparseMatrixCSC{Float64, Int}[], L,
            [0.1, 0.2])
        terms = [GroupingTerm(:unit; mode = :indep, common = false),
                 GroupingTerm(:unit; mode = :indep, common = false)]
        @test_throws ArgumentError _JCI._jci_validate_inputs(
            phy, [1, 2, 3], terms,
            [spdiagm(0 => ones(3)), spdiagm(0 => ones(3))],
            reshape([0.1, 0.2], 2, 1), nothing)
        bad_term = [GroupingTerm(:cluster; mode = :dep, common = false)]
        @test_throws ArgumentError _JCI._jci_validate_inputs(
            phy, [1, 2], bad_term, [spdiagm(0 => ones(2))], reshape([0.5], 1, 1),
            nothing)
        @test_throws ArgumentError _JCI._jci_validate_inputs(
            phy, [1.5, 2.0], GroupingTerm[], SparseMatrixCSC{Float64, Int}[], L,
            nothing)
    end

    @testset "_jci_invalid shape" begin
        inv = _JCI._jci_invalid(["a", "b"]; message = "test msg")
        @test inv.reason === :invalid && !inv.valid
        @test inv.message == "test msg"
        @test inv.rank == 0
    end
end