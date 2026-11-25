# Lane cursor-cloud-1125y15 — src/takahashi_selinv.jl (selected inverse / diagonal)
using GLLVModels, Test, LinearAlgebra, SparseArrays, Random

const G = GLLVModels

function _spd_tree(n::Int)
    # tridiagonal + small coupling keeps the pattern tree-like for Takahashi.
    Isp = spdiagm(0 => fill(3.0, n))
    for i in 1:(n - 1)
        Isp[i, i + 1] = -1.0
        Isp[i + 1, i] = -1.0
    end
    return Isp
end

@testset "cov 1125y15 takahashi selected inverse" begin
    @testset "_csc_rowidx binary search" begin
        A = sparse([1, 2, 3], [1, 1, 2], [1.0, 2.0, 3.0])
        cp, rv = A.colptr, A.rowval
        @test G._csc_rowidx(cp, rv, 1, 1) == 1
        @test G._csc_rowidx(cp, rv, 1, 2) == 2
        @test G._csc_rowidx(cp, rv, 2, 3) == 3
        @test G._csc_rowidx(cp, rv, 1, 9) == -1
        @test G._csc_rowidx(cp, rv, 2, 1) == -1
    end

    @testset "takahashi_diag matches dense inverse diagonal" begin
        for n in 3:7
            Q = _spd_tree(n)
            ch = cholesky(Symmetric(Q))
            d_t = G.takahashi_diag(ch)
            d_d = diag(inv(Matrix(Q)))
            @test d_t ≈ d_d rtol = 1e-10 atol = 1e-10
            @test all(isfinite, d_t)
            @test length(d_t) == n
        end
    end

    @testset "takahashi_selinv pattern is subset of dense" begin
        Q = _spd_tree(8)
        ch = cholesky(Symmetric(Q))
        Z = G.takahashi_selinv(ch)
        Qinv = inv(Matrix(Q))
        I, J, V = findnz(Z)
        for k in eachindex(V)
            @test V[k] ≈ Qinv[I[k], J[k]] rtol = 1e-9 atol = 1e-9
        end
        @test issymmetric(Matrix(Z))
    end

    @testset "diag-only path matches full takahashi diagonal" begin
        Q = sprand(6, 6, 0.35)
        Q = Q + Q' + 4I
        ch = cholesky(Symmetric(Q))
        @test G.takahashi_diag(ch) ≈ diag(G.takahashi_selinv(ch)) rtol = 1e-12
    end

    @testset "permutation recovery" begin
        Random.seed!(1125)
        Q = sprand(7, 7, 0.4)
        Q = Q + Q' + 5I
        ch = cholesky(Symmetric(Q))
        d1 = G.takahashi_diag(ch)
        # Factor reordering is internal; diagonal should stay invariant on Q.
        @test d1 ≈ diag(inv(Matrix(Q))) rtol = 1e-9
    end
end