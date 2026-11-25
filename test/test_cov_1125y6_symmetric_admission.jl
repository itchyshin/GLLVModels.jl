# Lane cursor-cloud-1125y6: symmetric_admission.jl and precision gate symmetry errors.
using Test, GLLVModels, LinearAlgebra, SparseArrays

const G = GLLVModels

@testset "1125y6 symmetric admission" begin
    @testset "_symmetric_admit dense" begin
        C = [1.0 0.2; 0.2 1.0]
        out = G._symmetric_admit(C, "Σ")
        @test issymmetric(out)
        @test out[1, 2] ≈ 0.2 rtol = 1e-12
        S = [1.0 0.0; 0.0 1.0]
        @test G._symmetric_admit(S, "I") === S
        @test_throws ArgumentError G._symmetric_admit([1 Inf; 2 1], "bad")
        asym = [1.0 2.0; 0.0 1.0]
        @test_throws ArgumentError G._symmetric_admit(asym, "Q")
        tiny = [1.0 1.0 + 1e-12; 1.0 1.0]
        @test G._symmetric_admit(tiny, "ε") ≈ [1.0 1.0; 1.0 1.0] atol = 1e-12
    end

    @testset "_symmetric_sparse_admit" begin
        Q = sparse([1.0 0.0; 0.0 2.0])
        out = G._symmetric_sparse_admit(Q, "Q")
        @test out === Q
        Q2 = sparse([1.0 0.5; 0.0 1.0])
        @test_throws ArgumentError G._symmetric_sparse_admit(Q2, "Q")
    end

    @testset "precision gate surfaces symmetry refusal" begin
        Q = sparse([3.0 -1.0 0.0; -1.0 3.0 -1.0; 0.5 -1.0 2.0])
        I, J, V = findnz(Q)
        raw = PrecisionPhy(I, J, V, 3, 2, ["a", "t1", "t2"],
            logdet(cholesky(Symmetric(Matrix(Q)))), 1.0, [2, 3])
        err = try
            G._validate_precision_fit_input(raw)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("GJL-GATE-PRECISION-FIT-SYMMETRY", err.msg)
    end
end