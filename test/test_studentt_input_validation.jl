using Test, GLLVModels

# Exercise public admission without allowing a response read or a model fit.
struct StudentInputRead <: Exception end
struct StudentInputMatrix <: AbstractMatrix{Float64} end
Base.size(::StudentInputMatrix) = (3, 5)
Base.getindex(::StudentInputMatrix, ::Int, ::Int) = throw(StudentInputRead())

@testset "Student-t fixed nu input boundary" begin
    Y = StudentInputMatrix()
    for nu in (Inf, -Inf, NaN, 0.0, -1.0,
               [4.0, Inf, 4.0], [4.0, -Inf, 4.0], [4.0, NaN, 4.0],
               [4.0, 0.0, 4.0], [4.0, -1.0, 4.0])
        @test_throws ArgumentError fit_studentt_gllvm(Y; K=1, nu=nu)
    end
    @test_throws ArgumentError fit_studentt_gllvm(Y; K=1, nu=[4.0, 4.0])
    for nu in (Inf, -Inf, NaN, 0.0, -1.0)
        @test_throws ArgumentError fit_gllvm(Y; K=1, family=StudentTFamily(nu))
    end
    # Positive controls reach data access. Finite df <= 1 is a native extension,
    # although the frozen R constructor admits only finite df > 1.
    for nu in (0.5, 1.0, 2.0, 4.0, 1e6, [0.5, 1.0, 4.0], nothing)
        @test_throws StudentInputRead fit_studentt_gllvm(Y; K=1, nu=nu)
    end
    for nu in (0.5, 1.0, 4.0, nothing)
        @test_throws StudentInputRead fit_gllvm(Y; K=1, family=StudentTFamily(nu))
    end
end
