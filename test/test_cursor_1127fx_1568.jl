using Test
using GLLVModels, Random

@testset "issue #1568: all-zero trait row rejected at fit_gaussian_gllvm" begin
    Random.seed!(1127)
    Y = randn(5, 12)
    Y[3, :] .= 0.0
    err = @test_throws ArgumentError fit_gaussian_gllvm(Y; K = 1)
    @test occursin("trait 3", sprint(showerror, err.value))
    @test occursin("no variation", sprint(showerror, err.value))
end