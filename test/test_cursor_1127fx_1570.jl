using Test
using GLLVModels
using Distributions
using Random

# #1570: fit_gllvm(Poisson(); X=...) used to forward X into the Laplace fitter and raise a
# confusing internal MethodError; refuse site covariates up front with a clear ArgumentError.

@testset "fit_gllvm Poisson site covariates refused (#1570)" begin
    Random.seed!(1127)
    Y = abs.(round.(Int, rand(5, 12) .+ 1))
    X = randn(2, 12, 5)
    err = nothing
    try
        fit_gllvm(Y; family = Poisson(), K = 1, X = X, iterations = 20)
    catch e
        err = e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("fit_gllvm", msg)
    @test occursin("X", msg)
    @test occursin("fit_gllvm_cov", msg)
end