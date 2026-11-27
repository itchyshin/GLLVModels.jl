# refs #1501: 1×1 fit_gllvm(Normal()) must not AssertionError in LineSearches.
using Test, GLLVModels, Distributions

@testset "issue 1501: single-cell fit_gllvm(Normal()) guard" begin
    Y = reshape([1.0], 1, 1)
    fgg = fit_gaussian_gllvm(Y; K = 1)
    @test isfinite(fgg.logLik) && fgg.converged
    err = try
        fit_gllvm(Y; family = Normal(), K = 1)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("fit_gaussian_gllvm", sprint(showerror, err))
end