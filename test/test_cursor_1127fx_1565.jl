using GLLVModels, Test, Random, Distributions

@testset "RowEffectFit predict site count (refs #1565)" begin
    Random.seed!(99)
    Y = [rand(Poisson(3)) for _ in 1:5, _ in 1:20]
    fit = fit_roweffect_gllvm(Y; family = Poisson(), K = 1, iterations = 80)

    @test size(predict(fit, Y)) == (5, 20)

    err = try
        predict(fit, Y[:, 1:10])
        nothing
    catch e
        e
    end
    @test err isa DimensionMismatch
    @test occursin("length(ρ)", err.msg)
    @test occursin("10", err.msg)
    @test occursin("20", err.msg)
end