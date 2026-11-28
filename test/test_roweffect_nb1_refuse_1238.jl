using Test, GLLVModels, Random, Distributions

@testset "row routes refuse NB1() before fitting" begin
    rng = MersenneTwister(1)
    Y = rand(rng, Poisson(4.0), 6, 40)
    for (lab, f) in (
        "fit_roweffect_gllvm" => () -> fit_roweffect_gllvm(Y; family = GLLVModels.NB1(), K = 1),
        "fit_row_random_gllvm" => () -> fit_row_random_gllvm(Y; family = GLLVModels.NB1(), K = 1),
    )
        err = try; f(); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin(lab, err.msg)
        @test occursin("NB1", err.msg) || occursin("NegativeBinomial", err.msg)
    end
end