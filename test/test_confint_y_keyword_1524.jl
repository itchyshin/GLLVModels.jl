# refs #1524: confint(f; y = Y) on Laplace family fits must not MethodError on unsupported keyword y.
using GLLVModels, Test, Random, Distributions

@testset "confint y keyword guard (refs #1524)" begin
    Random.seed!(271127)
    Y = rand(Poisson(2), 3, 12)
    f = fit_gllvm(Y; family = Poisson(), K = 1, iterations = 80)
  err = try
        confint(f; y = Y)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("confint(fit, Y)", err.msg)
    @test occursin("keyword y", err.msg)
    @test confint(f, Y) isa NamedTuple
end