using GLLVModels, Test, Random, Distributions, StatsModels

@testset "Binomial(n) family marker vs N keyword (refs #1523)" begin
    Random.seed!(2702)
    p, n, K = 3, 40, 1
    Y = rand(0:3, p, n)
    data = (z = zeros(n),)

    err = try
        gllvm(@formula(y ~ 1), Y, data; family = Binomial(10), K = K, iterations = 5)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("`N`", err.msg)
    @test occursin("Binomial(10)", err.msg)

    err2 = try
        fit_gllvm(Y; family = Binomial(10), K = K, N = fill(5, p, n))
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("conflicts", err2.msg)

    @test fit_gllvm(Y; family = Binomial(), K = K, N = fill(10, p, n),
        iterations = 50) isa BinomialFit
    @test fit_gllvm(Y; family = Binomial(10), K = K, N = fill(10, p, n),
        iterations = 50) isa BinomialFit
end