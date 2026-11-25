# Issue #1161: GP1 and per-group NB1 fitters must refuse a link whose mean range is not
# (0, Inf) with a clear ArgumentError, not a raw DomainError from the Laplace inner loop.
using Test, Random, Distributions, GLLVModels
const GM1161 = GLLVModels

@testset "GP1 / NB1 grouped: unsupported link refused (#1161)" begin
    rng = MersenneTwister(1161)
    Y = rand(rng, Distributions.Poisson(3.0), 5, 30)
    for bad in (GM1161.LogitLink(), GM1161.ProbitLink(), GM1161.CLogLogLink())
        e = try fit_gp1_gllvm(Y; K = 1, link = bad); nothing catch err; err end
        @test e isa ArgumentError
        e isa ArgumentError && @test occursin("fit_gp1_gllvm", e.msg) && occursin("LogLink", e.msg)
        e = try fit_nb1_gllvm_grouped(Y; K = 1, link = bad); nothing catch err; err end
        @test e isa ArgumentError
        e isa ArgumentError && @test occursin("fit_nb1_gllvm_grouped", e.msg)
        X = zeros(5, 30, 1)
        e = try fit_nb1_gllvm_grouped_cov(Y; X = X, K = 1, link = bad); nothing catch err; err end
        @test e isa ArgumentError
    end
    # supported links still fit
    @test fit_gp1_gllvm(Y; K = 1) isa GP1Fit
    @test fit_nb1_gllvm_grouped(Y; K = 1) isa NB1GroupedFit
end
