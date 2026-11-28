using Test, GLLVModels, Random, Distributions

@testset "COMPoisson refuses non-log links" begin
    rng = MersenneTwister(3)
    p, n = 6, 50
    z = randn(rng, n)
    L = [0.9, 0.6, -0.7, 0.5, 0.8, -0.4]
    Y = [rand(rng, Poisson(exp(0.8 + 0.5 * L[j] * z[i]))) for j in 1:p, i in 1:n]
    f0 = fit_compoisson_gllvm(Y; K = 1, link = LogLink())
    for l in (LogitLink(), ProbitLink(), IdentityLink())
        err = try; fit_compoisson_gllvm(Y; K = 1, link = l); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("LogLink", err.msg)
        err2 = try
            fit_gllvm(Y; family = COMPoisson(), K = 1, link = l)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
    end
    f0b = fit_compoisson_gllvm(Y; K = 1)
    @test f0.loglik == f0b.loglik
end