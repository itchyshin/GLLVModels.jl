using Test
using GLLVModels, Random, Distributions

@testset "issue 1585: Laplace confint/vcov/coeftable require Y (refs #1585, #1584)" begin
    Random.seed!(1127)
    p, n, K = 4, 50, 1
    Yp = rand(Poisson(3), p, n)
    Yb = rand(Bernoulli(0.4), p, n)
    fit_p = fit_gllvm(Yp; family = Poisson(), K = K)
    fit_b = fit_binomial_gllvm(Yb; K = K)
    msg = "requires the response matrix `Y`"
    for (fit, Y) in ((fit_p, Yp), (fit_b, Yb))
        for f in (confint, vcov, coeftable)
            err = try
                f(fit)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin(msg, sprint(showerror, err))
        end
        @test confint(fit, Y) isa NamedTuple
        @test vcov(fit, Y) isa AbstractMatrix
        @test coeftable(fit, Y) isa GLLVModels.GllvmCoefTable
    end
end