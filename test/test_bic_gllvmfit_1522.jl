using GLLVModels, Test, Random, StatsModels, Distributions

@testset "bic(fit) on GllvmFit (refs #1522)" begin
    Random.seed!(2701)
    p, n, K = 3, 25, 1
    Y = randn(p, n)
    data = (site_cov = zeros(n),)

    fit = gllvm(@formula(y ~ 1), Y, data; family = Normal(), K = K)
    @test isfinite(aic(fit))

    err = try
        bic(fit)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("bic(fit, Y)", err.msg)

    pn = nobs(fit, Y)
    @test isfinite(bic(fit, Y))
    @test bic(fit, Y) ≈ bic(fit, pn)
end