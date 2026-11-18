using Test
using GLLVModels, Random

@testset "confint rejects missing/NaN y unlike fit data (refs #1144)" begin
    Random.seed!(1)
    p, n = 6, 40
    Y = 0.8 .* randn(p, 1) * randn(1, n) .+ 0.5 .* randn(p, n)
    fit = fit_gaussian_gllvm(Y; K = 1)
    Ym = Matrix{Union{Missing, Float64}}(Y)
    Ym[1, 2] = missing
    Yn = copy(Y)
    Yn[1, 2] = NaN
    @test all(isfinite, confint(fit, Y).lower)
    @test_throws ArgumentError confint(fit, Ym)
    @test_throws ArgumentError confint(fit, Yn)
    trm = confint(fit, Y).term[1]
    @test_throws ArgumentError confint(fit, Ym; method = :profile, parm = trm)
    @test_throws ArgumentError profile_ci(fit, trm; y = Ym)
end