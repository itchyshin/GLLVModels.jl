using Test, Random, Distributions, GLLVModels

@testset "predict on training Y with missing cells (refs #1483)" begin
    @testset "Gamma" begin
        Random.seed!(55)
        Y = abs.(randn(4, 20)) .+ 0.1
        Ym = Matrix{Union{Missing, Float64}}(Y)
        Ym[1, 1] = missing
        fit = fit_gllvm(Ym; family = Gamma(), K = 1, iterations = 120)
        μ = predict(fit, Ym)
        @test size(μ) == size(Ym)
        @test all(isfinite, skipmissing(μ))
    end
    @testset "Poisson" begin
        Random.seed!(56)
        Y = rand(Poisson(2), 4, 20)
        Ym = Matrix{Union{Missing, Int}}(Y)
        Ym[2, 3] = missing
        fit = fit_gllvm(Ym; family = Poisson(), K = 1, iterations = 120)
        λ = predict(fit, Ym)
        @test size(λ) == size(Ym)
        @test all(x -> ismissing(x) || isfinite(x), λ)
    end
end