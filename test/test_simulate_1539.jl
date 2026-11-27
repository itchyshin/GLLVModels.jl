# refs #1539 — simulate(fit, n) for OrdinalPerTraitFit (default Ordinal() route)
using GLLVModels, Test, Random, Distributions, Statistics

@testset "simulate OrdinalPerTraitFit (#1539)" begin
    Random.seed!(1)
    p, K, n = 3, 1, 12
    Y = rand(1:3, p, n)
    fit = fit_gllvm(Y; family = Ordinal(), K = K, iterations = 500)
    @test fit isa OrdinalPerTraitFit

    rng = MersenneTwister(1539)
    n_sim = 4000
    Ys = simulate(fit, n_sim; rng = rng)
    @test size(Ys) == (p, n_sim)
    @test eltype(Ys) <: Integer
    @test all(1 .<= Ys .<= 3)

    # Category-1 probability under :prob for a coarse mean check
    P = predict(fit, Ys; type = :prob)
    Ecat = zeros(p)
    for t in 1:p
        Ecat[t] = mean(@view(P[t, :, 1]))
    end
    frac1 = vec(mean(Ys .== 1; dims = 2))
    @test isapprox(frac1, Ecat; rtol = 0.35)
end