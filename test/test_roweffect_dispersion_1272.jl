using Test, GLLVModels, Random, Distributions

# R scaffold `.convert_dispersion` reads `julia_fit$dispersion` on row-effect fits.
@testset "row-effect fits expose shared dispersion (refs 1272)" begin
    rng = MersenneTwister(2)
    Y = rand(rng, Poisson(3.0), 5, 40)
    f = fit_roweffect_gllvm(Y; family = NegativeBinomial(), K = 1)
    @test isfinite(f.dispersion)
    g = fit_roweffect_gllvm(rand(rng, Gamma(3, 1), 5, 40); family = Gamma(), K = 1)
    @test isfinite(g.dispersion)
end