using Test
using GLLVModels, LinearAlgebra, Random

@testset "confint/vcov without Σ_phy on phylo Gaussian fit (refs #1178)" begin
    phy = augmented_phy("((A:0.3,B:0.2):0.4,(C:0.5,D:0.1):0.2);")
    p = phy.n_leaves
    rng = MersenneTwister(3)
    Σ = Matrix(sigma_phy_dense(phy; σ²_phy = 1.0))
    Y = 0.5 .* (cholesky(Symmetric(Σ)).L * randn(rng, p)) .+ 0.3 .* randn(rng, p, 25)
    fit = fit_phylo_dep_gllvm(Y, phy)
    @test fit.converged
    @test_throws ArgumentError confint(fit, Y)
    @test_throws ArgumentError vcov(fit, Y)
    @test_throws ArgumentError stderror(fit, Y)
    ci = confint(fit, Y; Σ_phy = Σ)
    @test all(isfinite, ci.lower)
    @test all(isfinite, ci.upper)
end