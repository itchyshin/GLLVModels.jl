# (refs #1178) confint/vcov refuse missing Σ_phy on phylogenetic Gaussian fits
using Test, GLLVModels, Random, LinearAlgebra

@testset "confint requires Σ_phy on phylo fits (#1178)" begin
    nw = "((((t1:0.6,t2:0.6):0.3,t3:0.9):0.4,(t4:0.5,t5:0.5):0.8):0.3,((t6:0.6,t7:0.6):0.5,t8:1.1):0.5);"
    phy = augmented_phy(nw)
    p = phy.n_leaves
    n = 30
    Σ = Matrix(sigma_phy_dense(phy; σ²_phy = 1.0))
    Y = 0.8 .* (cholesky(Symmetric(Σ)).L * randn(MersenneTwister(3), p)) .+
        0.5 .* randn(MersenneTwister(4), p, n)
    fit = fit_phylo_dep_gllvm(Y, phy)
    @test_throws ArgumentError confint(fit, Y)
    @test_throws ArgumentError GLLVModels.StatsAPI.vcov(fit, Y)
    ci = confint(fit, Y; Σ_phy = Σ)
    @test any(isfinite, ci.lower)
end