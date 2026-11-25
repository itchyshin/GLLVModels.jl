using GLLVModels, LinearAlgebra, Random, Test

# #1163: default start can stall at Λ_phy ≈ 0 with Optim converged=true while an
# interior start reaches a substantially higher log-likelihood. Assert the
# boundary stall is labeled, not an absolute log-likelihood gap (#505 / #1163).

const _NW_1163 = "((((t11:1.2854931105799,((t12:0.3252299389332,t5:0.3252299389332):0.3939704613692,((t6:0.2460238332159,t8:0.2460238332159):0.3551003455321,(t10:0.2887053781474,t7:0.2887053781474):0.3124188006006):0.1180762215544):0.5662927102775):0.5165334328704,t3:1.8020265434503):0.5102632290564,((t9:0.5005865681419,t2:0.5005865681419):0.5126771867901,t1:1.0132637549321):1.2990260175747):0.3904911086323,t4:2.7027808811391);"

@testset "phylo_dep 1163 zero Lambda_phy stall verdict" begin
    phy = augmented_phy(_NW_1163)
    p = phy.n_leaves
    Σ = sigma_phy_dense(phy; σ²_phy = 1.0)
    ds = 3
    Y = 0.8 .* (cholesky(Symmetric(Σ)).L * randn(MersenneTwister(ds), p)) .+
        0.5 .* randn(MersenneTwister(ds + 10), p, 20)
    f = fit_phylo_dep_gllvm(Y, phy)
    f0 = fit_gaussian_gllvm(Y; K = 1)
    @test norm(f.pars.Λ_phy) < 1e-5
    @test f.logLik ≈ f0.logLik atol = 1e-8
    @test !f.converged
    @test f.stopping_reason === :boundary
    @test f.stopping_reason !== :converged
end