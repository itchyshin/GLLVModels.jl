using GLLVModels, Test
using GLLVModels: StatsAPI, dof, loglikelihood, aic, _nparams

@testset "EMPhyloFit and BranchREFit _nparams (#745)" begin
    p, K_B = 6, 2
    emf = EMPhyloFit(
        zeros(p, K_B),
        1.0,
        ones(p),
        -50.0,
        0,
        true,
        Float64[],
        zeros(p),
        zeros(p),
    )
    k_em = 1 + (p * K_B - div(K_B * (K_B - 1), 2)) + p
    @test k_em > 1
    @test _nparams(emf) == k_em
    @test dof(emf) == k_em
    @test aic(emf) ≈ 2 * k_em - 2 * loglikelihood(emf)

    br = BranchREFit(0.0, 1.0, 0.5, 12.0, [0.0], [0.0], [0.0], 3, true)
    @test _nparams(br) == 3
    @test dof(br) == 3
    @test loglikelihood(br) ≈ -12.0
    @test aic(br) ≈ 2 * 3 - 2 * loglikelihood(br)
end
