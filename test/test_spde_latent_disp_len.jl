using GLLVModels, Test, Distributions

@testset "spde latent dispersion block length (refs #1227)" begin
    @test GLLVModels._spde_disp_len(Normal()) == 1
    @test GLLVModels._spde_disp_len(NegativeBinomial(3.0, 0.5)) == 1
    @test GLLVModels._spde_disp_len(Poisson()) == 0
    @test GLLVModels._spde_disp_len(Gamma(1.0, 1.0)) == 0
    @test GLLVModels._spde_disp_len(Beta(1.0, 1.0)) == 0
end