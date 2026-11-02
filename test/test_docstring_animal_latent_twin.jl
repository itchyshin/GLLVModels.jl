using GLLVModels
using Test

@testset "animal_latent docstring names the right twin (#823)" begin
    doc = string(@doc fit_animal_latent_gllvm)

    @test occursin("Hadamard", doc)
    @test occursin("not", doc) && occursin("Σ_trait ⊗ A", doc)
    @test occursin("fit_kernel_latent_gllvm", doc)
    @test !occursin("same estimand class", doc)
end
