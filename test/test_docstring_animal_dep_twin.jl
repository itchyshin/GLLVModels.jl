using GLLVModels
using Test

@testset "animal_dep / kernel_dep docstrings name the right twin (#823)" begin
    animal = string(@doc fit_animal_dep_gllvm)
    kernel = string(@doc fit_kernel_dep_gllvm)

    @testset "fit_animal_dep_gllvm is the Hadamard row model, not Σ_trait ⊗ A" begin
        @test occursin("Hadamard", animal)
        @test occursin("not", animal) && occursin("Σ_trait ⊗ A", animal)
        @test occursin("fit_kernel_dep_gllvm", animal)
        @test !occursin("Same estimand class as twin", animal)
    end

    @testset "fit_kernel_dep_gllvm is the Kronecker model" begin
        @test occursin("Kronecker", kernel)
        @test occursin("fit_phylo_dep_gllvm", kernel)
        @test occursin("Hadamard", kernel)
        @test !occursin("Same estimand class as twin", kernel)
    end
end
