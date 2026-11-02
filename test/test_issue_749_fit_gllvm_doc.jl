using GLLVModels
using Test

@testset "fit_gllvm docstring documents optional K and Kmax (#749)" begin
    doc = string(@doc fit_gllvm)
    @test occursin("K = nothing", doc)
    @test occursin("Kmax", doc)
    @test occursin("select_lv", doc)
end
