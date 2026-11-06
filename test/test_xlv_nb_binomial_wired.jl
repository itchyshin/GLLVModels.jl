using Test

@testset "nb/binomial xlv tests are wired into runtests (#948)" begin
    src = read(joinpath(@__DIR__, "runtests.jl"), String)
    @test occursin("_shard_include(\"test_phylo_binomial_xlv.jl\")", src)
    @test occursin("_shard_include(\"test_phylo_nb_xlv.jl\")", src)
    # Commented-out leftovers must not be the only mention.
    @test !occursin(r"#\s+_shard_include\(\"test_phylo_binomial_xlv\.jl\"\)", src)
    @test !occursin(r"#\s+_shard_include\(\"test_phylo_nb_xlv\.jl\"\)", src)
end
