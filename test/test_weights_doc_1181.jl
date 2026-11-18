using GLLVModels, Test

# `?fit_gllvm` said a weight multiplies the conditional log-density and did
# not say what that means once a latent term is present (#1181).

_doc1181(needle) = occursin(needle, string(@doc fit_gllvm))

@testset "fit_gllvm weights docstring states the latent-term meaning (#1181)" begin
    @test _doc1181("integer weight `w` is not the same as repeating the unit")
    @test _doc1181("constant weight `c` is not a `c`-fold log-likelihood")
    @test _doc1181("at `K = 0` both of those hold")
end
