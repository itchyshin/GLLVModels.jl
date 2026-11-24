# fit_mixed_gllvm / fit_gllvm(family = [...]): a family marker outside the supported set
# (Normal, Poisson, Binomial, NegativeBinomial, Gamma, Beta) is refused with the
# documented ArgumentError whether or not `links` is given. Before this check the
# default-link step ran first and raised `MethodError: no method matching
# default_link(::X)`, so the documented message was reachable only with explicit links.
using Test
using GLLVModels
using Distributions
using Random

const _MU = GLLVModels

@testset "mixed fit: unsupported family marker gives the documented ArgumentError" begin
    rng = MersenneTwister(1)
    p, n = 4, 20
    Y = Float64.(rand(rng, 1:5, p, n))
    base = Any[Normal(), Poisson(), Binomial(), NegativeBinomial()]
    for mk in (() -> _MU.TweedieED(1.5, 1.0), () -> _MU.NB1(1.0), () -> _MU.ZIPoisson(),
               () -> Exponential(), () -> _MU.DeltaGamma(), () -> _MU.HurdlePoisson(),
               () -> _MU.OrderedBeta(), () -> _MU.Ordinal())
        fs = copy(base); fs[4] = mk()
        e1 = try fit_mixed_gllvm(Y; families = fs, K = 1); nothing catch e; e end
        @test e1 isa ArgumentError
        @test occursin("fit_mixed_gllvm v1 supports", sprint(showerror, e1))
        e2 = try fit_gllvm(Y; family = fs, K = 1); nothing catch e; e end
        @test e2 isa ArgumentError
    end
    # supported mixes are unchanged
    f = fit_mixed_gllvm(Y; families = Any[Normal(), Poisson(), Binomial(), NegativeBinomial()],
                        K = 1, N = fill(5, p, n))
    @test f isa MixedFamilyFit
end
