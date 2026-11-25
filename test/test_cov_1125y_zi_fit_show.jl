# Coverage round 4: ZiFit show and _zi_rname helpers in zi_twin.jl.
using Test
using GLLVModels

const G = GLLVModels

@testset "1125y zi twin show and names" begin
    @test G._zi_rname(G.ZiPoisson()) == "zi_poisson"
    @test G._zi_rname(G.ZiNbinom2()) == "zi_nbinom2"
    @test G._zi_rname(G.ZiBinomial()) == "zi_binomial"
    @test G._offset_unsupported(G.ZiPoisson()) === true
    @test G.default_link(G.ZiPoisson()) isa G.LogLink
    @test G.default_link(G.ZiBinomial()) isa G.LogitLink

    fit = G.ZiFit(
        :zi_poisson, [0.0, 0.0], zeros(2, 1), [0.0, 0.0], [0.1, 0.2],
        fill(1.0, 2), nothing, -10.0, true, 5, 1.0,
    )
    s = sprint(show, fit)
    @test occursin("ZiFit", s)
    @test occursin("zi_poisson", s)
    @test occursin("loglik", s)

    fitnc = G.ZiFit(
        :zi_binomial, [0.0], zeros(1, 1), [0.0], [0.2], fill(1.0, 1),
        fill(3, 1, 2), -5.0, false, 0, 0.5,
    )
    @test occursin("NOT CONVERGED", sprint(show, fitnc))

    @testset "direct _zi_admit messages" begin
        Y = fill(0.5, 2, 4)
        @test_throws ArgumentError G._zi_admit(G.ZiPoisson(), Y, nothing)
        @test_throws ArgumentError G._zi_admit(G.ZiNbinom2(), Y, fill(3, 2, 4))
        err = try
            G._zi_admit(G.ZiBinomial(), zeros(Int, 2, 2), [1 1; 1 1])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("trials >= 2", err.msg)
    end
end