# Lane cursor-cloud-1125y9: aghq_fit_info.jl request/controls validation.
using Test, GLLVModels

const G = GLLVModels

@testset "1125y9 aghq controls" begin
    @testset "_aghq_request" begin
        @test G._aghq_request(false) === :off
        @test G._aghq_request(nothing) === :off
        @test G._aghq_request(true) === :auto
        @test G._aghq_request(:auto) === :auto
        @test G._aghq_request(5) == 5
        @test_throws ArgumentError G._aghq_request(-1)
        @test_throws ArgumentError G._aghq_request(1.5)
        @test_throws ArgumentError G._aghq_request(:on)
    end

    @testset "_aghq_controls defaults" begin
        c = G._aghq_controls((n_adapt = 100,))
        @test c.n_adapt == 100
        @test c.continuation isa Bool
        @test c.mode_tol > 0
        @test 0 < c.rho_min <= 1
    end

    @testset "_aghq_controls validation" begin
        @test_throws ArgumentError G._aghq_controls((bad_key = 1,))
        @test_throws ArgumentError G._aghq_controls((n_adapt = 0,))
        @test_throws ArgumentError G._aghq_controls((continuation = 1,))
        @test_throws ArgumentError G._aghq_controls((shift_tol = NaN,))
        @test_throws ArgumentError G._aghq_controls((rho_min = 1.1,))
        @test_throws ArgumentError G._aghq_controls((mode_tol = 0.0,))
        @test_throws ArgumentError G._aghq_controls("not a named tuple")
    end

    @testset "_aghq_outer_controls projection" begin
        c = G._aghq_controls((;))
        o = G._aghq_outer_controls(c)
        @test haskey(o, :n_adapt)
        @test !haskey(o, :mode_gradient_tol)
    end
end