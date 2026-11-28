using Test, GLLVModels, Random

@testset "bridge_fit: ci_seed must be an integer" begin
    rng = MersenneTwister(7)
    Y = Float64.(rand(rng, 0:6, 5, 60))
    call(v; family = "poisson") = bridge_fit(; y = Y, family = family, d = 1,
        options = Dict("ci_method" => "bootstrap", "ci_nboot" => 20, "ci_seed" => v))

    @test call(1).loglik isa Real
    @test call(2.0).loglik isa Real
    for bad in (2.5, NaN, "a")
        err = try; call(bad); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("ci_seed", err.msg)
    end
    # ci_seed is read before ci_method is applied — bad seed fails even when CIs are off
    err = try
        bridge_fit(; y = Y, family = "poisson", d = 1,
                   options = Dict("ci_method" => "none", "ci_seed" => 2.5))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("ci_seed", err.msg)
end