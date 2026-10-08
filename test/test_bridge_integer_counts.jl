using Test, Random, GLLVModels

const _BRIDGE_OPTS = Dict("ci_method" => "none")

function _bridge_poisson(y)
    GLLVModels.bridge_fit(y = y, family = "poisson", d = 1, options = _BRIDGE_OPTS)
end

@testset "bridge_fit rejects non-integer count inputs (#740)" begin
    Random.seed!(1)
    Y = rand(0:3, 4, 40)
    Yf = Float64.(Y)
    r_int = _bridge_poisson(Yf)
    @test r_int.converged

    err = @test_throws ArgumentError _bridge_poisson(Yf .+ 0.45)
    @test occursin("not rounded", sprint(showerror, err.value))

    err_half = @test_throws ArgumentError _bridge_poisson(Yf .+ 0.5)
    @test occursin("not rounded", sprint(showerror, err_half.value))

    N = fill(10.0, size(Yf)...)
    err_n = @test_throws ArgumentError GLLVModels.bridge_fit(
        y = Yf, family = "binomial", d = 1, N = N .+ 0.4, options = _BRIDGE_OPTS)
    @test occursin("trial", lowercase(sprint(showerror, err_n.value)))
end
