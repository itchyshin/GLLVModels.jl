using GLLVModels, Test, TOML
@assert realpath(Base.pkgdir(GLLVModels)) == realpath(joinpath(@__DIR__, ".."))
module Core070ShapeBoundaries
using GLLVModels, Test, TOML
import Distributions
struct ResponseRead <: Exception end
struct NoRead <: AbstractMatrix{Float64} end
Base.size(::NoRead) = (2, 4)
Base.getindex(::NoRead, ::Int, ::Int) = throw(ResponseRead())
const Y = NoRead()
function observed(f)
    try
        f()
        "RETURNED"
    catch error
        error isa ResponseRead ? "RESPONSE_READ_SENTINEL" : string(nameof(typeof(error)))
    end
end
const results = Dict{String,Any}()
function boundary(id, call, expected)
    result = observed(call)
    results[id] = result
    println(id, '\t', result)
    @test result == expected
end
@testset "Frozen R shape boundaries and explicit Julia extensions" begin
    for (suffix,power) in (("LOW",1.0),("HIGH",2.0),("INF",Inf),("NA",NaN))
        boundary("FAMILY-06-SHAPE-INVALID-"*suffix,
            ()->fit_tweedie_gllvm_grouped(Y;K=1,power=power),"ArgumentError")
    end
    boundary("FAMILY-06-SHAPE-INVALID-VECTOR",
        ()->fit_tweedie_gllvm_grouped(Y;K=1,power=[2.0,3.0]),"TypeError")
    for (suffix,nu) in (("INF",Inf),("NA",NaN))
        boundary("FAMILY-09-SHAPE-INVALID-"*suffix,
            ()->fit_studentt_gllvm(Y;K=1,nu=nu),"ArgumentError")
    end
    boundary("FAMILY-09-SHAPE-INVALID-LOW",
        ()->fit_studentt_gllvm(Y;K=1,nu=1.0),"RESPONSE_READ_SENTINEL")
    boundary("FAMILY-09-SHAPE-INVALID-VECTOR",
        ()->fit_studentt_gllvm(Y;K=1,nu=[2.0,3.0]),"RESPONSE_READ_SENTINEL")
    # Positive and adjacent negative controls distinguish domain checks from a dead entry point.
    @test observed(()->fit_tweedie_gllvm_grouped(Y;K=1,power=1.5)) == "RESPONSE_READ_SENTINEL"
    @test observed(()->fit_studentt_gllvm(Y;K=1,nu=4.0)) == "RESPONSE_READ_SENTINEL"
    @test observed(()->fit_studentt_gllvm(Y;K=1,nu=[2.0])) == "ArgumentError"
    @test observed(()->fit_studentt_gllvm(Y;K=1,nu=[2.0,NaN])) == "ArgumentError"
    @test observed(()->fit_studentt_gllvm(Y;K=1,nu=0.0)) == "ArgumentError"
    # Exact df1 Cauchy law, independent Distributions implementation.
    for location in (-2.0,0.0,3.0), scale in (0.1,1.0,4.0), residual in (-10.0,0.0,0.25,8.0)
        y=location+residual*scale
        actual=GLLVModels._glm_logpdf(StudentTFamily(1.0,scale),location,1,y)
        expected=Distributions.logpdf(Distributions.Cauchy(location,scale),y)
        @test actual ≈ expected atol=1e-12 rtol=1e-12
    end
end
if haskey(ENV,"CORE070_BOUNDARY_OUTPUT")
    open(io->TOML.print(io,Dict("status"=>"PASS","scope"=>"DOMAIN_AND_KERNEL_ONLY_NO_FITS",
        "results"=>results,"julia_version"=>string(VERSION),"package_root"=>realpath(Base.pkgdir(GLLVModels)))),ENV["CORE070_BOUNDARY_OUTPUT"],"w")
end
end
if haskey(ENV,"CORE070_BOUNDARY_OUTPUT")
    using SHA
    println("BOUNDARY_SHA256 ",bytes2hex(sha256(read(ENV["CORE070_BOUNDARY_OUTPUT"]))))
end
