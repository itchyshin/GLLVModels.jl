# Lane 1124d: spatial_dep entry-point validation (Arc 0; no mesh transport).
using Test, GLLVModels

@testset "1124d spatial_dep.jl" begin
  Y = randn(3, 5)
  @test_throws ArgumentError fit_spatial_dep_gllvm(Y, randn(2, 2))
  err = try
    fit_spatial_dep_gllvm(Y, randn(3, 2))
    nothing
  catch e
    e
  end
  @test err isa ArgumentError
  @test occursin("not implemented", lowercase(err.msg))
end