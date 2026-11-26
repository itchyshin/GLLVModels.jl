# refs #1430: docs must show widening Y before assigning `missing` on a numeric matrix.
using GLLVModels, Random, Test

const _RESP_FAM = joinpath(@__DIR__, "..", "docs", "src", "response-families.md")

@testset "issue 1430 missing-response docs (#1107 workflow)" begin
  text = read(_RESP_FAM, String)
  @test occursin("allowmissing(Y)", text)
  @test occursin("Matrix{Union{Missing, Float64}}(Y)", text)
  @test occursin("MethodError", text)

  rng = MersenneTwister(777)
  Y = randn(rng, 4, 10)
  @test_throws MethodError (Y[2, 3] = missing)

  # Documented snippet in response-families.md (Gaussian — Normal())
  Ym = Matrix{Union{Missing, Float64}}(Y)
  Ym[2, 3] = missing
  fit = fit_gaussian_gllvm(Ym; K = 1, iterations = 20, g_tol = 1e-4)
  @test isfinite(fit.logLik)
  @test GLLVModels.observed_mask(Ym) == [t != 2 || s != 3 for t in 1:4, s in 1:10]
end