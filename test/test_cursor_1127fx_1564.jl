# row_effects(fit, Y) for RowEffectFit (#1564).

using GLLVModels, Test, Random, Distributions

@testset "row_effects RowEffectFit (#1564)" begin
  Y = [
      4 1 1 2 4 3 4 1 2 4 2 2 5 1 1 5 4 2 4 4
      1 5 1 2 3 3 5 2 2 2 6 3 5 4 5 5 5 4 2 0
      2 2 5 6 2 5 4 1 6 4 6 1 3 1 4 6 1 2 2 3
      5 2 1 0 3 3 4 1 4 5 4 3 5 1 4 4 0 1 4 1
      2 4 6 3 4 2 1 2 1 5 3 6 2 3 0 3 6 6 2 2
  ]
  fit = fit_roweffect_gllvm(Y; family = Poisson(), K = 1, iterations = 80)
  ρ = row_effects(fit, Y)
  @test ρ == fit.ρ
  @test length(ρ) == size(Y, 2)
  @test ρ[1] == 0.0
  @test all(isfinite, ρ)
  @test_throws ArgumentError row_effects(fit, Y[:, 1:(end - 1)])
end