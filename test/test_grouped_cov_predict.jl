using GLLVModels, Test, Random, Distributions, StatsModels

# Issue #1529: Beta/Gamma/Ordinal grouped @formula covariate fits lacked in-sample predict.

@testset "#1529 grouped cov in-sample predict(fit, Y, X)" begin
  Random.seed!(11271127)
  p, n = 5, 30
  x = collect(range(-1.0, 1.0; length = n))
  data = (x = x,)

  mm, _ = GLLVModels._build_site_modelmatrix(@formula(y ~ 1 + x).rhs, data)
  X = zeros(p, n, size(mm, 2))
  for k in axes(mm, 2), s in 1:n, t in 1:p
    X[t, s, k] = mm[s, k]
  end

  Yβ = [clamp(rand(Beta(2, 5)) + 0.05 * x[s], 1e-4, 1 - 1e-4) for t in 1:p, s in 1:n]
  fitβ = gllvm(@formula(y ~ 1 + x), Yβ, data; family = Beta(), K = 1, iterations = 400)
  @test fitβ isa BetaGroupedCovFit
  μβ = predict(fitβ, Yβ, X)
  @test size(μβ) == (p, n)
  @test all(0 .< μβ .< 1)
  @test fitted(fitβ, Yβ, X) == μβ
  Lβ = predict(fitβ, Yβ, X; type = :link)
  @test size(Lβ) == (p, n)
  @test_throws MethodError predict(fitβ, X)

  Yg = [rand(Gamma(2, 2.0)) for t in 1:p, s in 1:n]
  fitγ = gllvm(@formula(y ~ 1 + x), Yg, data; family = Gamma(), K = 1, iterations = 400)
  @test fitγ isa GammaGroupedCovFit
  μg = predict(fitγ, Yg, X)
  @test size(μg) == (p, n)
  @test all(μg .> 0)
  @test fitted(fitγ, Yg, X) == μg
  @test_throws MethodError predict(fitγ, X)

  Yo = [rand(1:4) for t in 1:p, s in 1:n]
  fito = gllvm(@formula(y ~ 1 + x), Yo, data; family = Ordinal(), K = 1, iterations = 500)
  @test fito isa OrdinalPerTraitCovFit
  cls = predict(fito, Yo, X)
  @test size(cls) == (p, n)
  @test all(1 .<= cls .<= 4)
  @test predict(fito, Yo, X; type = :response) == cls
  P = predict(fito, Yo, X; type = :prob)
  @test size(P) == (p, n, 4)
  @test_throws MethodError predict(fito, X)

  Ynb = [rand(NegativeBinomial(5, 0.4)) for t in 1:p, s in 1:n]
  fitnb = gllvm(@formula(y ~ 1 + x), Ynb, data; family = NegativeBinomial(), K = 1, iterations = 400)
  @test size(predict(fitnb, Ynb, X)) == (p, n)
end