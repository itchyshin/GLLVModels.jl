using GLLVModels, Test, Random, LinearAlgebra, Statistics

@testset "Gaussian profile sufficient statistics (X covariate)" begin
  Random.seed!(88428)
  p, n, q, K = 5, 40, 1, 1
  X = randn(n, 1)
  X3 = zeros(Float64, p, n, q)
  for s in 1:n, t in 1:p
    X3[t, s, 1] = X[s, 1]
  end
  βx = 0.4
  Λ = 0.7 .* ones(p, K)
  y = Λ * randn(K, n) .+ βx .* reshape(X3[:, :, 1], p, n) .+ 0.3 .* randn(p, n)

  suff = GLLVModels._gaussian_suff(y, X3, q)
  @test suff.q == q
  @test size(suff.Syy) == (p, p)
  @test size(suff.C) == (p * p, q)
  @test size(suff.Sxx) == (p * p, q * q)

  fit_stock = fit_gaussian_gllvm(y; K = K, X = X3)
  fit_ss = fit_gaussian_gllvm(y; K = K, X = X3, _sufficient_stat = true)
  @test fit_stock.logLik ≈ fit_ss.logLik atol = 1e-9
  @test fit_stock.pars.σ_eps ≈ fit_ss.pars.σ_eps rtol = 1e-8

  ci_sym = profile_ci(fit_ss, :sigma_eps; y = y)
  ci_str = profile_ci(fit_ss, "sigma_eps"; y = y)
  @test ci_sym.method == ci_str.method
  if isfinite(ci_sym.lower) && isfinite(ci_sym.upper)
    @test ci_str.lower ≈ ci_sym.lower
    @test ci_str.upper ≈ ci_sym.upper
  else
    @test isnan(ci_sym.lower) || isnan(ci_sym.upper)
    @test isnan(ci_str.lower) || isnan(ci_str.upper)
  end
end