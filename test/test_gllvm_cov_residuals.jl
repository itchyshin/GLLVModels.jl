using GLLVModels, Test, Random, Distributions, Statistics, StatsModels

# Issue #1528: GllvmCovFit (Poisson/Binomial @formula and fit_gllvm_cov) had no residuals.

@testset "#1528 GllvmCovFit residuals(fit, Y, X)" begin
  Random.seed!(27001)
  logistic(x) = 1 / (1 + exp(-x))

  p, n = 4, 25
  x = collect(range(-1.0, 1.0; length = n))
  data = (x = x,)

  Yp = [rand(Poisson(exp(0.1 + 0.4 * x[s]))) for t in 1:p, s in 1:n]
  fit_p = gllvm(@formula(y ~ 1 + x), Yp, data; family = Poisson(), K = 1, iterations = 300)

  N = fill(8, p, n)
  Yb = [rand(Binomial(N[t, s], logistic(0.2 + 0.3 * x[s]))) for t in 1:p, s in 1:n]
  fit_b = gllvm(@formula(y ~ 1 + x), Yb, data; family = Binomial(), K = 1, N = N, iterations = 400)

  mm, _ = GLLVModels._build_site_modelmatrix(@formula(y ~ 1 + x).rhs, data)
  X = zeros(p, n, size(mm, 2))
  for k in axes(mm, 2), s in 1:n, t in 1:p
    X[t, s, k] = mm[s, k]
  end

  μp = predict(fit_p, Yp, X; type = :response)
  Rp = residuals(fit_p, Yp, X; type = :response)
  @test Rp ≈ Yp .- μp
  Rpp = residuals(fit_p, Yp, X; type = :pearson)
  @test Rpp ≈ (Yp .- μp) ./ sqrt.(μp) rtol = 1e-10

  fit_p2 = fit_gllvm_cov(Yp; family = Poisson(), X = X, K = 1, iterations = 300)
  @test residuals(fit_p2, Yp, X; type = :response) ≈ Yp .- predict(fit_p2, Yp, X)

  μb = predict(fit_b, Yb, X; type = :response, N = N)
  Rb = residuals(fit_b, Yb, X; type = :response, N = N)
  @test Rb ≈ Yb .- μb
  Rbp = residuals(fit_b, Yb, X; type = :pearson, N = N)
  @test Rbp ≈ (Yb .- N .* μb) ./ sqrt.(N .* μb .* (1 .- μb)) rtol = 1e-10

  @test_throws ArgumentError residuals(fit_p, Yp, X; type = :dunnsmyth)
end