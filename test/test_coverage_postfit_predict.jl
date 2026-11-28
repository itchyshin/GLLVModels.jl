using GLLVModels, Test, Random, Distributions, Statistics

@testset "postfit predict/residuals (one-part families)" begin
  @testset "Gamma + Exponential dunnsmyth vs pearson" begin
    Random.seed!(88425)
    p, K, n = 5, 1, 100
    β = 0.15 .* randn(p)
    Λt = 0.5 .* randn(p, K)
    Z = randn(K, n)
    η = β .+ Λt * Z

    Yg = [rand(Gamma(2.0, exp(η[t, s]) / 2)) for t in 1:p, s in 1:n]
    fit_g = fit_gamma_gllvm(Yg; K = K)
    μg = predict(fit_g, Yg; type = :response)
    @test all(μg .> 0)
    rg = residuals(fit_g, Yg; type = :dunnsmyth)
    rp = residuals(fit_g, Yg; type = :pearson)
    @test size(rg) == (p, n) && all(isfinite, rg)
    @test rp ≈ (Yg .- μg) ./ sqrt.(μg.^2 ./ fit_g.α) atol = 1e-8

    Ye = [rand(Exponential(exp(η[t, s]))) for t in 1:p, s in 1:n]
    fit_e = fit_exponential_gllvm(Ye; K = K, iterations = 80)
    if isfinite(fit_e.loglik)
      μe = predict(fit_e, Ye; type = :response)
      re = residuals(fit_e, Ye; type = :dunnsmyth)
      @test all(isfinite, re)
      @test fitted(fit_e, Ye) ≈ μe
    end
  end

  @testset "Beta response + NB pearson" begin
    Random.seed!(88426)
    p, K, n = 4, 1, 90
    β = 0.2 .* randn(p)
    Λt = 0.4 .* randn(p, K)
    η = β .+ Λt * randn(K, n)

    Yβ = [rand(Beta(1.5, 2.0)) for t in 1:p, s in 1:n]
    fit_β = fit_beta_gllvm(Yβ; K = K)
    pr = predict(fit_β, Yβ; type = :response)
    @test all(0 .< pr .< 1)
    rβ = residuals(fit_β, Yβ; type = :pearson)
    @test all(isfinite, rβ)

    r_nb = 1.6
    Yn = [rand(NegativeBinomial(r_nb, r_nb / (r_nb + exp(η[t, s])))) for t in 1:p, s in 1:n]
    fit_n = fit_nb_gllvm(Yn; K = K)
    μn = predict(fit_n, Yn; type = :response)
    rn = residuals(fit_n, Yn; type = :pearson)
    @test rn ≈ (Yn .- μn) ./ sqrt.(μn .+ μn.^2 ./ fit_n.r) atol = 1e-8
  end

  @testset "binomial cloglog residuals" begin
    Random.seed!(88427)
    p, K, n = 4, 1, 70
    η = 0.15 .* randn(p, n)
    μ = 1 .- exp.(-exp.(η))
    Y = Int.(rand(p, n) .< μ)
    fit = fit_binomial_gllvm(Y; K = K, link = CLogLogLink())
    μhat = predict(fit, Y; type = :response)
    @test all(0 .≤ μhat .≤ 1)
    r = residuals(fit, Y; type = :pearson)
    @test all(isfinite, r)
    v = link_residual(fit, Y)
    @test all(v .≈ π^2 / 6)
  end
end