using GLLVModels, Test, Random, Distributions, Statistics, SpecialFunctions, LinearAlgebra

@testset "link_residual (per-trait closed forms + latent extractors)" begin
  π2o3 = π^2 / 3
  π2o6 = π^2 / 6

  @testset "family markers (R extract-sigma twin)" begin
    μ = 4.2
    r = 1.5
    α = 3.0
    φ = 8.0
    @test link_residual(Normal(), IdentityLink(), μ, nothing) == 0.0
    @test link_residual(Poisson(), LogLink(), μ, nothing) ≈ log1p(1 / μ)
    @test link_residual(Poisson(), LogLink(), 0.0, nothing) == 0.0
    @test link_residual(Poisson(), LogLink(), -1.0, nothing) == 0.0
    @test link_residual(Binomial(), LogitLink(), μ, nothing) ≈ π2o3
    @test link_residual(Binomial(), ProbitLink(), μ, nothing) == 1.0
    @test link_residual(Binomial(), CLogLogLink(), μ, nothing) ≈ π2o6
    @test link_residual(NegativeBinomial(), LogLink(), μ, r) ≈ trigamma(r)
    @test link_residual(Gamma(), LogLink(), μ, α) ≈ trigamma(α)
    @test link_residual(StudentTFamily(5.0, 1.2), IdentityLink(), μ, 1.2) ≈
          1.2^2 * 5.0 / 3.0
    @test link_residual(StudentTFamily(2.0, 1.0), IdentityLink(), μ, 1.0) == Inf
    @test link_residual(Ordinal(), LogitLink(), μ, nothing) ≈ π2o3
    @test link_residual(Ordinal(), ProbitLink(), μ, nothing) == 1.0
    μc = 0.35
    @test link_residual(Beta(), LogitLink(), μc, φ) ≈
          trigamma(max(μc * φ, 1e-12)) + trigamma(max((1 - μc) * φ, 1e-12))
    @test_throws ArgumentError link_residual(Binomial(), IdentityLink(), μ, nothing)
  end

  @testset "on fitted models (σ²_d, Σ_latent, communality, correlation)" begin
    Random.seed!(88421)
    p, K, n = 4, 1, 80
    β = log.(fill(3.0, p))
    Λt = 0.35 .* randn(p, K)
    η = β .+ Λt * randn(K, n)
    Yp = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
    fit_p = fit_poisson_gllvm(Yp; K = K)
    σ2_p = link_residual(fit_p, Yp)
    @test length(σ2_p) == p
    @test all(σ2_p .> 0)
    μ̂_p = vec(mean(predict(fit_p, Yp; type = :response); dims = 2))
    @test σ2_p ≈ [log1p(1 / μ̂_p[t]) for t in 1:p] atol = 1e-10

    Yb = Int.(rand(p, n) .< 0.4)
    fit_b = fit_binomial_gllvm(Yb; K = K)
    @test all(link_residual(fit_b, Yb) .≈ π2o3)

    r_true = 2.0
    Yn = [rand(NegativeBinomial(r_true, r_true / (r_true + exp(η[t, s])))) for t in 1:p, s in 1:n]
    fit_n = fit_nb_gllvm(Yn; K = K)
    @test all(link_residual(fit_n, Yn) .≈ trigamma(fit_n.r))

    α_shape = 2.5
    Yg = [rand(Gamma(α_shape, exp(η[t, s]) / α_shape)) for t in 1:p, s in 1:n]
    fit_g = fit_gamma_gllvm(Yg; K = K)
    target_g = trigamma(fit_g.α)
    @test all(isapprox.(link_residual(fit_g, Yg), target_g; atol = 1e-9))

    Yβ = [rand(Beta(2.0, 3.0)) for t in 1:p, s in 1:n]
    fit_β = fit_beta_gllvm(Yβ; K = K)
    σ2_β = link_residual(fit_β, Yβ)
    @test all(σ2_β .> 0)

    # masked trait mean path
    mask = trues(p, n)
    mask[1, 1:20] .= false
    σ2_masked = link_residual(fit_p, Yp; mask = mask)
    @test length(σ2_masked) == p

    Σ = sigma_y_site(fit_p, Yp)
    @test size(Σ) == (p, p)
    @test Σ ≈ Σ' atol = 1e-12
    @test all(diag(Σ) .> diag(fit_p.Λ * fit_p.Λ'))

    c2 = communality(fit_p, Yp)
    @test length(c2) == p && all(0 .≤ c2 .≤ 1)

    R = correlation(fit_p, Yp)
    @test size(R) == (p, p)
    @test all(abs.(diag(R) .- 1) .< 1e-12)
    @test maximum(abs, R - R') < 1e-12
  end
end