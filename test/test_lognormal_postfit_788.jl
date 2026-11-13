using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra
const GM = GLLVModels

# #788: LognormalFit accepted `offset` at fit time but kept none and had no getLV / predict /
# residuals. It now stores the training offset (`fit.offset`) and has the three post-fit methods.
#
# Model: log y = beta + offset + Lambda z + sigma * eps. The log-scale model is Gaussian, so the
# conditional score of site s is the exact posterior mean z = Lambda' (Lambda Lambda' + sigma^2 I)^-1 r,
# r = log y - beta - offset, which is also the Laplace mode.
#
# Response scale: `predict(...; type = :response)` is exp(eta + sigma^2 / 2), the conditional mean,
# as gllvmTMB's predict(type = "response") returns for lognormal() (R/methods-gllvmTMB.R,
# .apply_linkinv_per_row). `type = :median` is exp(eta). The fit does not take a mask.

p, n, K = 5, 80, 1
rng = MersenneTwister(78803)
O = 0.5 .* randn(rng, p, n)
Λt = 0.7 .* randn(rng, p, K); Zt = randn(rng, n, K)
η0 = 1.0 .+ O .+ Λt * Zt'
Y = exp.(η0 .+ 0.4 .* randn(rng, p, n))

f  = fit_lognormal_gllvm(Y; K = K, offset = O)
f0 = fit_lognormal_gllvm(Y; K = K)

@testset "LognormalFit stored offset and post-fit methods (#788)" begin
    @test f isa LognormalFit
    @test f.offset == O
    @test f0.offset === nothing
    @test fit_gllvm(Y; family = Lognormal(), K = K, offset = O).offset == O

    # getLV: n x K, finite, default (stored) == explicit offset, differs from zero offset
    Z = getLV(f, Y; rotate = false)
    @test size(Z) == (n, K) && all(isfinite, Z)
    @test Z == getLV(f, Y; rotate = false, offset = O)
    @test maximum(abs, Z .- getLV(f, Y; rotate = false, offset = zeros(p, n))) > 1e-3
    @test size(getLV(f, Y)) == (n, K)
    @test getLV(f, Y) ≈ Z * GM._svd_rotation(f.Λ) atol = 1e-12

    # independent computation: posterior mean from the marginal covariance
    R = log.(Y) .- O .- f.β
    Σ = f.Λ * f.Λ' + f.σ^2 * I
    Zref = permutedims(f.Λ' * (Σ \ R))
    @test Z ≈ Zref atol = 1e-8
    # and it is the maximiser of the log-posterior of z at the site
    for s in (1, 11, 40)
        lp(z) = -sum(abs2, R[:, s] .- f.Λ * z) / (2 * f.σ^2) - sum(abs2, z) / 2
        z = Z[s, :]
        @test lp(z) >= lp(z .+ 0.05) && lp(z) >= lp(z .- 0.05)
    end

    # predict: link = beta + O + Lambda z by hand; response = exp(eta + sigma^2/2); median = exp(eta)
    η = predict(f, Y; type = :link)
    @test size(η) == (p, n)
    @test η ≈ f.β .+ O .+ f.Λ * Z' atol = 1e-10
    @test η == predict(f, Y; type = :link, offset = O)
    @test maximum(abs, η .- predict(f, Y; type = :link, offset = zeros(p, n))) > 1e-3
    @test predict(f, Y) ≈ exp.(η .+ f.σ^2 / 2) rtol = 1e-12
    @test predict(f, Y) ≈ GM.lognormal_response_mean.(η, f.σ) rtol = 1e-12
    @test predict(f, Y) == predict(f, Y; type = :response)
    @test predict(f, Y; type = :median) ≈ exp.(η) rtol = 1e-12
    @test predict(f, Y) ≈ exp(f.σ^2 / 2) .* predict(f, Y; type = :median) rtol = 1e-12
    @test_throws ArgumentError predict(f, Y; type = :mean)
    @test_throws ArgumentError predict(f, Y; type = :nonsense)

    # the fit's own marginal law agrees: E[y | z] at the training data is close to y on average
    @test abs(mean(log.(Y) .- η)) < 0.05

    # a different-size Y without an offset is refused; with one it works
    Ynew = Y[:, 1:10]
    @test_throws ArgumentError predict(f, Ynew)
    @test_throws ArgumentError getLV(f, Ynew)
    @test size(predict(f, Ynew; offset = O[:, 1:10])) == (p, 10)
    @test size(predict(f0, Ynew; type = :link)) == (p, 10)
    # a scalar or length-p offset is normalised as at fit time
    @test size(getLV(f, Ynew; offset = 0.2)) == (10, K)
    @test getLV(f, Ynew; offset = fill(0.2, p), rotate = false) ≈
          getLV(f, Ynew; offset = fill(0.2, p, 10), rotate = false) atol = 1e-12

    # modes = :training keeps the training units' modes at the stored offset
    ηt = predict(f, Y; type = :link, modes = :training, offset = O .+ 0.3)
    @test ηt ≈ f.β .+ (O .+ 0.3) .+ f.Λ * Z' atol = 1e-10
    @test_throws ArgumentError predict(f, Y; modes = :bad)

    # y must be positive
    Ybad = copy(Y); Ybad[2, 3] = 0.0
    @test_throws ArgumentError getLV(f, Ybad)
    @test_throws ArgumentError residuals(f, Ybad)
    # the fitter takes no mask, so neither do the post-fit methods
    @test_throws MethodError getLV(f, Y; mask = trues(p, n))
end

@testset "LognormalFit residuals (#788)" begin
    η = predict(f, Y; type = :link)
    R = residuals(f, Y)
    @test size(R) == (p, n) && all(isfinite, R)
    # Dunn-Smyth for a continuous law is deterministic: Phi^-1(F(y)) = (log y - eta) / sigma
    @test R ≈ (log.(Y) .- η) ./ f.σ atol = 1e-12
    @test R ≈ quantile.(Normal(), cdf.(LogNormal.(η, f.σ), Y)) atol = 1e-8
    @test R == residuals(f, Y; type = :dunnsmyth, offset = O)
    @test maximum(abs, R .- residuals(f, Y; offset = zeros(p, n))) > 1e-3
    # far tail stays finite (the plain CDF would round to 0 or 1)
    Yt = copy(Y); Yt[1, 1] = exp(η[1, 1] + 60 * f.σ)
    @test isfinite(residuals(f, Yt)[1, 1]) && residuals(f, Yt)[1, 1] > 10
    # pearson: response-scale mean and sd of the lognormal
    μ = predict(f, Y; type = :response)
    sd = sqrt.((exp(f.σ^2) - 1) .* exp.(2 .* η .+ f.σ^2))
    @test residuals(f, Y; type = :pearson) ≈ (Y .- μ) ./ sd rtol = 1e-10
    @test_throws ArgumentError residuals(f, Y; type = :deviance)
end

@testset "LognormalFit through extract_latent_scores / ordination (#788)" begin
    z = extract_latent_scores(f, Y)
    @test z == getLV(f, Y; rotate = false)
    @test extract_latent_scores(f, Y; offset = O) == z
    @test extract_latent_scores(f, Y; level = :unit_obs) === nothing
    @test rotation(f) isa AbstractMatrix
    @test getLoadings(f; rotate = false) == f.Λ
    @test isfinite(aic(f)) && isfinite(loglikelihood(f))
end

@testset "LognormalFit is in exactly one latent-score dispatch union (#788)" begin
    unions = (GM._ComponentAwareGllvmFit, GM._PlainGllvmFit, GM._PositionalArgGllvmFit)
    @test count(U -> LognormalFit <: U, unions) == 1
    @test LognormalFit <: GM._PlainGllvmFit
end

@testset "LognormalFit positional constructor without offset still works (#788)" begin
    g = LognormalFit(f.β, f.Λ, f.σ, f.link, f.loglik, f.converged, f.iterations, f.theta_packed)
    @test g.offset === nothing
end
