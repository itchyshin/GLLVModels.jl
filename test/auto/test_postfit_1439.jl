using GLLVModels, Test, Random, Distributions

function _Xcov(p::Int, n::Int)
    x = randn(n)
    X = zeros(p, n, 1)
    X[:, :, 1] .= reshape(x, 1, n)
    return X
end

@testset "GllvmCovFit keyword X (#1439)" begin
    Random.seed!(1911)
    p, K, n = 5, 1, 60
    X = _Xcov(p, n)
    x = vec(X[1, :, 1])
    Y = [rand(Poisson(exp(0.2 + 0.4 * x[s]))) for t in 1:p, s in 1:n]
    fit = fit_gllvm_cov(Y; family = Poisson(), X = X, K = K, iterations = 80)
    Pkw = predict(fit, Y; X = X, type = :response)
    Ppos = predict(fit, Y, X; type = :response)
    @test Pkw ≈ Ppos
    R = residuals(fit, Y; X = X)
    @test size(R) == size(Y)
    ord = ordination(fit, Y; X = X)
    @test size(ord.sites, 1) == n
end

@testset "GllvmCovFit Pearson residuals (#1439)" begin
    p, K, n = 4, 1, 55

    @testset "Poisson" begin
        Random.seed!(143901)
        X = _Xcov(p, n)
        x = vec(X[1, :, 1])
        Y = [rand(Poisson(exp(0.1 + 0.35 * x[s]))) for t in 1:p, s in 1:n]
        fit = fit_gllvm_cov(Y; family = Poisson(), X = X, K = K, iterations = 90)
        μ = predict(fit, Y; X = X, type = :response)
        @test residuals(fit, Y; X = X, type = :pearson) ≈ (Y .- μ) ./ sqrt.(μ) atol = 1e-10
    end

    @testset "NegativeBinomial" begin
        Random.seed!(143902)
        X = _Xcov(p, n)
        x = vec(X[1, :, 1])
        Y = [rand(NegativeBinomial(5.0, 5.0 / (5.0 + exp(0.2 + 0.3 * x[s])))) for t in 1:p, s in 1:n]
        fit = fit_gllvm_cov(Y; family = NegativeBinomial(), X = X, K = K, iterations = 90)
        μ = predict(fit, Y; X = X, type = :response)
        r = fit.dispersion
        @test residuals(fit, Y; X = X, type = :pearson) ≈ (Y .- μ) ./ sqrt.(μ .+ μ .^ 2 ./ r) atol = 1e-9
    end

    @testset "NB1" begin
        Random.seed!(143903)
        X = _Xcov(p, n)
        x = vec(X[1, :, 1])
        φ = 0.8
        Y = [begin
            μ = exp(0.15 + 0.25 * x[s])
            rand(NegativeBinomial(μ / φ, 1 / (1 + φ)))
        end for t in 1:p, s in 1:n]
        fit = fit_gllvm_cov(Y; family = GLLVModels.NB1(1.0), X = X, K = K, iterations = 120)
        μ = predict(fit, Y; X = X, type = :response)
        φ̂ = fit.dispersion
        @test residuals(fit, Y; X = X, type = :pearson) ≈ (Y .- μ) ./ sqrt.(μ .* (1 .+ φ̂)) atol = 1e-9
    end

    @testset "Binomial" begin
        Random.seed!(143904)
        X = _Xcov(p, n)
        x = vec(X[1, :, 1])
        prob = inv.(1 .+ exp.(-(0.1 .+ 0.4 .* x')))
        Nm = fill(10, p, n)
        Y = [rand(Binomial(Nm[t, s], prob[s])) for t in 1:p, s in 1:n]
        fit = fit_gllvm_cov(Y; family = Binomial(), X = X, K = K, N = Nm, iterations = 90)
        μ = predict(fit, Y; X = X, type = :response, N = Nm)
        @test residuals(fit, Y; X = X, type = :pearson, N = Nm) ≈
              (Y .- Nm .* μ) ./ sqrt.(Nm .* μ .* (1 .- μ)) atol = 1e-10
    end

    @testset "Beta" begin
        Random.seed!(143905)
        X = _Xcov(p, n)
        x = vec(X[1, :, 1])
        φ = 12.0
        Y = [begin
            μ = inv(1 + exp(-(0.1 + 0.35 * x[s])))
            rand(Beta(μ * φ, (1 - μ) * φ))
        end for t in 1:p, s in 1:n]
        fit = fit_gllvm_cov(Y; family = Beta(), X = X, K = K, iterations = 120)
        μ = predict(fit, Y; X = X, type = :response)
        φ̂ = fit.dispersion
        @test residuals(fit, Y; X = X, type = :pearson) ≈ (Y .- μ) ./ sqrt.(μ .* (1 .- μ) ./ (1 .+ φ̂)) atol = 1e-9
    end

    @testset "Gamma" begin
        Random.seed!(143906)
        X = _Xcov(p, n)
        x = vec(X[1, :, 1])
        α = 4.0
        Y = [begin
            μ = exp(0.3 + 0.2 * x[s])
            rand(Gamma(α, μ / α))
        end for t in 1:p, s in 1:n]
        fit = fit_gllvm_cov(Y; family = Gamma(), X = X, K = K, iterations = 120)
        μ = predict(fit, Y; X = X, type = :response)
        α̂ = fit.dispersion
        @test residuals(fit, Y; X = X, type = :pearson) ≈ (Y .- μ) ./ sqrt.(μ .^ 2 ./ α̂) atol = 1e-9
    end

    @testset "Exponential" begin
        Random.seed!(143907)
        X = _Xcov(p, n)
        x = vec(X[1, :, 1])
        Y = [rand(Exponential(exp(0.1 + 0.25 * x[s]))) for t in 1:p, s in 1:n]
        fit = fit_gllvm_cov(Y; family = Exponential(), X = X, K = K, iterations = 90)
        μ = predict(fit, Y; X = X, type = :response)
        @test residuals(fit, Y; X = X, type = :pearson) ≈ (Y .- μ) ./ μ atol = 1e-10
    end
end