using GLLVModels, Test, Random, Distributions

@testset "vcov(fit, Y) on GllvmCovFit without X (refs #1555)" begin
    Random.seed!(1555)
    p, K, n = 4, 1, 30
    β = 0.2 .* randn(p)
    γ = [0.5]
    Λ = 0.3 .* randn(p, K)
    x = randn(n)
    X = zeros(p, n, 1)
    @inbounds for t in 1:p, s in 1:n
        X[t, s, 1] = x[s]
    end
    Z = randn(K, n)
    η = β .+ γ[1] .* x' .+ Λ * Z
    Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
    fit = fit_gllvm_cov(Y; family = Poisson(), X = X, K = K, iterations = 40)

    err = try
        vcov(fit, Y)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("vcov", err.msg)
    @test occursin("X=X", err.msg)
    @test !occursin("confint", err.msg)

    V = vcov(fit, Y; X = X)
    @test size(V) == (length(fit.β) + count(.!fit.γ_fixed) + div(p * K * (K + 1), 2),) ||
          size(V, 1) == size(V, 2)
    @test all(isfinite, V)
end