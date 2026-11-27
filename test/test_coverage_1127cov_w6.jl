# Coverage lane w6 (27 Nov): validation / contract branches in quadratic, tweedie,
# and gamma family modules (refs coverage).
using GLLVModels, Test, Random, Distributions, LinearAlgebra

@testset "coverage 1127cov w6" begin

    @testset "quadratic — hessian and predict type guards" begin
        p, K, n = 3, 1, 4
        Y = [rand(Poisson(2)) for _ in 1:p, _ in 1:n]
        β = vec(log.(mean(Y; dims = 2)))
        Λ = fill(0.05, p, K)
        D = zeros(p, K)
        N = ones(Int, p, n)
        @test_throws ArgumentError GLLVModels.quadratic_loglik_site(
            Poisson(), Y[:, 1], N[:, 1], Λ, D, β, LogLink(); hessian = :expected)
        fit = fit_quadratic_gllvm(Y; family = Poisson(), K = K, iterations = 40)
        @test_throws ArgumentError predict(fit, Y; type = :mean)
    end

    @testset "tweedie — hessian refusal" begin
        Y = abs.(randn(3, 10)) .+ 0.2
        @test_throws ArgumentError fit_tweedie_gllvm(Y; K = 1, hessian = :watanabe, iterations = 5)
    end

    @testset "gamma — X_lv packed objective and fit guards" begin
        p, K, n, q = 3, 1, 5, 1
        Y = abs.(randn(p, n)) .+ 0.5
        X = reshape(collect(1.0:n), n, q)
        rr = GLLVModels.rr_theta_len(p, K)
        npar = p + q * K + rr + 1
        θ = ones(npar)
        @test_throws ArgumentError GLLVModels.gamma_lv_nll_packed(
            θ[1:(end - 1)], Y, p, K, LogLink(); X_lv = X, q_lv = q)
        Xbad = reshape(collect(1.0:(n + 1)), n + 1, q)
        @test_throws ArgumentError GLLVModels.gamma_lv_nll_packed(
            θ, Y, p, K, LogLink(); X_lv = Xbad, q_lv = q)
        @test_throws ArgumentError fit_gamma_gllvm(Y; K = 0, X_lv = X, iterations = 5)
        @test_throws ArgumentError fit_gamma_gllvm(Y; K = K, alpha_lv_init = zeros(q, K))
        @test_throws ArgumentError fit_gamma_gllvm(
            Y; K = K, X_lv = X, hessian = :fisher, iterations = 5)
        @test_throws ArgumentError fit_gamma_gllvm(
            Y; K = K, X_lv = X, alpha_lv_init = zeros(q + 1, K), iterations = 5)
    end
end