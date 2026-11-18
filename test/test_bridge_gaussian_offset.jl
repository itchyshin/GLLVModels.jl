# bridge_fit `offset` for the Gaussian family (#912): the bridge forwards a known
# additive offset to fit_gaussian_gllvm on the plain no-X route and the
# fixed-effect X route, and refuses it everywhere it is not wired.
using Test
using GLLVModels
using Random
using Statistics

@testset "bridge Gaussian offset (#912)" begin
    Random.seed!(912)
    p, n, K = 4, 40, 1
    Λ = reshape([1.0, 0.8, -0.6, 0.5], p, 1)
    Z = randn(1, n)
    O = 3.0 .* randn(p, n) .+ [2.0, -1.0, 0.5, 4.0]   # large: ignoring it changes the fit
    Y = [0.5, -0.3, 1.2, 0.0] .+ O .+ Λ * Z .+ 0.5 .* randn(p, n)
    alpha_ref = vec(mean(Y .- O; dims = 2))

    @testset "no-X: parity with the direct offset fit" begin
        Yc = Y .- alpha_ref
        direct = fit_gaussian_gllvm(Yc; K = K, offset = O)
        br = bridge_fit(; y = Y, family = "gaussian", d = K, offset = O)
        @test br.family == "gaussian"
        @test isapprox(br.alpha, alpha_ref; atol = 1e-10, rtol = 0)
        @test isapprox(br.loglik, direct.logLik; atol = 1e-6, rtol = 0)
        @test br.converged == direct.converged
        @test isapprox(br.loadings, getLoadings(direct; rotate = true); atol = 1e-6, rtol = 0)
        @test isapprox(br.sigma_eps, direct.pars.σ_eps; atol = 1e-6, rtol = 0)

        # Removing the offset by hand and fitting without one is the same model.
        br_manual = bridge_fit(; y = Y .- O, family = "gaussian", d = K)
        @test isapprox(br.loglik, br_manual.loglik; atol = 1e-5, rtol = 0)
        @test isapprox(br.alpha, br_manual.alpha; atol = 1e-10, rtol = 0)

        # The offset is not silently dropped.
        br_none = bridge_fit(; y = Y, family = "gaussian", d = K)
        @test !isapprox(br_none.alpha, br.alpha; atol = 0.5, rtol = 0)

        # A zero offset is the no-offset fit.
        br_zero = bridge_fit(; y = Y, family = "gaussian", d = K, offset = zeros(p, n))
        @test isapprox(br_zero.loglik, br_none.loglik; atol = 1e-6, rtol = 0)
        @test isapprox(br_zero.alpha, br_none.alpha; atol = 1e-10, rtol = 0)

        # No gradient is reported for an offset fit (the reconstruction lacks it).
        @test isnan(br.gradient_max)
    end

    @testset "offset shapes: per-trait vector and scalar" begin
        ov = [2.0, -1.0, 0.5, 4.0]
        br_vec = bridge_fit(; y = Y, family = "gaussian", d = K, offset = ov)
        br_mat = bridge_fit(; y = Y, family = "gaussian", d = K,
                            offset = repeat(ov, 1, n))
        @test isapprox(br_vec.loglik, br_mat.loglik; atol = 1e-8, rtol = 0)
        @test isapprox(br_vec.alpha, br_mat.alpha; atol = 1e-10, rtol = 0)
        br_s = bridge_fit(; y = Y, family = "gaussian", d = K, offset = 2.0)
        @test isapprox(br_s.alpha, vec(mean(Y .- 2.0; dims = 2)); atol = 1e-10, rtol = 0)
        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = K,
                                              offset = zeros(p + 1, n))
    end

    @testset "mask plus offset" begin
        M = trues(p, n)
        M[1, 3] = false; M[2, 11] = false; M[4, 30] = false
        alpha_obs = [sum(Y[t, s] - O[t, s] for s in 1:n if M[t, s]) / count(M[t, :])
                     for t in 1:p]
        Yc = Y .- alpha_obs
        Yc[.!M] .= 0.0
        direct = fit_gaussian_gllvm(Yc; K = K, mask = M, offset = O)
        br = bridge_fit(; y = Y, family = "gaussian", d = K, mask = M, offset = O)
        @test br.nobs == count(M)
        @test isapprox(br.alpha, alpha_obs; atol = 1e-10, rtol = 0)
        @test isapprox(br.loglik, direct.logLik; atol = 1e-6, rtol = 0)
        @test br.converged == direct.converged
    end

    @testset "fixed-effect X route with offset" begin
        x = randn(n)
        X = zeros(p, n, p + 1)
        for t in 1:p, s in 1:n
            X[t, s, t] = 1.0
            X[t, s, p + 1] = x[s]
        end
        Yx = [0.5, -0.3, 1.2, 0.0] .+ O .+ 0.7 .* reshape(x, 1, n) .+
             Λ * Z .+ 0.5 .* randn(p, n)
        direct = fit_gaussian_gllvm(Yx; K = K, X = X, offset = O)
        br = bridge_fit(; y = Yx, family = "gaussian", d = K, X = X, offset = O)
        @test isapprox(br.loglik, direct.logLik; atol = 1e-6, rtol = 0)
        @test br.converged == direct.converged
        @test isapprox(br.mean_coef, collect(direct.pars.β); atol = 1e-6, rtol = 0)
        @test abs(br.mean_coef[end] - 0.7) < 0.5
        @test isnan(br.gradient_max)
    end

    @testset "unsupported offset routes fail loudly" begin
        Yp = Float64.(rand(0:4, p, n))
        @test_throws ArgumentError bridge_fit(; y = Yp, family = "poisson", d = K,
                                              offset = zeros(p, n))
        @test_throws ArgumentError bridge_fit(; y = Y, family = ["gaussian", "gaussian",
                                              "gaussian", "gaussian"], d = K,
                                              offset = zeros(p, n))
        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = K,
                                              offset = O, X_lv = randn(n, 1))
        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = K,
                                              offset = O, options = Dict("reml" => true))
        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = K,
                                              offset = O,
                                              options = Dict("ci_method" => "wald"))
        @test_throws ArgumentError bridge_fit(; y = Y, family = nothing, d = K, offset = O)
    end
end
