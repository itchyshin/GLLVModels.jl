# Lane cursor-cloud-1125y10: confint_bootstrap.jl internal helpers (mean, site cov, simulate, warn).
using Test, GLLVModels, LinearAlgebra, Random, Statistics

const G = GLLVModels

@testset "1125y10 confint_bootstrap helpers" begin
    @testset "_bootstrap_lambda_term_names packing order" begin
        names = G._bootstrap_lambda_term_names("Lambda_B", 4, 2)
        @test names == ["Lambda_B[1,1]", "Lambda_B[2,2]",
            "Lambda_B[2,1]", "Lambda_B[3,1]", "Lambda_B[4,1]",
            "Lambda_B[3,2]", "Lambda_B[4,2]"]
        one = G._bootstrap_lambda_term_names("Λ", 3, 1)
        @test one == ["Λ[1,1]", "Λ[2,1]", "Λ[3,1]"]
        k3 = G._bootstrap_lambda_term_names("L", 3, 3)
        @test length(k3) == G.rr_theta_len(3, 3)
        @test k3[1] == "L[1,1]" && k3[2] == "L[2,2]" && k3[3] == "L[3,3]"
    end

    rng = MersenneTwister(1125)
    p, K, n = 5, 2, 40
    Λ = 0.6 .* randn(rng, p, K)
    y = Λ * randn(rng, K, n) + 0.4 * randn(rng, p, n)
    fit0 = fit_gaussian_gllvm(y; K = K)

    @testset "_bootstrap_term_names vs confint" begin
        @test G._bootstrap_term_names(fit0) == G._confint_all_term_names(fit0)[1]
        X = randn(rng, p, n, 1)
        yx = y + reshape(X[:, :, 1], p, n)
        fx = fit_gaussian_gllvm(yx; K = K, X = X)
        bt = G._bootstrap_term_names(fx)
        @test "beta[1]" in bt
        @test startswith(bt[end], "Lambda_B[")
    end

    @testset "_bootstrap_mean_from_X" begin
        p2, n2, q = 3, 5, 2
        β = [1.0, -0.5]
        X = zeros(p2, n2, q)
        for s in 1:n2, t in 1:p2, k in 1:q
            X[t, s, k] = 0.1 * (t + k + s)
        end
        model = GllvmModel(p2, 1)
        pars = (σ_eps = 0.5, Λ = ones(p2, 1) * 0.2, β = β, Λ_W = nothing,
            σ²_B = nothing, σ²_W = nothing, Λ_phy = nothing, σ_phy = nothing,
            θ_packed = Float64[])
        fitβ = GllvmFit(model, pars, -1.0, 1, true, nothing, 0.0)
        μ = G._bootstrap_mean_from_X(fitβ, X)
        @test size(μ) == (p2, n2)
        for s in 1:n2, t in 1:p2
            expected = sum(X[t, s, k] * β[k] for k in 1:q)
            @test μ[t, s] ≈ expected rtol = 1e-14
        end
        μ0 = G._bootstrap_mean_from_X(
            GllvmFit(GllvmModel(p2, 1),
                merge(pars, (β = nothing,)),
                -1.0, 1, true, nothing, 0.0),
            X)
        @test μ0 == zeros(p2, n2)
    end

    @testset "_bootstrap_site_cov structure" begin
        A = G._bootstrap_site_cov(fit0)
        @test size(A) == (p, p)
        @test issymmetric(A)
        ΛB = fit0.pars.Λ
        @test A ≈ ΛB * ΛB' + fit0.pars.σ_eps^2 * I rtol = 1e-12
        rng2 = MersenneTwister(9)
        yw = Λ * randn(rng2, K, n) + 0.3 * randn(rng2, p, n)
        fw = fit_gaussian_gllvm(yw; K = K, K_W = 1, has_diag = true)
        Aw = G._bootstrap_site_cov(fw)
        @test issymmetric(Aw)
        @test diag(Aw) |> minimum > 0
        @test tr(Aw) > tr(ΛB * ΛB')
    end

    @testset "_bootstrap_simulate! Gaussian draws" begin
        p3, n3 = 4, 20
        μ = fill(0.5, p3, n3)
        d = fill(0.25, p3)
        Λs = [0.9 0.0; 0.1 0.8; 0.0 0.0; 0.2 0.3]
        A = Λs * Λs' + Diagonal(d)
        L = cholesky(Symmetric(A)).L
        ybuf = zeros(p3, n3)
        r1 = MersenneTwister(1)
        G._bootstrap_simulate!(r1, ybuf, μ, L, nothing, nothing)
        @test all(isfinite, ybuf)
        @test std(ybuf) > 0
        @test isapprox(mean(ybuf), mean(μ), atol = 1.0)
        Lp = cholesky(Symmetric([1.0 0.3; 0.3 1.0])).L
        Λphy = [0.5 0.0; 0.0 0.4]
        y0 = zeros(2, n3)
        L2 = cholesky(Symmetric([1.2 0.1; 0.1 0.9])).L
        G._bootstrap_simulate!(r1, y0, zeros(2, n3), L2, Lp, Λphy)
        @test all(isfinite, y0)
        @test size(y0) == (2, n3)
        μ3 = fill(3.0, 2, n3)
        G._bootstrap_simulate!(MersenneTwister(99), y0, μ3, L2, nothing, nothing)
        @test mean(y0) > 2.0
        @test mean(y0) < 5.0
    end

    @testset "_bootstrap_warn_dropped threshold" begin
        @test G._bootstrap_warn_dropped("test", 0, 10) === nothing
        @test G._bootstrap_warn_dropped("test", 5, 10) === nothing
        @test G._bootstrap_warn_dropped("x", 4, 10) === nothing
        @test_logs (:warn,) G._bootstrap_warn_dropped("bootstrap_ci", 6, 10)
        @test_logs (:warn,) G._bootstrap_warn_dropped("bootstrap_ci", 11, 20)
    end
end