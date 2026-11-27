# Targeted coverage for src/[p-z]*.jl (Cursor cloud 1127cov lane).
# Exercises user-facing simulate / latent_score_sd / SPDE postfit paths and
# internal helpers documented in source docstrings — not parity oracles.

using GLLVModels, Test, Random, Distributions, LinearAlgebra, SparseArrays

# ---- simulate_fit.jl (scalar μ families + X_lv guards) -----------------------

@testset "simulate(fit) coverage: Exponential and X_lv guards" begin
    @testset "ExponentialFit: positive draws and reproducible rng" begin
        Random.seed!(112701)
        p, K, n = 4, 1, 80
        Y = [rand(Exponential(1.0)) * exp(0.1 * randn()) for t in 1:p, s in 1:n]
        fit = fit_exponential_gllvm(Y; K = K, iterations = 40)
        Ys = simulate(fit, 35; rng = MersenneTwister(1127))
        @test size(Ys) == (p, 35)
        @test eltype(Ys) == Float64 && all(Ys .> 0)
        @test simulate(fit, 12; rng = MersenneTwister(9)) ==
              simulate(fit, 12; rng = MersenneTwister(9))
    end

    @testset "NBFit simulate: refuses missing X_lv when alpha_lv was fit" begin
        Random.seed!(112702)
        p, n, K = 4, 50, 1
        X_lv = randn(n, 2)
        Y = [rand(Poisson(3)) for _ in 1:p, _ in 1:n]
        fit = fit_nb_gllvm(Y; K = K, X_lv = X_lv, iterations = 25)
        @test fit.alpha_lv !== nothing
        @test_throws ArgumentError simulate(fit, 10)
        X_bad = randn(9, size(X_lv, 2))
        @test_throws ArgumentError simulate(fit, 10; X_lv = X_bad)
        Ys = simulate(fit, 10; X_lv = X_lv[1:10, :], rng = MersenneTwister(3))
        @test size(Ys) == (p, 10) && eltype(Ys) <: Integer
    end

    @testset "OrdinalFit simulate: category range" begin
        Random.seed!(112703)
        p, n, K = 3, 40, 1
        Y = [rand(1:4) for _ in 1:p, _ in 1:n]
        fit = fit_ordinal_gllvm(Y; K = K, iterations = 30)
        Ys = simulate(fit, 15; rng = MersenneTwister(5))
        @test size(Ys) == (p, 15)
        @test eltype(Ys) == Int
        @test all(1 .<= Ys .<= fit.C)
    end
end

# ---- re_sd.jl ----------------------------------------------------------------

@testset "latent_score_sd coverage: families and guards" begin
    @testset "NBFit / GammaFit / BetaFit return finite n×K SDs" begin
        rng = MersenneTwister(112704)
        p, n, K = 4, 45, 1
        Λ0 = 0.4 .* randn(rng, p, K)
        β0 = 0.1 .* randn(rng, p)
        z = randn(rng, n, K)
        η = β0 .+ Λ0 * z'

        Ynb = [rand(rng, NegativeBinomial(2.5, clamp(exp(η[t, s]) / (2.5 + exp(η[t, s])), 1e-6, 1 - 1e-6)))
               for t in 1:p, s in 1:n]
        fit_nb = fit_nb_gllvm(Ynb; K = K, iterations = 35)
        sd_nb = latent_score_sd(fit_nb, Ynb)
        @test size(sd_nb) == (n, K) && all(isfinite, sd_nb)

        μg = exp.(η)
        Yg = [rand(rng, Gamma(3.0, μg[t, s] / 3.0)) for t in 1:p, s in 1:n]
        fit_g = fit_gamma_gllvm(Yg; K = K, iterations = 35)
        sd_g = latent_score_sd(fit_g, Yg)
        @test size(sd_g) == (n, K) && all(isfinite, sd_g)

        Yb = [clamp(rand(rng, Beta(2.0, 2.0)), 1e-4, 1 - 1e-4) for t in 1:p, s in 1:n]
        fit_b = fit_beta_gllvm(Yb; K = K, iterations = 35)
        sd_b = latent_score_sd(fit_b, Yb)
        @test size(sd_b) == (n, K) && all(isfinite, sd_b)
    end

    @testset "Binomial: integer response guard and probit observed curvature path" begin
        rng = MersenneTwister(112705)
        p, n, K = 4, 35, 1
        Y = rand(0:1, p, n)
        fit_f = fit_binomial_gllvm(Y; K = K, link = LogitLink(), iterations = 25,
                                   hessian = :fisher)
        Yfloat = float.(Y)
        @test_throws ArgumentError latent_score_sd(fit_f, Yfloat)

        fit_p = fit_binomial_gllvm(Y; K = K, link = ProbitLink(), iterations = 25,
                                   hessian = :observed)
        @test fit_p.hessian === :observed
        sd_p = latent_score_sd(fit_p, Y)
        @test size(sd_p) == (n, K) && all(isfinite, sd_p)
    end

    @testset "Poisson mask zeroes weight on masked cells" begin
        rng = MersenneTwister(112706)
        p, n, K = 3, 30, 1
        Y = [rand(rng, Poisson(2.0)) for _ in 1:p, _ in 1:n]
        fit = fit_poisson_gllvm(Y; K = K, iterations = 30)
        mask = trues(p, n)
        mask[1, 1] = false
        sd = latent_score_sd(fit, Y; mask = mask)
        @test size(sd) == (n, K) && all(isfinite, sd)
    end
end

# ---- spde_latent_postfit.jl --------------------------------------------------

@testset "SPDE-latent postfit: trial matrix branches and accessors" begin
    Random.seed!(112707)
    m, L = 5, 4.0
    xs = range(0.0, L; length = m)
    nodes = Matrix{Float64}(undef, m^2, 2)
    for j in 1:m, i in 1:m
        nodes[(j - 1) * m + i, :] = [xs[i], xs[j]]
    end
    tris = Matrix{Int}(undef, 2 * (m - 1)^2, 3)
    t = 0
    for j in 1:(m - 1), i in 1:(m - 1)
        a = (j - 1) * m + i
        b = a + 1; c = a + m; d = c + 1
        t += 1; tris[t, :] = [a, b, d]
        t += 1; tris[t, :] = [a, d, c]
    end
    locs = nodes[1:3:end, :]
    p, M, K = 2, size(locs, 1), 1
    Y = rand(0:3, p, M)
    fit = fit_spde_latent_gllvm(Y, nodes, tris, locs; family = Poisson(), K = K,
                                iterations = 25, newton_maxiter = 30)

    @testset "getLoadings for SPDELatentFit" begin
        Ld = getLoadings(fit; rotate = false)
        @test size(Ld) == (p, K)
        @test isapprox(Ld, fit.Λ; rtol = 1e-10, atol = 1e-12)
    end

    @testset "empty stored N falls back to ones(p, M)" begin
        fit_emptyN = SPDELatentFit(
            fit.β, fit.Λ, fit.κ, fit.τ, fit.dispersion, fit.link, fit.family,
            fit.loglik, fit.converged, fit.iterations, fit.nodes, fit.tris)
        @test isempty(fit_emptyN.N)
        Z = getLV(fit_emptyN, Y, locs)
        @test size(Z) == (M, K) && all(isfinite, Z)
    end

    @testset "mismatched stored N is refused" begin
        Nbad = ones(p, M + 1)
        fit_bad = SPDELatentFit(
            fit.β, fit.Λ, fit.κ, fit.τ, fit.dispersion, fit.link, fit.family,
            fit.loglik, fit.converged, fit.iterations, fit.nodes, fit.tris, Nbad, fit.α)
        @test_throws ArgumentError predict(fit_bad, Y, locs)
    end

    @testset "invalid predict type" begin
        @test_throws ArgumentError predict(fit, Y, locs; type = :mean)
        @test_throws ArgumentError predict_spatial(fit, Y, locs, locs; type = :mean)
    end
end

# ---- takahashi_selinv.jl (_csc_rowidx) ---------------------------------------

@testset "_csc_rowidx binary search" begin
    S = sparse([1.0 0.0; 2.0 3.0; 0.0 4.0])
    colptr = S.colptr
    rowval = S.rowval
    @test GLLVModels._csc_rowidx(colptr, rowval, 1, 1) > 0
    @test GLLVModels._csc_rowidx(colptr, rowval, 1, 3) == -1
    @test GLLVModels._csc_rowidx(colptr, rowval, 2, 3) > 0
    @test GLLVModels._csc_rowidx(colptr, rowval, 2, 1) == -1
end