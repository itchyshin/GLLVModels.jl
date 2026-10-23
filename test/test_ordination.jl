using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra

@testset "ordination — Poisson GLLVModels" begin
    Random.seed!(2026)
    p, K, n = 5, 2, 100
    β_true = log.([4.0, 6.0, 3.0, 5.0, 4.0])
    Λ_true = 0.5 .* randn(p, K)
    η = β_true .+ Λ_true * randn(K, n)
    Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]

    fit = fit_poisson_gllvm(Y; K = K)
    @test fit.converged

    o = ordination(fit, Y)

    @testset "shapes" begin
        @test size(o.sites) == (n, K)
        @test size(o.species) == (p, K)
        @test size(o.rotation) == (K, K)
    end

    @testset "rotation orthogonality" begin
        @test o.rotation' * o.rotation ≈ I atol = 1e-10
    end

    @testset "reconstruction invariance" begin
        S0 = getLV(fit, Y; rotate = false)
        @test o.sites * o.species' ≈ S0 * fit.Λ' atol = 1e-8
    end

    @testset "rotate=false is identity / raw" begin
        o0 = ordination(fit, Y; rotate = false)
        @test o0.sites == getLV(fit, Y; rotate = false)
        @test o0.species == fit.Λ
        @test o0.rotation ≈ I
    end
end

@testset "ordiplot data layer" begin
    Random.seed!(11)
    p, K, n = 5, 2, 80
    β_true = log.([4.0, 6.0, 3.0, 5.0, 4.0])
    Λ_true = 0.5 .* randn(p, K)
    η = β_true .+ Λ_true * randn(K, n)
    Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]

    fit = fit_poisson_gllvm(Y; K = K)
    @test fit.converged

    o = ordiplot(fit, Y)

    @testset "shapes & equivalence" begin
        @test size(o.sites) == (n, K)
        @test size(o.species) == (p, K)
        @test o.sites ≈ ordination(fit, Y).sites atol = 1e-10
    end

    @testset "axis_prop" begin
        @test length(o.axis_prop) == K
        @test all(o.axis_prop .>= 0)
        @test sum(o.axis_prop) ≈ 1
    end

    @testset "default labels" begin
        @test length(o.site_labels) == n
        @test o.site_labels[1] == "site 1"
        @test o.site_labels[2] == "site 2"
        @test length(o.species_labels) == p
        @test o.species_labels[1] == "sp 1"
        @test o.species_labels[2] == "sp 2"
    end

    @testset "biplot=false drops species" begin
        o2 = ordiplot(fit, Y; biplot = false)
        @test size(o2.species) == (0, K)
        @test isempty(o2.species_labels)
        @test size(o2.sites) == (n, K)
    end
end

@testset "ordination forwards the fit's design to getLV" begin
    # Regression: ordination/extract_ordination/ordiplot called
    # getLV(fit, Y; rotate=false) with no way to pass X, so a fit with fixed
    # effects got site scores computed at a zero mean (silently wrong).
    Random.seed!(7)
    p, K, n = 6, 2, 80
    x = randn(n)
    X = zeros(p, n, 2); X[:, :, 1] .= 1.0; X[:, :, 2] .= x'
    Y = 3.0 .+ 2.0 .* x' .+ 0.6 .* randn(p, K) * randn(K, n) .+ 0.3 .* randn(p, n)
    fit = fit_gaussian_gllvm(Y; K = K, X = X)
    @test fit.converged
    S = getLV(fit, Y; rotate = false, X = X)

    @testset "Gaussian X: scores match getLV(...; X)" begin
        @test ordination(fit, Y; rotate = false, X = X).sites ≈ S atol = 1e-10
        @test extract_ordination(fit, Y; rotate = false, X = X).sites ≈ S atol = 1e-10
        o = ordination(fit, Y; X = X)
        @test o.sites * o.species' ≈ S * fit.pars.Λ' atol = 1e-8
        @test ordiplot(fit, Y; X = X).sites ≈ o.sites atol = 1e-10
    end

    @testset "Gaussian X omitted: loud error, not a zero-mean answer" begin
        @test_throws ArgumentError ordination(fit, Y)
        @test_throws ArgumentError extract_ordination(fit, Y; rotate = false)
        @test_throws ArgumentError ordiplot(fit, Y)
    end

    @testset "Binomial trials N reach getLV" begin
        Random.seed!(8)
        pb, nb = 5, 60
        N = rand(5:15, pb, nb)
        η = 0.3 .+ 0.7 .* randn(pb, K) * randn(K, nb)
        Yb = [rand(Binomial(N[t, s], 1 / (1 + exp(-η[t, s])))) for t in 1:pb, s in 1:nb]
        fb = fit_binomial_gllvm(Yb; K = K, N = N)
        Sb = getLV(fb, Yb; N = N, rotate = false)
        @test ordination(fb, Yb; N = N, rotate = false).sites ≈ Sb atol = 1e-10
        @test extract_ordination(fb, Yb; N = N, rotate = false).sites ≈ Sb atol = 1e-10
    end
end
