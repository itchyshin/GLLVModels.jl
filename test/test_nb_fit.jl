using GLLVModels, Test, Random, LinearAlgebra, Statistics, Distributions

@testset "fit_nb_gllvm — recovery" begin
    @testset "recovers loading structure + intercepts; sane dispersion" begin
        Random.seed!(70)
        p, K, n = 6, 2, 400
        β_true = log.([4.0, 6.0, 3.0, 5.0, 4.0, 7.0])
        Λ_true = 0.5 .* randn(p, K)
        r_true = 5.0
        μ = exp.(β_true .+ Λ_true * randn(K, n))
        Y = [rand(NegativeBinomial(r_true, r_true / (r_true + μ[t, s]))) for t in 1:p, s in 1:n]

        fit = fit_nb_gllvm(Y; K = K)
        @test fit.converged
        @test size(fit.Λ) == (p, K)
        @test maximum(abs.(fit.β .- β_true)) < 0.5
        @test cor(vec(fit.Λ * fit.Λ'), vec(Λ_true * Λ_true')) > 0.6
        # dispersion estimation is identifiable-but-noisy (latent absorbs some
        # overdispersion); assert it stays finite and positive.
        @test isfinite(fit.r) && fit.r > 0
    end

    @testset "fit_gllvm(family = NegativeBinomial(), disp_group = :species) dispatches to NBGroupedFit" begin
        Random.seed!(71)
        p, K, n = 5, 1, 200
        β = log.(fill(5.0, p))
        Y = [rand(NegativeBinomial(8.0, 8.0 / (8.0 + exp(β[t] + 0.4 * randn()))))
             for t in 1:p, s in 1:n]
        fit = fit_gllvm(Y; family = NegativeBinomial(), K = K, disp_group = :species)
        @test fit isa NBGroupedFit
        @test fit_gllvm(Y; family = NegativeBinomial(), K = K) isa NBFit   # shared r default (#615)
        @test length(fit.r_group) == p
        @test fit.converged
    end
end
