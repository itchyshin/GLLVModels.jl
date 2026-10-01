using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GMT = GLLVModels

# Two-start fit for `fit_nb_gllvm_grouped_cov`: the NB2 covariate objective is multimodal
# (beetle, 20 species: -2461.37 from the trait-mean start vs -2432.56 from the covariate-
# adjusted start). `starts = :both` keeps the better; `starts = :default` is the old fit.
function _two_start_sim(seed; p = 6, n = 40, K = 1)
    rng = MersenneTwister(seed)
    X = randn(rng, p, n, 1); Z = randn(rng, n, K); Λ = randn(rng, p, K) * 0.8
    Y = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(0.5 + 0.8 * X[t, s, 1] + Λ[t, :]' * Z[s, :]))))
         for t in 1:p, s in 1:n]
    return Y, X
end

@testset "NB2 grouped-cov two starts" begin
    @testset "_cov_ols_start recovers a noiseless shared slope" begin
        rng = MersenneTwister(1)
        p, n = 5, 30
        X = randn(rng, p, n, 2)
        β = randn(rng, p); γ = [0.7, -0.4]
        Z = [β[t] + γ[1] * X[t, i, 1] + γ[2] * X[t, i, 2] for t in 1:p, i in 1:n]
        β̂, γ̂, R = GMT._cov_ols_start(Z, X)
        @test γ̂ ≈ γ atol = 1e-6
        @test β̂ ≈ β atol = 1e-6
        @test maximum(abs, R) < 1e-6
    end

    @testset "starts = :default is the deterministic single-start fit" begin
        # No cross-platform optimum values are hard-coded: the objective is multimodal and the
        # single default start reaches different (equally legitimate) optima on different
        # BLAS/libm paths (Mac vs Linux CI differed by tens of log-likelihood units). The fit
        # has no separable single-start internal function, so the same-run checks are:
        #  (1) `:default` is reproducible bit-for-bit in one process (nothing random), and
        #  (2) `:both` returns exactly the `:default` fit unless the second start is better by
        #      more than the 1e-6 tie margin, i.e. the first start is never altered or lost.
        for seed in (11, 12)
            Y, X = _two_start_sim(seed)
            a1 = GMT.fit_nb_gllvm_grouped_cov(Y; X = X, K = 1, starts = :default)
            a2 = GMT.fit_nb_gllvm_grouped_cov(Y; X = X, K = 1, starts = :default)
            @test a1.loglik == a2.loglik
            @test a1.γ == a2.γ && a1.β == a2.β && a1.Λ == a2.Λ
            b = GMT.fit_nb_gllvm_grouped_cov(Y; X = X, K = 1, starts = :both)
            if b.loglik - a1.loglik <= 1e-6
                @test b.loglik == a1.loglik && b.γ == a1.γ && b.β == a1.β
            else
                @test b.loglik > a1.loglik + 1e-6
            end
        end
    end

    @testset ":both is never worse than :default" begin
        for seed in 11:14
            Y, X = _two_start_sim(seed)
            a = GMT.fit_nb_gllvm_grouped_cov(Y; X = X, K = 1, starts = :default)
            b = GMT.fit_nb_gllvm_grouped_cov(Y; X = X, K = 1)
            @test b.loglik >= a.loglik - 1e-6
            # where the starts agree (a tie) the default-start fit is returned unchanged
            b.loglik - a.loglik <= 1e-6 && @test b.γ == a.γ
        end
    end

    @test_throws ArgumentError GMT.fit_nb_gllvm_grouped_cov(
        _two_start_sim(11)[1]; X = _two_start_sim(11)[2], K = 1, starts = :nope)
end
