using GLLVModels, Distributions, Test, Random, Logging

# #1107: Gaussian fitters must derive an observation mask from `missing` in Y
# (as Poisson does) and say so once, instead of throwing MethodError.
# On current main the derived-mask fits below throw MethodError.

function _info_text(f)
    io = IOBuffer()
    logger = SimpleLogger(io, Logging.Info)
    result = with_logger(f, logger)
    return result, String(take!(io))
end

_derived_count(text) = count("derived the observation mask", text)

@testset "issue 1107: derive the Gaussian mask from missing" begin
    Random.seed!(1)
    p, n, K = 4, 8, 1
    Y = 0.7 .* randn(p, K) * randn(K, n) .+ 0.4 .* randn(p, n)
    Ym = Matrix{Union{Float64,Missing}}(Y)
    Ym[1, 2] = missing
    Ym[3, 4] = missing
    mask = observed_mask(Ym)
    kw = (g_tol = 1e-4, iterations = 30)

    @testset "per-species likelihood of observed subvectors" begin
        Λ = randn(p, K)
        φ² = fill(0.3, p)
        full = gaussian_pervar_marginal_loglik(Y, Λ, φ²)
        masked_all = gaussian_pervar_marginal_loglik(Y, Λ, φ²; mask = trues(p, n))
        @test masked_all ≈ full atol = 1e-8
        m1 = trues(p, n)
        m1[1, 1] = false
        dropped = gaussian_pervar_marginal_loglik(Y, Λ, φ²; mask = m1)
        @test isfinite(dropped)
        @test abs(dropped - full) > 1e-6
        Yg = copy(Y)
        Yg[1, 1] = 1e6
        @test gaussian_pervar_marginal_loglik(Yg, Λ, φ²; mask = m1) ≈ dropped atol = 1e-8
    end

    @testset "explicit mask cannot mark a missing cell observed" begin
        bad = copy(mask)
        bad[1, 2] = true
        @test_throws ArgumentError fit_gaussian_gllvm(Ym; K = K, mask = bad, kw...)
        @test_throws ArgumentError fit_gaussian_pervar_gllvm(Ym; K = K, mask = bad,
                                                            method = :lbfgs, kw...)
    end

    @testset "complete data does not derive a mask" begin
        _, text = _info_text() do
            fit_gaussian_gllvm(Y; K = K, kw...)
        end
        @test _derived_count(text) == 0
        _, textp = _info_text() do
            fit_gaussian_pervar_gllvm(Y; K = K, method = :lbfgs, iterations = 5, g_tol = 1e-3)
        end
        @test _derived_count(textp) == 0
    end

    @testset "fit_gaussian_gllvm derives the mask once" begin
        fit_ref, ref_text = _info_text() do
            fit_gaussian_gllvm(Y; K = K, mask = mask, kw...)
        end
        @test _derived_count(ref_text) == 0
        (fit_d, fit_again), text = _info_text() do
            first = fit_gaussian_gllvm(Ym; K = K, kw...)
            second = fit_gaussian_gllvm(Ym; K = K, g_tol = 1e-3, iterations = 2)
            return first, second
        end
        @test isfinite(fit_d.logLik)
        @test fit_d.logLik ≈ fit_ref.logLik atol = 1e-8
        @test occursin("fit_gaussian_gllvm: derived the observation mask", text)
        @test _derived_count(text) == 1
        @test isfinite(fit_again.logLik)
    end

    @testset "fit_gllvm(Normal) derives the mask" begin
        fit_ref = fit_gllvm(Y; family = Normal(), K = K, mask = mask, kw...)
        fit_d, text = _info_text() do
            fit_gllvm(Ym; family = Normal(), K = K, kw...)
        end
        @test isfinite(fit_d.logLik)
        @test fit_d.logLik ≈ fit_ref.logLik atol = 1e-8
        @test occursin("fit_gaussian_gllvm: derived the observation mask", text)
        @test _derived_count(text) == 1
    end

    @testset "fit_gaussian_pervar_gllvm derives the mask once" begin
        Yg = copy(Y)
        Yg[1, 2] = 1e6
        Yg[3, 4] = -1e6
        pkw = (method = :lbfgs, g_tol = 1e-5, iterations = 40)
        # The no-mask call on `Ym` is the #1107 failure (MethodError on main).
        (fit_d, fit_again), text = _info_text() do
            first = fit_gaussian_pervar_gllvm(Ym; K = K, pkw...)
            second = fit_gaussian_pervar_gllvm(Ym; K = K, method = :lbfgs,
                                               g_tol = 1e-3, iterations = 2)
            return first, second
        end
        fit_ref = fit_gaussian_pervar_gllvm(Yg; K = K, mask = mask, pkw...)
        @test isfinite(fit_d.loglik)
        @test fit_d.loglik ≈ fit_ref.loglik atol = 1e-6
        @test fit_d.β ≈ fit_ref.β atol = 1e-5
        @test fit_d.φ² ≈ fit_ref.φ² atol = 1e-5
        @test occursin("fit_gaussian_pervar_gllvm: derived the observation mask", text)
        @test _derived_count(text) == 1
        @test isfinite(fit_again.loglik)
        Xint = GLLVModels._trait_intercept_design(p, n)
        Yf = Float64.(coalesce.(Ym, 0.0))
        ll = gaussian_pervar_marginal_loglik(Yf, fit_d.Λ, fit_d.φ²;
                                             X = Xint, β = fit_d.β, mask = mask)
        @test ll ≈ fit_d.loglik atol = 1e-6
        # The missing cells are not data: fitting the garbage values unmasked moves the likelihood.
        fit_garbage = fit_gaussian_pervar_gllvm(Yg; K = K, pkw...)
        @test abs(fit_garbage.loglik - fit_d.loglik) > 1e-2
    end
end
