using GLLVModels, Test, TOML, SpecialFunctions
const GM = GLLVModels

# NB1 toward its Poisson limit φ → 0 (NB1 follow-up to #615). Two defects:
# (1) the density, score and curvatures differenced digamma / trigamma / loggamma of
#     r = μ/φ, which loses the result as r grows: at φ = 1e-6, d logpdf / d log φ was
#     40 times too large, so the gradient in log φ was noise near the limit;
# (2) like NB2, the likelihood in log φ flattens toward the limit, and the fit crawled
#     there until the iteration cap. The NB1 grouped fitters now use the NB2 polish
#     mirrored (species with φ < 1e-3 fixed at φ = 1e-10, BFGS refit).

@testset "NB1 toward the Poisson limit" begin
    @testset "density, score and curvature match a BigFloat reference" begin
        setprecision(BigFloat, 256) do
            refl(μ, φ, y) = (r = big(μ) / big(φ);
                loggamma(y + r) - loggamma(r) - loggamma(big(y) + 1) - r * log1p(big(φ)) +
                y * (log(big(φ)) - log1p(big(φ))))
            rise1(r, y) = sum((1 / (r + k) for k in 0:(y - 1)); init = big(0))
            rise2(r, y) = sum((1 / (r + k)^2 for k in 0:(y - 1)); init = big(0))
            for μ in (0.05, 0.7, 3.0, 25.0), y in (0, 1, 3, 12, 60),
                    φ in (5.0, 1.0, 0.1, 1e-3, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12)
                f = GM.NB1(φ)
                r = big(μ) / big(φ)
                lr = Float64(refl(μ, φ, y))
                sr = Float64((rise1(r, y) - log1p(big(φ))) / big(φ))
                wr = Float64(-big(μ) * (rise1(r, y) - log1p(big(φ))) / big(φ) +
                             (big(μ) / big(φ))^2 * rise2(r, y))
                @test GM._glm_logpdf(f, μ, 1, y) ≈ lr rtol = 1e-12 atol = 1e-12
                @test GM._glm_score(f, μ, 1, 1.0, y) ≈ sr rtol = 1e-12 atol = 1e-12
                @test GM._nb1_grouped_laplace_weight(:observed, f, μ, μ, y, GM.LogLink()) ≈ wr rtol = 1e-10 atol = 1e-12
            end
        end
    end

    @testset "d logpdf / d log φ is right near the limit" begin
        # Near φ = 0 the derivative in log φ is about φ times a constant; the old density
        # gave 6.2e-5 at φ = 1e-6 (true 1.5e-6) for y = 0, μ = 3.
        μ, h = 3.0, 1e-4
        for y in (0, 3, 12), φ in (1e-4, 1e-6, 1e-8)
            d = (GM._glm_logpdf(GM.NB1(φ * exp(h)), μ, 1, y) -
                 GM._glm_logpdf(GM.NB1(φ * exp(-h)), μ, 1, y)) / 2h
            dpois = φ * ((y - μ)^2 - y) / (2μ)      # first-order term of logpdf − Poisson, times φ
            @test d ≈ dpois rtol = 1e-2 atol = 1e-12
        end
    end

    @testset "per-species fit that crawled to φ → 0" begin
        fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "nb1_grouped_poisson_ridge.toml"))
        p, n, K = fixture["p"], fixture["n"], fixture["K"]
        Y = reshape(Int64.(fixture["Y_column_major"]), p, n)
        fit = GM.fit_nb1_gllvm_grouped(Y; K = K)
        @test fit.converged
        @test fit.loglik >= fixture["loglik_main"]
        @test GM.nb1_grouped_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.φ[fit.group];
                                                     maxiter = 100, tol = 1e-9) ≈ fit.loglik rtol = 1e-10
    end
end
