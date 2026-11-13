using GLLVModels, Test, Distributions, ForwardDiff
using SpecialFunctions: logabsgamma
const GM = GLLVModels

# Beta conditional log-density at a very large precision (#1143). `logpdf(Beta(a, b), y)` forms
# `loggamma(a+b) - loggamma(a) - loggamma(b)` from three terms of size φ log φ; at φ ≈ 1e12 Float64
# leaves ~1e-3 of rounding error per observation and at φ ≈ 1e18 hundreds, so a fit that ran to such
# a precision chased noise and reported a large positive log-likelihood. Reference: the same closed
# form in 512-bit BigFloat, evaluated at the Float64 shape parameters the code actually uses.

function _blp_ref(a, b, y)
    setprecision(BigFloat, 512) do
        A = BigFloat(a); B = BigFloat(b); Y = BigFloat(y)
        Float64(first(logabsgamma(A + B)) - first(logabsgamma(A)) - first(logabsgamma(B)) +
                (A - 1) * log(Y) + (B - 1) * log1p(-Y))
    end
end

@testset "Beta log-density is accurate at large precision" begin
    for φ in (1e3, 1e5, 1e8, 1e12, 1e15, 1e18),
        (μ, y) in ((0.5, 0.5), (0.3, 0.3 + 1e-9), (0.3, 0.3), (0.9, 0.9 + 0.5 / sqrt(φ)), (0.1, 0.2), (0.5, 0.01))
        a = μ * φ; b = (1 - μ) * φ
        got = GM._glm_logpdf(Beta(φ, 1.0), μ, 1, y)
        ref = _blp_ref(a, b, y)
        @test isapprox(got, ref; atol = 1e-7 + 1e-9 * abs(ref))
    end
end

@testset "Beta log-density below the switch is the original expression" begin
    for (a, b) in ((0.3, 0.7), (2.0, 5.0), (50.0, 50.0), (999.9, 5000.0), (5000.0, 999.9)), y in (0.2, 0.5, 0.97)
        @test GM._beta_logpdf(a, b, y) === logpdf(Beta(a, b), y)
    end
    for φ in (0.5, 8.0, 150.0, 1999.0), (μ, y) in ((0.5, 0.5), (0.3, 0.25))
        @test GM._glm_logpdf(Beta(φ, 1.0), μ, 1, y) === logpdf(Beta(μ * φ, (1 - μ) * φ), y)
    end
end

@testset "Beta log-density is continuous across the switch" begin
    for y in (0.5, 0.45, 0.3), (a, b) in ((1000.0, 1000.0), (prevfloat(1000.0), 1000.0), (1000.0, 3000.0))
        ref = _blp_ref(a, b, y)
        @test isapprox(GM._beta_logpdf(a, b, y), ref; atol = 1e-11 * max(1.0, abs(ref)))
    end
end

@testset "Beta log-density: ForwardDiff derivatives match BigFloat" begin
    fb(μ, φ, y) = setprecision(BigFloat, 512) do
        A = μ * φ; B = (1 - μ) * φ
        first(logabsgamma(A + B)) - first(logabsgamma(A)) - first(logabsgamma(B)) + (A - 1) * log(y) + (B - 1) * log1p(-y)
    end
    h = setprecision(() -> BigFloat(10)^-25, BigFloat, 512)
    for φ in (999.0, 2001.0, 1e6, 1e12), (μ, y) in ((0.5, 0.5), (0.3, 0.31), (0.9, 0.2))
        dμ = ForwardDiff.derivative(m -> GM._glm_logpdf(Beta(φ, 1.0), m, 1, y), μ)
        dφ = ForwardDiff.derivative(p -> GM._glm_logpdf(Beta(p, 1.0), μ, 1, y), φ)
        rμ, rφ = setprecision(BigFloat, 512) do
            M = BigFloat(μ); P = BigFloat(φ); Y = BigFloat(y)
            Float64((fb(M + h, P, Y) - fb(M - h, P, Y)) / 2h), Float64((fb(M, P + h * P, Y) - fb(M, P - h * P, Y)) / (2h * P))
        end
        @test isapprox(dμ, rμ; atol = 1e-6 * max(1.0, abs(rμ)))
        @test isapprox(dφ, rφ; atol = 1e-6 * max(1.0, abs(rφ)))
    end
end

@testset "Beta sibling densities agree with the stable form at large precision" begin
    φ = 1e12; μ = 0.4; y = 0.4 + 3e-7
    ref = _blp_ref(μ * φ, (1 - μ) * φ, y)
    @test isapprox(GM._beta_logpdf_full(μ, φ, y), ref; atol = 1e-7)
    @test isapprox(GM._beta_logker(μ, φ, y), ref; atol = 1e-7)
    η = log(μ / (1 - μ))
    @test isapprox(GM._beta_logpdf_eta(η, y, φ), ref; atol = 1e-6)
end

# Through the public fitter: the #1143 reproducer (3 species x 8 sites, one latent variable). On the
# cancelling density this fit ran to phi ~ 1e154 and reported a log-likelihood of ~5e140 with
# converged = false; with the stable density it ends near phi 1e11 at 12.8174508 (gllvmTMB: 12.817).
@testset "Beta fit on the #1143 reproducer reports an honest log-likelihood" begin
    Yb = [0.571331379859268 0.744755880395627 0.682314552251466 0.660536244100211 0.415202834511263 0.719573832324389 0.822579429039263 0.462027859928568;
          0.321409339262023 0.623485815704506 0.325871014829878 0.595350879884525 0.508422290025419 0.439766684969984 0.325323221129993 0.634288313010143;
          0.0917066732817654 0.501343958215348 0.319153002097073 0.335549643433248 0.636781236642335 0.864004794607531 0.277371918185606 0.528216395177476]
    fit = GM.fit_gllvm(Yb; family = GM.Beta(), K = 1)
    @test isfinite(fit.loglik)
    @test isapprox(fit.loglik, 12.8174508; atol = 1e-3)
end
