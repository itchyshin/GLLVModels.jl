using GLLVModels, Test, Distributions, ForwardDiff
using Random
using SpecialFunctions: logabsgamma, loggamma
const GM = GLLVModels

# Gamma sibling sites of `_glm_logpdf(::Gamma, ...)` (refs #1175, #1143): the analytic-gradient
# kernel (`_gamma_logker`), the missing-predictor kernel (`_gamma_logpdf_full`), the delta-Gamma
# AGHQ cell density (`_delta_gamma_logf`) and the
# cross-validation cell log-likelihood all evaluated the Gamma(shape a, mean mu) log-density as
# `a log a - loggamma(a) + ...`, which cancels at a near 1e18 (error ~1e4 per cell). Each now goes
# through `_gamma_logpdf` (shape above 1e3) and is the original expression below it.
# Reference: the closed form in 512-bit BigFloat, at the Float64 mu the code uses.

function _gls_ref(a, μ, y)
    setprecision(BigFloat, 512) do
        A = BigFloat(a); M = BigFloat(μ); Y = BigFloat(y)
        Float64(-first(logabsgamma(A)) + A * log(A) - A * log(M) + (A - 1) * log(Y) - A * Y / M)
    end
end

const _GLS_CELLS = ((10.0, 10.0), (10.0, 12.5), (3.0, 0.4), (250.0, 244.0))
_ok(got, ref) = isapprox(got, ref; atol = 1e-9 + 1e-12 * abs(ref))

@testset "Gamma sibling densities are accurate at large shape" begin
    for a in (1e6, 1e12, 1e18), (μ, y) in _GLS_CELLS
        ref = _gls_ref(a, μ, y)
        @test _ok(GM._gamma_logker(μ, a, y), ref)
        @test _ok(GM._gamma_logpdf_full(μ, a, y), ref)
        # delta-Gamma AGHQ cell: log(pi) + Gamma log-density, mean mu = exp(eta_c)
        ηz = 0.3; ηc = log(μ); μc = exp(ηc)
        lπ = -log1p(exp(-ηz))                      # log pi, accurate to a few ulps
        refd = _gls_ref(a, μc, y) + lπ
        @test _ok(GM._delta_gamma_logf(y, ηz, ηc, a), refd)
    end
end

@testset "Gamma sibling densities below the switch are the original expressions" begin
    for a in (0.7, 4.0, 100.0, 999.0), (μ, y) in _GLS_CELLS
        old_expr(μ, a, y) = (a - 1) * log(y) - y * a / μ - a * log(μ) + a * log(a) - loggamma(a)
        @test GM._gamma_logker(μ, a, y) === old_expr(μ, a, y)
        @test GM._gamma_logpdf_full(μ, a, y) === old_expr(μ, a, y)
        ηz = 0.3; ηc = log(μ)
        @test GM._delta_gamma_logf(y, ηz, ηc, a) === -log1p(exp(-ηz)) + a * log(a) - loggamma(a) - a * ηc + (a - 1) * log(y) - a * y * exp(-ηc)
    end
end

# The cross-validation cell also evaluates a Gamma cdf for its residual, which throws in the far tail
# for shapes above 50 (SpecialFunctions `expint`), so its density is checked on cells where the cdf is
# fine: the original expression below the switch, BigFloat at shape 1e6 with y at the mean.
@testset "cross-validation Gamma cell log-likelihood" begin
    for a in (0.7, 4.0), (μ, y) in _GLS_CELLS
        @test first(GM._cv_cell_eval((α = a,), Gamma(1.0, 1.0), y, μ, 1, 1, nothing, Random.default_rng())) === logpdf(Gamma(a, max(μ, 1e-12) / a), y)
    end
    ll = first(GM._cv_cell_eval((α = 1e6,), Gamma(1.0, 1.0), 10.0, 10.0, 1, 1, nothing, Random.default_rng()))
    @test isapprox(ll, _gls_ref(1e6, 10.0, 10.0); atol = 1e-12)
end

@testset "analytic-gradient kernel: ForwardDiff derivative in log shape matches BigFloat" begin
    # the shared-shape fit differentiates in log(alpha); d/dlog(alpha) = alpha d/dalpha, so the
    # rounding of the old kernel (about 4e-15 per unit alpha) is multiplied by alpha
    h = setprecision(() -> BigFloat(10)^-25, BigFloat, 512)
    f(a, μ, y) = -first(logabsgamma(a)) + a * log(a) - a * log(μ) + (a - 1) * log(y) - a * y / μ
    for a0 in (999.0, 1001.0, 1.37e6, 3.1e10, 7.3e13, 1.37e17), (μ, y) in ((10.0, 10.0), (10.0, 12.5), (3.0, 0.4))
        s0 = log(a0)
        d = ForwardDiff.derivative(s -> GM._gamma_logker(μ, exp(s), y), s0)
        r = setprecision(BigFloat, 512) do
            A = BigFloat(exp(s0))
            Float64(A * (f(A + h * A, BigFloat(μ), BigFloat(y)) - f(A - h * A, BigFloat(μ), BigFloat(y))) / (2h * A))
        end
        @test isapprox(d, r; atol = 1e-9 * max(1.0, abs(r)))
    end
end
