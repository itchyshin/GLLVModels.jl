using GLLVModels, Test, ForwardDiff
using SpecialFunctions: loggamma
const GM = GLLVModels

# Variational Gamma and delta-Gamma ELBO at large shape (refs #1175, #1143). The per-species
# expectation E_q[log f(y | eta, a)] was coded as `(a-1) log y - y a w - a mu_eta + a log a - loggamma(a)`
# with w = exp(-mu_eta + sigma2/2); the last two terms and the a-sized terms before them cancel at
# large a. From shape 1e3 the site now evaluates `_gamma_logpdf(a, e^mu_eta, y) - a (y / e^mu_eta) expm1(sigma2/2)`.
# Reference values are the closed-form expectation in 512-bit BigFloat, stored as literals. They take the
# Float64 predictor mu_eta and variance sigma2 that the site forms as given: a reference at the exact real
# predictor would differ by the Float64 rounding of mu_eta times the condition number a (u + sigma2/2),
# which no regrouping removes. All inputs are literals, so Julia 1.10 and 1.13 evaluate identical cells.

const _VGS_Y = [3.0, 5.0, 0.8]
const _VGS_Y0 = [3.0, 0.0, 0.8]
const _VGS_BZ = [0.3, -0.2, 0.5]
const _VGS_PSI = [0.3, -0.7]
# (shape, loading scale, betac, BigFloat gamma ELBO, BigFloat delta-Gamma ELBO)
const _VGS_CELLS = (
    (1.0e6, 0.0001, [1.0986122886681098, 1.6094379124341003, -0.2231435513142097], 15.335611083691305, 9.330130508949408),
    (1.0e6, 1.0e-6, [1.0996122886681097, 1.6084379124341004, -0.2226435513142097], 14.212866232906693, 8.707203200201809),
    (1.0e6, 1.0, [1.0986122886681098, 1.6094379124341003, -0.2231435513142097], -244455.56379870305, -191271.2296238793),
    (1.0e9, 0.0001, [1.0986122886681098, 1.6094379124341003, -0.2231435513142097], 23.06028189648801, 14.069727624067033),
    (1.0e9, 1.0e-6, [1.0986439114447115, 1.6094062896574985, -0.22312773992590887], 24.561813392674534, 15.605832813906424),
    (1.0e9, 1.0, [1.0986122886681098, 1.6094379124341003, -0.2231435513142097], -2.4447087634950483e8, -1.9128054568466064e8),
    (1.0e12, 0.0001, [1.0986122886681098, 1.6094379124341003, -0.2231435513142097], -2603.540440611289, -2147.180847461296),
    (1.0e12, 1.0e-6, [1.0986132886681097, 1.6094369124341004, -0.22314305131420972], 34.26755363582374, 22.020775528361266),
    (1.0e12, 1.0, [1.0986122886681098, 1.6094379124341003, -0.2231435513142097], -2.4447090201332718e11, -1.912805619015691e11),
    (1.0e15, 0.0001, [1.0986122886681098, 1.6094379124341003, -0.2231435513142097], -2.6395555342341927e6, -2.1702986034566937e6),
    (1.0e15, 1.0e-6, [1.0986123202908864, 1.6094378808113237, -0.2231435355028214], -231.4724253909172, -196.6184441336069),
    (1.0e15, 1.0, [1.0986122886681098, 1.6094379124341003, -0.2231435513142097], -2.4447090204934228e14, -1.9128056192468684e14),
)
# The new form is limited by Float64 rounding of exp(mu_eta) (relative 1.1e-16) times a (u + sigma2/2), at most
# 1e-8 absolute and 1e-11 relative on these cells; the old form is off by 6e-7 to 1.9 on the same cells.
_vgs_ok(got, ref) = isapprox(got, ref; atol = 2e-8, rtol = 1e-11)

@testset "variational Gamma and delta-Gamma ELBO are accurate at large shape" begin
    for (a, s, βc, rg, rd) in _VGS_CELLS
        Λ = s .* reshape([0.7, -0.4, 0.5], 3, 1); Λ2 = Λ .^ 2
        @test _vgs_ok(GM._va_site_gamma_elbo(_VGS_PSI, _VGS_Y, Λ, Λ2, βc, a), rg)
        @test _vgs_ok(GM._va_site_dgamma_elbo(_VGS_PSI, _VGS_Y0, Λ, Λ2, _VGS_BZ, βc, a), rd)
    end
end

@testset "public marginals at large shape (Lambda = 0: independent Gamma log-likelihood)" begin
    Y = [3.0 3.2 2.9; 5.0 4.8 5.1; 0.8 0.81 0.79]; β = log.([3.0, 5.0, 0.8])
    for (a, ref) in ((1.0e9, -3.871915783753714e6), (1.0e12, -3.8719931952169247e9))
        @test _vgs_ok(GM.gamma_marginal_loglik_va(Y, zeros(3, 1), β, a), ref)
    end
end

@testset "below the switch the variational Gamma sites are the original expressions" begin
    Λ = [0.7; -0.4; 0.5]; Λ = reshape(Λ, 3, 1); Λ2 = Λ .^ 2
    βc = log.(_VGS_Y) .+ [0.05, -0.1, 0.0]; ψ = _VGS_PSI
    m = ψ[1:1]; lv = ψ[2:2]; v = exp.(lv)
    σ2 = Λ2 * v; μη = βc .+ Λ * m
    kl = 0.5 * sum(v .+ m .^ 2 .- 1.0 .- lv)
    for a in (0.7, 4.0, 100.0, 999.0)
        ℓ = 0.0
        for t in 1:3
            w = exp(GM._clamp_eta(-μη[t] + 0.5 * σ2[t]))
            ℓ += (a - 1) * log(_VGS_Y[t]) - _VGS_Y[t] * a * w - a * μη[t] + a * log(a) - loggamma(a)
        end
        @test GM._va_site_gamma_elbo(ψ, _VGS_Y, Λ, Λ2, βc, a) === ℓ - kl
        ℓd = 0.0
        for t in 1:3
            π = clamp(inv(one(_VGS_BZ[t]) + exp(-_VGS_BZ[t])), 1e-12, 1 - 1e-12)
            w = exp(GM._clamp_eta(-μη[t] + 0.5 * σ2[t]))
            ℓd += log(π) + (a - 1) * log(_VGS_Y[t]) - _VGS_Y[t] * a * w - a * μη[t] + a * log(a) - loggamma(a)
        end
        @test GM._va_site_dgamma_elbo(ψ, _VGS_Y, Λ, Λ2, _VGS_BZ, βc, a) === ℓd - kl
    end
end

@testset "hand-written ELBO gradients match ForwardDiff at large shape" begin
    Λ = reshape([0.7, -0.4, 0.5], 3, 1); Λ2 = Λ .^ 2
    for a in (1.0e3, 1.0e6, 1.0e12), s in (1.0, 1e-3)
        βc = log.(_VGS_Y) .+ 1e-3 / sqrt(a) .* [1.0, -1.0, 0.5]
        Λs = s .* Λ; Λs2 = Λs .^ 2
        ψ = [0.3, -0.7]; G = zeros(2)
        f(x) = -GM._va_site_gamma_elbo(x, _VGS_Y, Λs, Λs2, βc, a)
        GM._va_site_gamma_grad!(G, ψ, _VGS_Y, Λs, Λs2, βc, a)
        @test isapprox(G, ForwardDiff.gradient(f, ψ); rtol = 1e-8, atol = 1e-8 * max(1.0, a * s^2))
        fd(x) = -GM._va_site_dgamma_elbo(x, _VGS_Y0, Λs, Λs2, _VGS_BZ, βc, a)
        GM._va_site_dgamma_grad!(G, ψ, _VGS_Y0, Λs, Λs2, _VGS_BZ, βc, a)
        @test isapprox(G, ForwardDiff.gradient(fd, ψ); rtol = 1e-8, atol = 1e-8 * max(1.0, a * s^2))
    end
end
