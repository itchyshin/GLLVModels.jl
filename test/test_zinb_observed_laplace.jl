# #932: fit_zinb_gllvm's betaz Wald intervals under-covered (cell 13: 0.83) because the ZINB
# Laplace log-determinant used the EXPECTED count information even though `hessian = :observed`
# (the default) promises the observed one. Same defect as ZIP (#739 / #913). The reference
# below is independent of the package's mode search and weight code: Newton with ForwardDiff
# on the joint site log-posterior, then q(ẑ) - ½ logdet(-∇²q(ẑ)).
using Test, GLLVModels, ForwardDiff, LinearAlgebra, Random, Distributions, SpecialFunctions
const _G932 = GLLVModels

# log f(y | ηz, ηc, r) for the zero-inflated NB2, written out here (not via `_tp_pieces`).
function _zinb_logf932(y, ηz, ηc, r)
    π = 1 / (1 + exp(-ηz)); μ = exp(ηc)
    lp0 = r * (log(r) - log(r + μ))                      # log NB2 P(0)
    y > 0 && return log1p(-π) + loggamma(y + r) - loggamma(r) - loggamma(y + 1) +
                    r * (log(r) - log(r + μ)) + y * (ηc - log(r + μ))
    return log(π + (1 - π) * exp(lp0))
end

function _zinb_site_exact932(y, Λ, βz, βc, r)
    p, K = size(Λ)
    q(z) = sum(_zinb_logf932(y[t], βz[t], βc[t] + dot(Λ[t, :], z), r) for t in 1:p) - 0.5 * dot(z, z)
    z = zeros(K)
    for _ in 1:200
        g = ForwardDiff.gradient(q, z); H = ForwardDiff.hessian(q, z)
        step = (-H) \ g
        z = z + step
        norm(step) < 1e-13 && break
    end
    H = ForwardDiff.hessian(q, z)
    return q(z) - 0.5 * logdet(Symmetric(-H))
end

const _Y932 = Float64[
    0 3 0 5 2 0 7 1;
    4 0 0 2 6 1 0 3;
    0 0 6 1 0 4 2 0;
    2 5 1 0 3 0 4 6;
    1 0 3 4 0 2 0 5]
const _Λ932 = [0.5 0.0; -0.3 0.4; 0.2 -0.5; 0.4 0.3; -0.2 0.2]
const _βz932 = [-1.5, -1.0, -1.2, -0.8, -1.4]
const _βc932 = [0.9, 1.0, 0.8, 1.1, 0.7]
const _r932 = 2.0

@testset "#932 ZINB Laplace log-determinant uses the observed count curvature" begin
    exact = sum(_zinb_site_exact932(_Y932[:, s], _Λ932, _βz932, _βc932, _r932) for s in axes(_Y932, 2))

    @testset ":observed matches the independent exact-Hessian Laplace" begin
        obs = _G932.zinb_marginal_loglik_laplace(_Y932, _Λ932, _βz932, _βc932, _r932; hessian = :observed)
        @test obs ≈ exact atol = 1e-7
    end

    @testset "the default is :observed" begin
        @test _G932.zinb_marginal_loglik_laplace(_Y932, _Λ932, _βz932, _βc932, _r932) ==
              _G932.zinb_marginal_loglik_laplace(_Y932, _Λ932, _βz932, _βc932, _r932; hessian = :observed)
    end

    @testset ":fisher is a different surface (the selector is effective)" begin
        fis = _G932.zinb_marginal_loglik_laplace(_Y932, _Λ932, _βz932, _βc932, _r932; hessian = :fisher)
        @test abs(fis - exact) > 1e-3
    end

    @testset "agrees with the gllvmTMB-named zi_nbinom2() evaluator" begin
        twin = _G932.zi_marginal_loglik_laplace(_G932.zi_nbinom2(), _Y932, _Λ932, _βc932, _βz932;
                                                phi = fill(_r932, 5))
        obs = _G932.zinb_marginal_loglik_laplace(_Y932, _Λ932, _βz932, _βc932, _r932)
        @test obs ≈ twin atol = 1e-8
    end

    @testset "the hand weight equals the ForwardDiff curvature of the log-density" begin
        for y in (0, 1, 4, 12), ηz in (-2.0, -0.5, 1.0), ηc in (-1.0, 0.3, 1.4), r in (0.7, 2.0, 9.0)
            ad = -ForwardDiff.derivative(e -> ForwardDiff.derivative(e2 -> _zinb_logf932(y, ηz, e2, r), e), ηc)
            @test _G932._tp_observed_Wc_z(_G932.ZINB(r), y, ηz, ηc, NaN) ≈ ad rtol = 1e-9 atol = 1e-12
        end
    end
end

# The mechanism in the issue: the objective's own pseudo-true shift of betaz at the TRUE
# parameters. δ* = -score/curvature of the marginal log-likelihood along a common betaz shift.
# On unmodified main the Fisher-weight objective gives δ* ≈ +0.16 at π = 0.2 (it does not
# shrink with n); the observed-curvature Laplace and AGHQ give about -0.01.
function _zinb_delta_star932(; n, π0, hessian, seed = 777)
    p, K = 10, 2
    Λ = [0.8 0.0; 0.5 0.6; -0.4 0.5; 0.6 -0.3; 0.3 0.6; -0.5 0.4; 0.7 -0.2; 0.2 -0.5; -0.3 0.5; 0.4 0.3]
    βc = collect(range(0.5, 1.5; length = p)); r = 2.0; βz = fill(log(π0 / (1 - π0)), p)
    rng = Xoshiro(seed); Z = randn(rng, K, n); Y = zeros(Int, p, n)
    for s in 1:n, t in 1:p
        μ = exp(βc[t] + dot(Λ[t, :], Z[:, s]))
        Y[t, s] = rand(rng) < π0 ? 0 : rand(rng, NegativeBinomial(r, r / (r + μ)))
    end
    f(δ) = _G932.zinb_marginal_loglik_laplace(Y, Λ, βz .+ δ, βc, r; hessian = hessian)
    h = 0.05; fp = f(h); fm = f(-h); f0 = f(0.0)
    return -((fp - fm) / (2h)) / ((fp + fm - 2 * f0) / h^2)
end

@testset "#932 the objective's betaz shift at the true parameters is near zero" begin
    δobs = _zinb_delta_star932(; n = 3000, π0 = 0.2, hessian = :observed)
    δfis = _zinb_delta_star932(; n = 3000, π0 = 0.2, hessian = :fisher)
    @test abs(δobs) < 0.06            # main: +0.16
    @test δfis > 0.10                 # :fisher keeps the biased expected-information surface
end

@testset "#932 K = 0: the eigenvalue floor is skipped (eigmin throws on a 0x0 precision)" begin
    Yk0 = Int.(_Y932)
    @test isfinite(_G932.fit_zinb_gllvm(Yk0; K = 0).loglik)
    @test isfinite(_G932.fit_zip_gllvm(Yk0; K = 0).loglik)
end
