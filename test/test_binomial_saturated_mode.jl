# Binomial probit / cloglog: the Laplace mode search must find the mode of a site whose
# starting predictor is deep in a link tail on the wrong side (a Bernoulli cell with a
# large intercept and a large loading). Before the fix the μ-based score and log density
# were flat once μ hit the `_clamp_mu` bound, and the search stopped near z = 0.
#
# The reference is a 256-bit bisection on the exact score of the same Laplace problem
# (K = 1, so the log posterior is concave in z for these log-concave links), with the
# tail probabilities from `erfc` / `expm1`, never from `1 - μ`.
using Test, GLLVModels
using SpecialFunctions: erfc

const _GMsat = GLLVModels

# Exact pieces of the Bernoulli log density, score and observed curvature wrt η.
_Φb(x) = erfc(-x / sqrt(one(x) * 2)) / 2
_lφb(x) = -x^2 / 2 - log(2 * big(π)) / 2
_mb(x) = exp(_lφb(x)) / _Φb(x)
_ell(::Val{:probit}, yt, η) = yt == 1 ? log(_Φb(η)) : log(_Φb(-η))
_sc(::Val{:probit}, yt, η) = yt == 1 ? _mb(η) : -_mb(-η)
_wt(::Val{:probit}, yt, η) = yt == 1 ? _mb(η) * (_mb(η) + η) : _mb(-η) * (_mb(-η) - η)
_ell(::Val{:cloglog}, yt, η) = (u = exp(η); yt == 1 ? log(-expm1(-u)) : -u)
_sc(::Val{:cloglog}, yt, η) = (u = exp(η); yt == 1 ? u * exp(-u) / (-expm1(-u)) : -u)
_wt(::Val{:cloglog}, yt, η) = (u = exp(η); v = exp(-u); o = -expm1(-u);
                               yt == 1 ? u^2 * v / o^2 - u * v / o : u)

function _ref_site(link::Symbol, y, Λ, β)
    old = precision(BigFloat)
    setprecision(BigFloat, 256)
    try
        L = Val(link); p = length(y)
        Λb = BigFloat.(Λ); βb = BigFloat.(β)
        grad(z) = sum(Λb[t] * _sc(L, y[t], βb[t] + Λb[t] * z) for t in 1:p) - z
        lo, hi = BigFloat(-200), BigFloat(200)
        for _ in 1:400
            mid = (lo + hi) / 2
            grad(mid) > 0 ? (lo = mid) : (hi = mid)
        end
        z = (lo + hi) / 2
        g = sum(_ell(L, y[t], βb[t] + Λb[t] * z) for t in 1:p) - z^2 / 2
        A = 1 + sum(_wt(L, y[t], βb[t] + Λb[t] * z) * Λb[t]^2 for t in 1:p)
        return Float64(z), Float64(g - log(A) / 2)
    finally
        setprecision(BigFloat, old)
    end
end

@testset "Binomial probit/cloglog: mode search from a saturated wrong-side start" begin
    Λ = reshape([38.5, 0.2, -0.3, 0.5], 4, 1)
    cases = (
        (:probit,  _GMsat.ProbitLink(),  [8.9, 0.4, 0.2, -0.3],  [0, 1, 0, 1]),
        (:probit,  _GMsat.ProbitLink(),  [-8.9, 0.4, 0.2, -0.3], [1, 1, 0, 1]),
        (:cloglog, _GMsat.CLogLogLink(), [4.0, 0.4, 0.2, -0.3],  [0, 1, 0, 1]),
        (:cloglog, _GMsat.CLogLogLink(), [-4.0, 0.4, 0.2, -0.3], [1, 1, 0, 1]),
    )
    for (nm, lk, β, y) in cases
        zref, vref = _ref_site(nm, y, Λ[:, 1], β)
        z = _GMsat._laplace_mode(_GMsat.Binomial(), Float64.(y), ones(4), Λ, β, lk)
        @test isapprox(z[1], zref; atol = 1e-6)
        v = _GMsat.binomial_marginal_loglik_laplace(reshape(Float64.(y), 4, 1), ones(4, 1), Λ, β, lk)
        @test isapprox(v, vref; atol = 1e-6)
    end
end

@testset "Binomial probit/cloglog: unsaturated cells are unchanged by the saturated-cell branch" begin
    Λ = reshape([0.8, -0.5, 0.3, 0.6], 4, 1); β = [0.2, -0.1, 0.4, 0.0]; y = [1.0, 0.0, 1.0, 0.0]
    for lk in (_GMsat.ProbitLink(), _GMsat.CLogLogLink())
        η = β .+ Λ * zeros(1)
        μ = _GMsat._clamp_mu.(Ref(_GMsat.Binomial()), _GMsat.linkinv.(Ref(lk), η))
        me = _GMsat.mu_eta.(Ref(lk), η)
        for t in 1:4
            @test _GMsat._glm_score_eta(_GMsat.Binomial(), μ[t], 1, me[t], y[t], lk, η[t]) ===
                  _GMsat._glm_score(_GMsat.Binomial(), μ[t], 1, me[t], y[t])
            @test _GMsat._laplace_mode_step_weight(_GMsat.Binomial(), μ[t], 1, me[t], y[t], lk, η[t]) ===
                  _GMsat._glm_weight(_GMsat.Binomial(), μ[t], 1, me[t])
        end
    end
end
