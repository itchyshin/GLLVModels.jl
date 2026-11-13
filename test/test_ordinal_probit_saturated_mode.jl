# Ordinal probit: the Laplace mode search and value for a site whose category probability is
# far in a tail (large intercept and loading, response on the wrong side). Before the fix the
# category probability was the difference of two CDFs floored at 1e-12, so the mode search
# saw no gradient and the Laplace value sat about 25 per such cell below the exact value.
#
# Reference: 256-bit bisection on the exact score (K = 1, log-concave in z), with tail
# probabilities from `erfc` on the complement side, never from `1 - Φ`.
using Test, GLLVModels
using SpecialFunctions: erfc

const _GMord = GLLVModels

_ordΦ(x) = erfc(-x / sqrt(one(x) * 2)) / 2
_ordlφ(x) = -x^2 / 2 - log(2 * big(π)) / 2

# P(y = c), score dlogP/dη and −d²logP/dη² at η, with cutpoints τ (length C-1).
function _ord_exact(c, η, τ)
    C = length(τ) + 1
    a = c == C ? nothing : τ[c] - η
    b = c == 1 ? nothing : τ[c - 1] - η
    P = if b !== nothing && b >= 0
        _ordΦ(-b) - (a === nothing ? zero(η) : _ordΦ(-a))
    else
        (a === nothing ? one(η) : _ordΦ(a)) - (b === nothing ? zero(η) : _ordΦ(b))
    end
    φa = a === nothing ? zero(η) : exp(_ordlφ(a))
    φb = b === nothing ? zero(η) : exp(_ordlφ(b))
    score = (φb - φa) / P
    ddP = (b === nothing ? zero(η) : b * φb) - (a === nothing ? zero(η) : a * φa)
    return log(P), score, score^2 - ddP / P
end

function _ord_ref_site(y, Λ, β, τ)
    old = precision(BigFloat)
    setprecision(BigFloat, 256)
    try
        p = length(y)
        Λb = BigFloat.(Λ); βb = BigFloat.(β); τb = BigFloat.(τ)
        grad(z) = sum(Λb[t] * _ord_exact(y[t], βb[t] + Λb[t] * z, τb[t, :])[2] for t in 1:p) - z
        lo, hi = BigFloat(-200), BigFloat(200)
        for _ in 1:400
            mid = (lo + hi) / 2
            grad(mid) > 0 ? (lo = mid) : (hi = mid)
        end
        z = (lo + hi) / 2
        ex = [_ord_exact(y[t], βb[t] + Λb[t] * z, τb[t, :]) for t in 1:p]
        g = sum(e[1] for e in ex) - z^2 / 2
        A = 1 + sum(ex[t][3] * Λb[t]^2 for t in 1:p)
        return Float64(z), Float64(g - log(A) / 2)
    finally
        setprecision(BigFloat, old)
    end
end

@testset "Ordinal probit: mode search from a saturated wrong-side start" begin
    Λ = [30.0, 0.4, -0.3]
    τ = [0.0 1.0 2.2; 0.0 0.9 2.0; 0.0 1.1 2.4]
    C = [4, 4, 4]
    cases = (
        ([10.0, 0.1, 0.2],  [1, 2, 3]),
        ([-10.0, 0.1, 0.2], [4, 2, 3]),
        ([10.0, 0.1, 0.2],  [2, 2, 3]),
    )
    for (β, y) in cases
        zref, vref = _ord_ref_site(y, Λ, β, τ)
        Y = reshape(y, 3, 1)
        z = _GMord._ordinal_laplace_mode_pertrait(y, reshape(Λ, 3, 1), β, τ, C, _GMord.ProbitLink())
        @test isapprox(z[1], zref; atol = 1e-6)
        v = _GMord.ordinal_marginal_loglik_laplace_pertrait(Y, reshape(Λ, 3, 1), β, τ, C;
                                                            link = _GMord.ProbitLink())
        @test isapprox(v, vref; atol = 1e-6)
    end
end

@testset "Ordinal probit: unsaturated cells use the original expressions" begin
    τv = [0.0, 1.0, 2.2]
    for c in 1:4, η in (-0.7, 0.3, 1.4)
        P = _GMord._ord_prob(c, η, τv, _GMord.ProbitLink())
        @test P > 1e-12
        @test _GMord._ord_logprob(c, η, τv, _GMord.ProbitLink()) === log(P)
    end
end
