# #575: Tweedie series density speed. Value invariance vs the pre-#575 reference
# implementation, plus a machine-independent cost assertion (allocations).
using Test
using GLLVModels
using SpecialFunctions: loggamma

# Verbatim copy of the pre-#575 `_tweedie_logA` / `_tweedie_logsumexp`.
function _lse_reference(logw::AbstractVector)
    m = maximum(logw)
    (isfinite(m) || return m)
    s = 0.0
    @inbounds for lw in logw
        s += exp(lw - m)
    end
    return m + log(s)
end
function _logW_reference(y::Float64, φ::Float64, p::Float64)
    α = (2.0 - p) / (1.0 - p)
    a = -α * log(y) + α * log(p - 1.0) - (1.0 - α) * log(φ) - log(2.0 - p)
    logW(j) = j * a - loggamma(j + 1.0) - loggamma(-j * α)
    jstar = max(1, round(Int, y^(2.0 - p) / (φ * (2.0 - p))))
    drop = 37.0
    cap = 5000
    W = 1
    local lo, hi, m, terms
    while true
        lo = max(1, jstar - W)
        hi = jstar + W
        terms = Float64[logW(float(j)) for j in lo:hi]
        m = maximum(terms)
        edge = max(terms[1], terms[end])
        if (m - edge) ≥ drop || W ≥ cap
            break
        end
        W *= 2
    end
    return -log(y) + _lse_reference(terms)
end

@testset "#575 Tweedie series density: value invariance and cost" begin
    ys = [1e-6, 1e-3, 0.05, 0.5, 1.0, 3.0, 12.4, 60.0, 400.0, 5000.0]
    φs = [0.05, 0.4, 1.0, 2.5, 10.0]
    ps = [1.01, 1.1, 1.5, 1.9, 1.99]
    maxrel = 0.0
    for y in ys, φ in φs, p in ps
        new = GLLVModels._tweedie_logA(y, φ, p)
        ref = _logW_reference(y, φ, p)
        maxrel = max(maxrel, abs(new - ref) / max(abs(ref), 1.0))
    end
    @info "#575 max relative |Δ log a| over grid" maxrel
    @test maxrel ≤ 1e-10
    # full density incl. y = 0 point mass
    for (y, μ, φ, p) in ((0.0, 2.0, 1.0, 1.5), (0.0, 0.1, 0.5, 1.2), (3.0, 2.0, 1.0, 1.5))
        ref = y == 0 ? -μ^(2 - p) / (φ * (2 - p)) :
              (y * μ^(1 - p) / (1 - p) - μ^(2 - p) / (2 - p)) / φ + _logW_reference(y, φ, p)
        @test GLLVModels.tweedie_logpdf(y, μ, φ, p) ≈ ref rtol = 1e-10
    end

    # Cost: the old window rebuilt `terms` vectors on every doubling pass;
    # the single-pass series allocates nothing.
    GLLVModels._tweedie_logA(20.0, 1.0, 1.5)   # compile
    GLLVModels._tweedie_logA(0.5, 1.0, 1.5)
    @test (@allocated GLLVModels._tweedie_logA(20.0, 1.0, 1.5)) == 0
    @test (@allocated GLLVModels._tweedie_logA(60.0, 0.4, 1.2)) == 0
end
