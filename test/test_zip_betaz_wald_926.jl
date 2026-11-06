# #926: Wald intervals for the ZIP zero-inflation logits (betaz) under-covered (0.86 to 0.91
# against 0.95) because betaz-hat was biased upward by about +0.1 on logit(0.2) = -1.386.
#
# Cause: the fit maximised the Laplace marginal with the EXPECTED Fisher count weight in the
# log-determinant (the #739 defect, present at the reported commit 033d4e1d). That surface is
# not the likelihood: its maximiser sits at larger betaz. #739 switched the log-determinant to
# the observed curvature. Measured on 24 replicates of the #926 design (ZIP, pi = 0.2, p = 10,
# n = 100, K = 2), against the exact likelihood maximised by adaptive Gauss-Hermite quadrature:
# the Fisher-weight betaz-hat sat +0.166 (SE 0.003) above the exact MLE, positive in every
# species x replicate; the observed-curvature betaz-hat sat +0.008 from it, and the Wald betaz
# coverage among converged fits was 0.959 (220 intervals, mean SE 0.357, empirical SD 0.365).
#
# This test pins that consequence on the issue's own replicate (seed 20301017; the issue's
# generator, data written out so the test does not depend on an RNG stream). The reference is
# independent of the package's Laplace code: the exact per-site integral by 12 x 12 adaptive
# Gauss-Hermite quadrature (centre and scale from a ForwardDiff Newton mode at the fitted
# values), maximised over betaz with betac and Lambda held at the fit.
using Test, GLLVModels, ForwardDiff, LinearAlgebra, Optim, SpecialFunctions
const _G926 = GLLVModels

const _Y926 = [
    3 4 1 0 0 4 11 2 4 0 4 2 0 2 0 4 0 3 1 3 0 0 1 1 0 1 2 0 4 5 0 0 2 0 5 1 1 0 1 5 3 2 0 0 1 4 1 0 0 0 0 3 0 3 3 0 0 3 3 2 1 3 2 2 8 0 0 0 3 0 3 1 0 5 6 1 5 0 5 6 0 0 4 3 0 4 0 0 2 2 1 5 5 0 5 2 4 1 1 0;
    2 3 1 1 4 0 6 2 0 0 8 0 0 10 2 7 0 0 3 0 0 1 2 0 5 1 1 4 2 0 1 0 7 4 3 8 1 0 3 0 0 1 2 0 0 1 0 0 2 1 0 3 1 0 2 2 2 2 13 2 0 1 3 0 0 1 0 0 1 3 0 0 1 0 0 1 2 0 5 1 4 1 4 4 0 9 0 7 3 7 6 1 13 0 5 1 1 1 0 1;
    2 2 3 0 9 0 3 0 0 7 2 1 2 7 5 0 0 4 0 2 2 0 1 1 0 6 4 8 1 1 1 21 8 2 1 0 1 6 2 0 4 2 2 2 0 1 0 4 0 2 1 1 2 1 0 4 4 1 2 3 2 4 0 0 1 5 3 3 0 0 1 1 1 1 0 0 0 1 0 1 2 2 4 1 1 3 2 1 2 7 2 1 0 3 1 0 1 2 3 2;
    2 2 0 0 1 15 15 3 9 1 6 2 2 0 0 0 0 1 0 2 0 5 0 2 1 0 1 1 3 10 3 1 2 4 7 3 1 2 0 0 1 1 3 1 0 5 2 1 1 2 1 4 0 7 2 2 1 11 3 3 0 0 3 1 2 5 3 3 4 2 8 0 0 3 12 0 0 0 3 2 2 3 0 1 4 0 3 1 0 1 1 2 6 0 4 3 6 5 0 0;
    0 0 1 1 11 0 9 4 0 4 6 2 0 13 7 6 0 1 1 0 2 0 6 0 2 1 3 1 6 1 0 1 4 1 0 0 0 4 0 6 6 3 6 1 1 0 1 0 0 0 1 2 4 0 2 0 5 5 0 7 0 5 6 0 2 0 0 0 0 5 0 0 2 5 2 4 0 3 3 0 1 0 11 11 0 0 0 0 4 15 13 0 0 0 2 4 3 4 4 4;
    6 0 0 3 17 1 0 0 0 12 4 0 0 8 8 0 0 1 0 3 4 0 6 0 10 0 9 7 0 0 0 23 0 0 1 0 0 0 0 0 4 7 6 5 0 0 0 4 8 4 2 2 2 0 3 2 0 1 3 6 0 0 0 5 0 1 5 3 0 0 1 4 0 2 1 2 2 0 3 0 0 2 5 1 0 1 6 0 2 9 3 0 0 4 0 1 3 1 8 0;
    5 3 0 0 0 13 20 7 8 2 3 1 1 2 1 0 23 0 2 3 2 14 2 3 0 1 3 0 7 13 3 2 1 8 4 4 1 1 2 3 0 2 5 2 0 3 4 0 0 0 0 8 5 8 5 4 0 14 11 2 0 1 5 0 10 2 1 2 5 6 8 2 0 7 0 2 3 7 11 9 3 2 1 7 0 2 0 0 0 3 2 2 9 4 13 8 12 0 0 1;
    3 5 0 0 3 12 3 3 7 0 2 7 6 0 1 0 15 2 1 1 6 14 5 4 2 3 2 4 2 17 8 1 1 5 3 4 0 3 1 5 1 1 2 0 12 6 3 3 3 1 6 3 0 12 7 2 3 2 0 1 5 2 5 2 6 6 0 2 11 4 0 5 3 0 11 4 3 10 9 2 3 4 2 3 12 3 5 3 0 3 2 5 4 3 7 6 7 6 0 0;
    0 0 0 8 0 0 2 5 0 7 1 0 3 0 8 7 0 4 4 4 6 2 5 5 12 6 11 10 0 2 3 19 18 4 3 0 1 9 6 8 3 9 0 0 3 0 1 3 5 9 2 0 5 3 4 11 6 0 6 0 3 9 3 0 0 0 0 8 0 3 0 6 2 0 2 0 0 2 0 0 0 2 0 6 3 5 0 2 6 14 0 0 4 6 3 0 0 0 3 6;
    2 8 3 4 5 3 21 3 0 0 7 0 1 9 3 14 4 3 4 5 3 2 8 0 9 1 0 8 4 0 0 2 6 6 0 0 0 1 4 11 5 2 5 2 3 2 5 2 3 9 2 6 4 3 5 5 0 16 19 7 4 6 0 0 7 0 3 5 3 5 4 0 0 0 4 1 9 0 4 6 4 4 0 13 3 8 1 7 2 12 6 4 16 2 9 4 0 2 0 6]

_zip_logf926(y, ηz, ηc) = (π = 1 / (1 + exp(-ηz)); μ = exp(ηc);
    y > 0 ? log1p(-π) + y * ηc - μ - SpecialFunctions.logfactorial(y) : log(π + (1 - π) * exp(-μ)))

function _gh926(N)
    E = eigen(SymTridiagonal(zeros(N), sqrt.((1:(N - 1)) ./ 2)))
    return E.values, sqrt(pi) .* E.vectors[1, :] .^ 2
end

# Per site: nodes z_k = ẑ + L⁻ᵀ √2 x_k and log-weights, from the joint mode at (βz0, βc, Λ).
# The nodes are fixed while βz moves, so the quadrature is a smooth function of βz.
function _aghq_nodes926(Y, Λ, βz0, βc; N = 12)
    p, K = size(Λ); x, w = _gh926(N)
    map(axes(Y, 2)) do s
        y = Y[:, s]
        q(z) = sum(_zip_logf926(y[t], βz0[t], βc[t] + dot(Λ[t, :], z)) for t in 1:p) - 0.5 * dot(z, z)
        z = zeros(K)
        for _ in 1:100
            step = -(ForwardDiff.hessian(q, z) \ ForwardDiff.gradient(q, z))
            z += step
            norm(step) < 1e-12 && break
        end
        L = cholesky(Symmetric(-ForwardDiff.hessian(q, z))).L
        nodes = [z + L' \ (sqrt(2) .* [x[i], x[j]]) for i in 1:N for j in 1:N]
        lw = [log(w[i]) + log(w[j]) + x[i]^2 + x[j]^2 - 0.5 * dot(n, n) - K / 2 * log(pi) - logdet(L)
              for (n, (i, j)) in zip(nodes, ((i, j) for i in 1:N for j in 1:N))]
        ηc = [βc .+ Λ * n for n in nodes]
        (y, ηc, lw)
    end
end

function _aghq_loglik926(sites, βz)
    tot = 0.0
    for (y, ηc, lw) in sites
        v = [lw[k] + sum(_zip_logf926(y[t], βz[t], ηc[k][t]) for t in eachindex(y)) for k in eachindex(lw)]
        m = maximum(v); tot += m + log(sum(exp.(v .- m)))
    end
    return tot
end

@testset "#926 ZIP betaz-hat is the exact-likelihood betaz, not the Fisher-Laplace one" begin
    fit = _G926.fit_zip_gllvm(_Y926; K = 2)
    @test fit.converged
    sites = _aghq_nodes926(_Y926, fit.Λc, fit.βz, fit.βc)
    res = Optim.optimize(b -> -_aghq_loglik926(sites, b), copy(fit.βz), LBFGS(),
                         Optim.Options(g_tol = 1e-7); autodiff = :forward)
    βz_exact = Optim.minimizer(res)
    @test Optim.converged(res)
    # fit - exact, measured: mean -0.004, max |.| 0.082 (species 2, betaz-hat -3.15 with a Wald
    # SE of 1.2). The Fisher-weight fit (`hessian = :fisher`, the 033d4e1d objective) against
    # the same reference: mean +0.185, every species between +0.088 and +0.334.
    @test abs(sum(fit.βz .- βz_exact) / length(βz_exact)) < 0.03
    @test maximum(abs.(fit.βz .- βz_exact)) < 0.15

    # The Wald interval is centred on that betaz-hat (it covers the truth logit(0.2) for 8 of
    # the 10 species here; the issue reported 7 of 10 at 033d4e1d).
    ci = confint(fit, _Y926; method = :wald)
    terms = String.(ci.term)
    iz = [findfirst(==("betaz[$t]"), terms) for t in 1:10]
    @test ci.pd_hessian
    @test ci.estimate[iz] ≈ fit.βz
    @test all(ci.lower[iz] .< fit.βz .< ci.upper[iz])
end
