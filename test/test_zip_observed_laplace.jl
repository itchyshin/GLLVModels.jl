# #739: fit_zip_gllvm's logLik sat below gllvmTMB's zi_poisson() on simulated ZIP data.
#
# Cause: the Laplace log-determinant for ZIPoisson used the EXPECTED Fisher count weight even
# though `hessian = :observed` (the default) promises the observed one, so `:observed` and
# `:fisher` gave the identical objective and the fit maximised a different surface from
# TMB's (which differentiates the joint negative log-likelihood). The reference below is
# independent of the package's mode search and weight code: Newton with ForwardDiff on the
# joint site log-posterior, then q(ẑ) - ½ logdet(-∇²q(ẑ)).
using Test, GLLVModels, ForwardDiff, LinearAlgebra
const _G739 = GLLVModels

# log f(y | ηz, ηc) for the zero-inflated Poisson, written out here (not via `_tp_pieces`).
function _zip_logf739(y, ηz, ηc)
    π = 1 / (1 + exp(-ηz)); μ = exp(ηc)
    return y > 0 ? log1p(-π) + y * ηc - μ - lgamma739(y + 1) : log(π + (1 - π) * exp(-μ))
end
lgamma739(x) = sum(log, 1:(Int(x) - 1); init = 0.0)      # log Γ(x) for integer x ≥ 1

function _zip_site_exact739(y, Λ, βz, βc)
    p, K = size(Λ)
    q(z) = sum(_zip_logf739(y[t], βz[t], βc[t] + dot(Λ[t, :], z)) for t in 1:p) - 0.5 * dot(z, z)
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

# Fixed small design (p = 5 species, n = 8 sites, K = 2): counts with zeros and sizeable means.
const _Y739 = Float64[
    0 3 0 5 2 0 7 1;
    4 0 0 2 6 1 0 3;
    0 0 6 1 0 4 2 0;
    2 5 1 0 3 0 4 6;
    1 0 3 4 0 2 0 5]
const _Λ739 = [0.5 0.0; -0.3 0.4; 0.2 -0.5; 0.4 0.3; -0.2 0.2]
const _βz739 = [-1.5, -1.0, -1.2, -0.8, -1.4]
const _βc739 = [0.9, 1.0, 0.8, 1.1, 0.7]

@testset "#739 ZIP Laplace log-determinant uses the observed count curvature" begin
    exact = sum(_zip_site_exact739(_Y739[:, s], _Λ739, _βz739, _βc739) for s in axes(_Y739, 2))

    @testset ":observed matches the independent exact-Hessian Laplace" begin
        obs = _G739.zip_marginal_loglik_laplace(_Y739, _Λ739, _βz739, _βc739; hessian = :observed)
        @test obs ≈ exact atol = 1e-7
    end

    @testset "the default is :observed" begin
        @test _G739.zip_marginal_loglik_laplace(_Y739, _Λ739, _βz739, _βc739) ==
              _G739.zip_marginal_loglik_laplace(_Y739, _Λ739, _βz739, _βc739; hessian = :observed)
    end

    @testset ":fisher is a different surface (the selector is effective)" begin
        fis = _G739.zip_marginal_loglik_laplace(_Y739, _Λ739, _βz739, _βc739; hessian = :fisher)
        @test abs(fis - exact) > 1e-3
    end

    @testset "agrees with the gllvmTMB-named zi_poisson() evaluator" begin
        twin = _G739.zi_marginal_loglik_laplace(_G739.zi_poisson(), _Y739, _Λ739, _βc739, _βz739)
        obs = _G739.zip_marginal_loglik_laplace(_Y739, _Λ739, _βz739, _βc739)
        @test obs ≈ twin atol = 1e-8
    end
end

# Fit level. p = 6, n = 50, K = 1, true ZI probability 0.25 (simulated once; literal data so the
# test does not depend on an RNG stream). On unmodified main `fit_zip_gllvm` ends at -663.283
# against -659.541 for the zi_poisson() route (gap 3.74, both converged); it maximised the
# Fisher-weight surface, whose maximiser has a higher mean ZI estimate. Both routes now
# maximise the same observed-curvature Laplace, so they reach the same optimum.
const _Y739FIT = Float64[
    0 1 28 5 4 3 11 19 5 14 5 0 0 2 5 9 18 6 4 4 7 6 5 4 5 0 14 6 0 0 4 14 9 12 0 3 18 13 4 0 2 8 0 16 3 11 10 4 6 4;
    6 7 1 8 0 1 0 1 5 6 7 0 15 1 0 2 3 0 7 15 1 2 7 0 5 3 1 6 4 0 0 3 4 3 5 0 1 2 6 7 16 8 8 0 6 0 4 2 2 4;
    1 0 10 0 4 0 4 0 0 3 0 2 0 2 1 2 0 2 2 1 1 0 0 1 1 1 11 2 2 0 0 3 3 3 1 2 0 3 0 4 0 0 1 5 0 0 0 4 2 0;
    2 3 0 4 5 0 0 53 2 0 0 1 0 10 7 8 20 3 6 0 7 0 3 0 0 7 73 2 1 4 0 0 0 20 0 0 0 39 3 8 1 0 6 32 3 7 0 0 6 7;
    5 5 1 0 3 0 3 1 8 0 1 0 0 0 0 3 2 3 3 0 3 0 0 19 4 1 1 4 4 7 4 2 0 2 1 14 0 1 0 2 7 7 1 0 4 3 1 1 2 7;
    4 9 0 5 2 5 4 3 6 0 4 4 4 1 0 0 3 0 6 4 5 0 4 0 2 5 1 2 2 0 1 6 5 4 4 0 7 5 5 4 4 0 4 6 3 4 3 0 0 2]

@testset "#739 fit_zip_gllvm reaches the zi_poisson() optimum" begin
    zip = _G739.fit_zip_gllvm(_Y739FIT; K = 1)
    twin = _G739.fit_gllvm(_Y739FIT; family = _G739.zi_poisson(), K = 1)
    @test zip.converged && twin.converged
    @test zip.loglik ≈ twin.loglik atol = 1e-3
    @test zip.loglik > -660.0                    # main: -663.283
end

@testset "#739 :fisher is unchanged and unguarded" begin
    f = _G739.fit_zip_gllvm(_Y739FIT; K = 1, hessian = :fisher)
    @test f.converged
    # The Fisher-surface optimum sits below the observed-surface one on this data.
    @test f.loglik ≈ -663.283 atol = 5e-3
end
