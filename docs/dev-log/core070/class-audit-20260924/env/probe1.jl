using Pkg; Pkg.activate(@__DIR__; io=devnull)
using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels

# generic mode residual for single-family pieces
function resid_core(fam, y, n, Λ, η0, link, z)
    η = G._clamp_eta.(η0 .+ Λ*z)
    μ = G._clamp_mu.(Ref(fam), G.linkinv.(Ref(link), η))
    me = G.mu_eta.(Ref(link), η)
    s = G._glm_score.(Ref(fam), μ, n, me, y)
    maximum(abs, Λ'*s .- z)
end

function stress(fam, link, sampler; p=10, K=2, nsite=400, scale=2.5, seed=1)
    rng = MersenneTwister(seed)
    Λ = scale .* randn(rng, p, K); β = randn(rng, p)
    n = ones(Int, p)
    bad_off = 0; bad_core = 0; worst = 0.0; gapmax = 0.0; fin_bad_off = 0
    for s in 1:nsite
        z0 = randn(rng, K)
        η = β .+ Λ*z0
        y = [sampler(rng, G.linkinv(link, clamp(η[t], -30, 30))) for t in 1:p]
        zo = G._laplace_mode_off(fam, y, n, Λ, β, link)
        zc = G._laplace_mode(fam, y, n, Λ, β, link)
        ro = resid_core(fam, y, n, Λ, β, link, zo)
        rc = resid_core(fam, y, n, Λ, β, link, zc)
        vo = G._laplace_site_off(fam, y, n, Λ, β, link; hessian=:fisher)
        vc = G.laplace_loglik_site(fam, y, n, Λ, β, link; hessian=:fisher)
        if ro > 1e-4
            bad_off += 1
            isfinite(vo) && (fin_bad_off += 1)
            worst = max(worst, ro); gapmax = max(gapmax, abs(vo - vc))
        end
        rc > 1e-4 && (bad_core += 1)
    end
    println(rpad(string(nameof(typeof(fam))),10), " scale=", scale,
      "  _laplace_mode_off nonconverged=", bad_off, "/", nsite, " (finite value in ", fin_bad_off, ")",
      "  core _laplace_mode nonconverged=", bad_core, "/", nsite,
      "  worst resid=", round(worst, sigdigits=3), "  max |v_off - v_core|=", round(gapmax, sigdigits=3))
end

for sc in (1.0, 2.5, 4.0)
    stress(Poisson(), G.LogLink(), (rng, μ) -> rand(rng, Poisson(min(μ, 1e6))); scale=sc)
    stress(Binomial(), G.LogitLink(), (rng, μ) -> rand(rng, Bernoulli(μ)); scale=sc)
end
