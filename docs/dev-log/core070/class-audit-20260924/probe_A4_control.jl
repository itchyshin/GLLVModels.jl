using GLLVModels, LinearAlgebra, Random, Distributions, Optim, ForwardDiff
const G = GLLVModels
include("/tmp/claude-503/audit/probe_common.jl")
rng = MersenneTwister(99)
rsc() = rand(rng, (0.5, 1.0, 2.0, 3.0))
for (fam, link, nm, draw) in (
    (Poisson(), LogLink(), "generic _laplace_mode Poisson (backtrack)", μ -> rand(rng, Poisson(μ))),
    (Binomial(), LogitLink(), "generic _laplace_mode Binomial (backtrack)", μ -> rand(rng) < μ/(1+μ) ? 1 : 0),
    (NegativeBinomial(2.0, 0.5), LogLink(), "generic _laplace_mode NB2 (backtrack)", μ -> rand(rng, NegativeBinomial(2.0, 2.0/(2.0+μ)))),
    (Gamma(2.0, 1.0), LogLink(), "generic _laplace_mode Gamma (backtrack)", μ -> rand(rng, Gamma(2.0, μ/2))))
    probe(nm, 1500, () -> begin
        p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rsc()
        Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
        y = [draw(exp(clamp(β[t] + dot(Λ[t, :], zt), -5, 6))) for t in 1:p]
        Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K)); n = ones(Int, p)
        q = z -> sum(G._glm_logpdf(fam, G._clamp_mu(fam, G.linkinv(link, G._clamp_eta(β[t] + dot(Λe[t, :], z)))), 1, y[t]) for t in 1:p) - 0.5 * dot(z, z)
        (K, () -> G._laplace_mode(fam, y, n, Λe, β, link), q, () -> G.laplace_loglik_site(fam, y, n, Λe, β, link))
    end)
end
