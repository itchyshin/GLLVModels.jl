using GLLVModels, LinearAlgebra, Random, Distributions, Optim, ForwardDiff
const G = GLLVModels
include("/tmp/claude-503/audit/probe_common.jl")
rng = MersenneTwister(31337)
rsc() = rand(rng, (0.5, 1.0, 2.0, 3.0))
# NB2 grouped kernel (excluded fitters' kernel, grouped_dispersion.jl:39-86) - replicate loop
probe("NB2 grouped kernel (note only; excluded)", 1500, () -> begin
    p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rsc(); r = exp.(randn(rng, p) .+ 1)
    fams = [NegativeBinomial(r[t], 0.5) for t in 1:p]; n = ones(Int, p)
    Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
    y = [rand(rng, NegativeBinomial(r[t], r[t] / (r[t] + exp(clamp(β[t] + dot(Λ[t, :], zt), -5, 6))))) for t in 1:p]
    Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K))
    q = z -> sum(G._glm_logpdf(fams[t], G._clamp_mu(fams[t], exp(G._clamp_eta(β[t] + dot(Λe[t, :], z)))), 1, y[t]) for t in 1:p) - 0.5 * dot(z, z)
    zk = () -> begin
        z = zeros(K)
        for _ in 1:100
            η = G._clamp_eta.(β .+ Λe * z); μ = G._clamp_mu.(fams, exp.(η)); me = μ
            s = G._glm_score.(fams, μ, n, me, y); W = G._glm_weight.(fams, μ, n, me)
            Δ = G._safe_solve(Symmetric(Λe' * (W .* Λe) + I), Λe' * s .- z)
            (Δ === nothing || !all(isfinite, Δ)) && break
            z = z .+ Δ; maximum(abs, Δ) < 1e-9 && break
        end
        z
    end
    (K, zk, q, () -> G._nb_grouped_loglik_site(fams, y, n, Λe, β, LogLink()))
end)
