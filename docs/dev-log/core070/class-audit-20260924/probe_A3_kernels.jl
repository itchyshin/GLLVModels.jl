using GLLVModels, LinearAlgebra, Random, Distributions, Optim, ForwardDiff
const G = GLLVModels
include("/tmp/claude-503/audit/probe_common.jl")
rng = MersenneTwister(777)
rsc() = rand(rng, (0.5, 1.0, 2.0, 3.0))
# ---- mixed-family kernel (bridge family vector)
probe("mixed _mixed_laplace_mode (Pois/Binom/Beta/NB2)", 1200, () -> begin
    p = 8; K = rand(rng, 1:3); sc = rsc()
    fams = Any[Poisson(), Poisson(), Binomial(), Binomial(), Beta(8.0, 1.0), Beta(8.0, 1.0), NegativeBinomial(2.0, 0.5), NegativeBinomial(2.0, 0.5)]
    links = Any[LogLink(), LogLink(), LogitLink(), LogitLink(), LogitLink(), LogitLink(), LogLink(), LogLink()]
    Λ = sc .* randn(rng, p, K); β = randn(rng, p) .* 0.7; zt = randn(rng, K)
    y = Any[]
    for t in 1:p
        η = β[t] + dot(Λ[t, :], zt)
        if fams[t] isa Poisson; push!(y, rand(rng, Poisson(exp(clamp(η, -4, 5)))))
        elseif fams[t] isa Binomial; push!(y, rand(rng) < 1/(1+exp(-η)) ? 1 : 0)
        elseif fams[t] isa Beta; m = clamp(1/(1+exp(-η)), 1e-3, 1-1e-3); push!(y, clamp(rand(rng, Beta(m*8, (1-m)*8)), 1e-4, 1-1e-4))
        else; μ = exp(clamp(η, -4, 5)); push!(y, rand(rng, NegativeBinomial(2.0, 2.0/(2.0+μ)))) end
    end
    yv = [float(v) for v in y]; n = ones(Int, p)
    Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K))
    q = z -> begin
        s = -0.5 * dot(z, z)
        for t in 1:p
            ηt = G._clamp_eta(β[t] + dot(Λe[t, :], z)); μt = G._clamp_mu(fams[t], G.linkinv(links[t], ηt))
            s += G._glm_logpdf(fams[t], μt, 1, yv[t])
        end
        s
    end
    (K, () -> G._mixed_laplace_mode(fams, links, yv, n, Λe, β), q, () -> G._mixed_loglik_site(fams, links, yv, n, Λe, β))
end)
# ---- Tweedie grouped kernel (disp_group route) – replicate loop
probe("Tweedie grouped kernel", 300, () -> begin
    p = rand(rng, 4:8); K = rand(rng, 1:2); sc = rsc(); pw = rand(rng, (1.2, 1.5, 1.8))
    φ = exp.(randn(rng, p) .* 0.5); fams = [G.TweedieED(φ[t], pw) for t in 1:p]
    Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
    y = map(1:p) do t   # compound Poisson-Gamma draw
        μ = exp(clamp(β[t] + dot(Λ[t, :], zt), -4, 4)); λ = μ^(2 - pw) / (φ[t] * (2 - pw))
        a = (2 - pw) / (pw - 1); sc_g = φ[t] * (pw - 1) * μ^(pw - 1)
        Nn = rand(rng, Poisson(λ)); Nn == 0 ? 0.0 : sum(rand(rng, Gamma(a, sc_g)) for _ in 1:Nn)
    end
    n = ones(Int, p); Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K))
    q = z -> sum(G.tweedie_logpdf(y[t], exp(G._clamp_eta(β[t] + dot(Λe[t, :], z))), φ[t], pw) for t in 1:p) - 0.5 * dot(z, z)
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
    (K, zk, q, () -> G._tweedie_grouped_loglik_site(fams, y, n, Λe, β, LogLink()))
end)
