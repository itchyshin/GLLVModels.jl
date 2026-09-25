using GLLVModels, LinearAlgebra, Random, Distributions, Optim, ForwardDiff
const G = GLLVModels
rng = MersenneTwister(424242)
cl(η) = G._clamp_eta(η)
function refmode(q, K, zk)
    best = nothing
    for z0 in (zeros(K), zk isa AbstractVector && all(isfinite, zk) ? clamp.(zk, -50, 50) : zeros(K))
        r = try Optim.optimize(z -> -q(z), z0, Optim.BFGS(), Optim.Options(g_tol=1e-10, iterations=2000); autodiff=:forward) catch; nothing end
        r === nothing && continue
        (best === nothing || Optim.minimum(r) < Optim.minimum(best)) && (best = r)
    end
    return best === nothing ? nothing : Optim.minimizer(best)
end
gnorm(q, z) = try maximum(abs, ForwardDiff.gradient(q, z)) catch; Inf end
function probe(name, ntr, gen)
    nbad = 0; nfin = 0; nband = 0; worst = 0.0; ex = nothing
    for _ in 1:ntr
        K, zk_fn, q, v_fn = gen()
        zk = zk_fn()
        gk = gnorm(q, zk)
        gk > 1e-4 || continue
        zr = refmode(q, K, zk); zr === nothing && continue
        gr = gnorm(q, zr); gr < 1e-5 || continue
        nbad += 1
        v = try v_fn() catch; NaN end
        if isfinite(v)
            nfin += 1
            dq = q(zr) - q(zk)
            abs(v) < 1e11 && (nband += 1)
            if abs(v) < 1e11 && dq > worst
                worst = dq; ex = (K=K, grad_at_kernel_z=gk, logpost_gap=dq, site_value=v, zk=round.(zk; sigdigits=4), zref=round.(zr; sigdigits=4))
            end
        end
    end
    println(rpad(name, 44), " non-mode returns: $nbad/$ntr; finite site value: $nfin; finite & |v|<1e11 (escapes sentinel): $nband; worst logpost gap in-band: $(round(worst; sigdigits=4))")
    ex === nothing || println("    worst example: ", ex)
end
rsc() = rand(rng, (0.5, 1.0, 2.0, 3.0))
# ---- two-part (ZIP / ZINB / HurdlePoisson / DeltaGamma): _twopart_mode
for (fam, nm, draw) in (
    (G.ZIPoisson(), "twopart ZIP", (π, μ) -> rand(rng) < π ? 0 : rand(rng, Poisson(μ))),
    (G.ZINB(2.0), "twopart ZINB(r=2)", (π, μ) -> rand(rng) < π ? 0 : rand(rng, NegativeBinomial(2.0, 2.0/(2.0+μ)))),
    (G.HurdlePoisson(), "twopart HurdlePoisson", (π, μ) -> rand(rng) < π ? 0 : (x = 0; while x == 0; x = rand(rng, Poisson(max(μ,1e-3))); end; x)),
    (G.DeltaGamma(2.0), "twopart DeltaGamma(α=2)", (π, μ) -> rand(rng) < π ? 0.0 : rand(rng, Gamma(2.0, μ/2.0))))
    probe(nm, 1500, () -> begin
        p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rsc()
        Λc = sc .* randn(rng, p, K); Λz = zeros(p, K); βz = randn(rng, p); βc = randn(rng, p)
        zt = randn(rng, K)
        y = [draw(1/(1+exp(-βz[t])), exp(clamp(βc[t] + dot(Λc[t, :], zt), -5, 6))) for t in 1:p]
        Λe = Λc .* (1 .+ 0.5 .* randn(rng, p, K))
        q = z -> sum(G._tp_pieces(fam, y[t], cl(βz[t]), cl(βc[t] + dot(Λe[t, :], z)))[5] for t in 1:p) - 0.5 * dot(z, z)
        (K, () -> G._twopart_mode(fam, y, Λz, Λe, βz, βc), q, () -> G.twopart_loglik_site(fam, y, Λz, Λe, βz, βc))
    end)
end
# ---- COMPoisson: _compoisson_mode
probe("COMPoisson _compoisson_mode", 800, () -> begin
    p = rand(rng, 4:10); K = rand(rng, 1:2); sc = rsc(); ν = exp(randn(rng) * 0.7)
    Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
    y = [rand(rng, Poisson(exp(clamp(β[t] + dot(Λ[t, :], zt), -4, 5)))) for t in 1:p]
    Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K))
    q = z -> sum(G.compoisson_logpdf(y[t], β[t] + dot(Λe[t, :], z), ν) for t in 1:p) - 0.5 * dot(z, z)
    (K, () -> G._compoisson_mode(y, Λe, β, ν), q, () -> G._compoisson_loglik_site(y, Λe, β, ν))
end)
# ---- OrderedBeta: _ordered_beta_mode
probe("OrderedBeta _ordered_beta_mode", 1500, () -> begin
    p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rsc(); φ = exp(randn(rng) + 1.5); c0 = -2.0; c1 = 2.0
    Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
    y = map(1:p) do t
        η = β[t] + dot(Λ[t, :], zt); u = rand(rng)
        p0 = 1 - 1/(1+exp(-(η - c0))); p1 = 1/(1+exp(-(η - c1)))
        u < p0 ? 0.0 : u < p0 + p1 ? 1.0 : clamp(rand(rng, Beta(max(1/(1+exp(-η)),1e-3)*φ, max(1-1/(1+exp(-η)),1e-3)*φ)), 1e-4, 1-1e-4)
    end
    Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K))
    q = z -> sum(G.ordered_beta_logp(y[t], β[t] + dot(Λe[t, :], z), c0, c1, φ) for t in 1:p) - 0.5 * dot(z, z)
    (K, () -> G._ordered_beta_mode(y, Λe, β, c0, c1, φ), q, () -> G._ordered_beta_loglik_site(y, Λe, β, c0, c1, φ))
end)
# ---- BetaBinomial: _beta_binomial_mode (per-trait φ vector, as the grouped default)
probe("BetaBinomial _beta_binomial_mode", 1500, () -> begin
    p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rsc(); φ = exp.(randn(rng, p) .+ 1.0)
    Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K); N = fill(rand(rng, (3, 10, 30)), p)
    y = [rand(rng, BetaBinomial(N[t], max(1/(1+exp(-(β[t]+dot(Λ[t,:],zt)))),1e-3)*φ[t], max(1-1/(1+exp(-(β[t]+dot(Λ[t,:],zt)))),1e-3)*φ[t])) for t in 1:p]
    Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K))
    q = z -> sum(G.betabinomial_logp(y[t], β[t] + dot(Λe[t, :], z), N[t], φ[t]) for t in 1:p) - 0.5 * dot(z, z)
    (K, () -> G._beta_binomial_mode(y, N, Λe, β, φ), q, () -> G._beta_binomial_loglik_site(y, N, Λe, β, φ))
end)
# ---- Student-t shared (generic _laplace_mode, NOT in backtrack list) and grouped kernel loop
function tdraw(p, K, sc)
    ν = rand(rng, (1.5, 3.0, 5.0)); σ = exp(randn(rng) * 0.5)
    Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
    y = [β[t] + dot(Λ[t, :], zt) + σ * rand(rng, TDist(ν)) * (rand(rng) < 0.15 ? 20 : 1) for t in 1:p]
    return ν, σ, Λ .* (1 .+ 0.5 .* randn(rng, p, K)), β, y
end
probe("StudentT shared: generic _laplace_mode", 1500, () -> begin
    p = rand(rng, 4:12); K = rand(rng, 1:3); ν, σ, Λe, β, y = tdraw(p, K, rsc())
    f = G.StudentTFamily(ν, σ); n = ones(Int, p)
    q = z -> sum(G._glm_logpdf(f, β[t] + dot(Λe[t, :], z), 1, y[t]) for t in 1:p) - 0.5 * dot(z, z)
    (K, () -> G._laplace_mode(f, y, n, Λe, β, IdentityLink()), q, () -> G.laplace_loglik_site(f, y, n, Λe, β, IdentityLink()))
end)
probe("StudentT grouped kernel (default route)", 1500, () -> begin
    p = rand(rng, 4:12); K = rand(rng, 1:3); ν, σ, Λe, β, y = tdraw(p, K, rsc())
    fams = [G.StudentTFamily(ν, σ * exp(0.3 * randn(rng))) for _ in 1:p]; n = ones(Int, p)
    q = z -> sum(G._glm_logpdf(fams[t], β[t] + dot(Λe[t, :], z), 1, y[t]) for t in 1:p) - 0.5 * dot(z, z)
    zk = () -> begin
        z = zeros(K)
        for _ in 1:100
            η = cl.(β .+ Λe * z); μ = G._clamp_mu.(fams, η); me = ones(p)
            s = G._glm_score.(fams, μ, n, me, y); W = G._glm_weight.(fams, μ, n, me)
            Δ = G._safe_solve(Symmetric(Λe' * (W .* Λe) + I), Λe' * s .- z)
            (Δ === nothing || !all(isfinite, Δ)) && break
            z = z .+ Δ; maximum(abs, Δ) < 1e-9 && break
        end
        z
    end
    (K, zk, q, () -> G._studentt_grouped_loglik_site(fams, y, n, Λe, β, IdentityLink()))
end)
# ---- GP1: generic _laplace_mode (no backtracking)
probe("GP1 generic _laplace_mode", 1000, () -> begin
    p = rand(rng, 4:10); K = rand(rng, 1:2); sc = rsc(); α = rand(rng, (-0.05, 0.1, 0.5, 1.0))
    Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
    y = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(clamp(β[t] + dot(Λ[t, :], zt), -4, 4))))) for t in 1:p]
    α < 0 && (y = min.(y, floor(Int, -1/α - 1)))
    Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K)); f = G.GeneralizedPoisson1(α); n = ones(Int, p)
    q = z -> sum(G._glm_logpdf(f, G._clamp_mu(f, exp(cl(β[t] + dot(Λe[t, :], z)))), 1, y[t]) for t in 1:p) - 0.5 * dot(z, z)
    (K, () -> G._laplace_mode(f, y, n, Λe, β, LogLink()), q, () -> G.laplace_loglik_site(f, y, n, Λe, β, LogLink()))
end)
# ---- Ordinal per-trait: _ordinal_laplace_mode_pertrait
probe("Ordinal per-trait _ordinal_laplace_mode_pertrait", 1500, () -> begin
    p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rsc(); C = fill(4, p)
    τ = repeat([-1.0 0.0 1.0], p); Λ = sc .* randn(rng, p, K); β = zeros(p); zt = randn(rng, K)
    y = [begin η = dot(Λ[t, :], zt) + randn(rng) * 0.3; u = rand(rng); F = [1/(1+exp(-(τ[t, j] - η))) for j in 1:3]; u < F[1] ? 1 : u < F[2] ? 2 : u < F[3] ? 3 : 4 end for t in 1:p]
    Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K))
    q = z -> sum(log(max(G._ord_prob(y[t], cl(β[t] + dot(Λe[t, :], z)), view(τ, t, 1:3), LogitLink()), 1e-12)) for t in 1:p) - 0.5 * dot(z, z)
    (K, () -> G._ordinal_laplace_mode_pertrait(y, Λe, β, τ, C, LogitLink()), q, () -> G.ordinal_loglik_site_pertrait(y, Λe, β, τ, C, LogitLink()))
end)
