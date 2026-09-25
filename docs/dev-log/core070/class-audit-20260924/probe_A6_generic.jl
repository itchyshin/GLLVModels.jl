using GLLVModels, LinearAlgebra, Random, Distributions, Optim, ForwardDiff, Statistics
const G = GLLVModels
rng = MersenneTwister(2024)
rsc() = rand(rng, (0.5, 1.0, 2.0, 3.0))
function run(fam, link, nm, draw; ntr = 1500)
    gaps = Float64[]; nbad = 0; nsig = 0
    for _ in 1:ntr
        p = rand(rng, 4:12); K = rand(rng, 1:3); sc = rsc()
        Λ = sc .* randn(rng, p, K); β = randn(rng, p); zt = randn(rng, K)
        y = [draw(G.linkinv(link, clamp(β[t] + dot(Λ[t, :], zt), -5, 6))) for t in 1:p]
        Λe = Λ .* (1 .+ 0.5 .* randn(rng, p, K)); n = ones(Int, p)
        q = zz -> sum(G._glm_logpdf(fam, G._clamp_mu(fam, G.linkinv(link, G._clamp_eta(β[t] + dot(Λe[t, :], zz)))), 1, y[t]) for t in 1:p) - 0.5 * dot(zz, zz)
        zk = G._laplace_mode(fam, y, n, Λe, β, link)
        gk = maximum(abs, ForwardDiff.gradient(q, zk)); gk > 1e-4 || continue
        r = Optim.optimize(zz -> -q(zz), zk, Optim.Newton(), Optim.Options(g_tol = 1e-10); autodiff = :forward)
        zr = Optim.minimizer(r); maximum(abs, ForwardDiff.gradient(q, zr)) < 1e-6 || continue
        v = G.laplace_loglik_site(fam, y, n, Λe, β, link)
        nbad += 1; isfinite(v) && abs(v) < 1e11 && (push!(gaps, q(zr) - q(zk)); q(zr) - q(zk) > 1e-2 && (nsig += 1))
    end
    println(rpad(nm, 34), " non-mode (unconverged) returns: $nbad/$ntr; in-band finite: $(length(gaps)); logpost gap >0.01: $nsig; median gap: $(isempty(gaps) ? NaN : round(median(gaps); sigdigits=3)); max gap: $(isempty(gaps) ? NaN : round(maximum(gaps); sigdigits=3))")
end
run(Gamma(2.0, 1.0), LogLink(), "Gamma(α=2)/log generic", μ -> rand(rng, Gamma(2.0, μ/2)))
run(Gamma(0.5, 1.0), LogLink(), "Gamma(α=0.5)/log generic", μ -> rand(rng, Gamma(0.5, μ/0.5)))
run(Exponential(1.0), LogLink(), "Exponential/log generic", μ -> rand(rng, Exponential(μ)))
run(Beta(10.0, 1.0), LogitLink(), "Beta(φ=10)/logit generic", μ -> clamp(rand(rng, Beta(clamp(μ,1e-3,1-1e-3)*10, clamp(1-μ,1e-3,1-1e-3)*10)), 1e-4, 1-1e-4))
run(G.NB1(1.0), LogLink(), "NB1(φ=1) generic (shared NB1)", μ -> rand(rng, NegativeBinomial(μ/1.0, 0.5)))
run(G.TruncatedPoisson(), LogLink(), "TruncatedPoisson generic", μ -> (x = 0; while x == 0; x = rand(rng, Poisson(max(μ, 0.05))); end; x))
