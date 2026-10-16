# Self-contained mechanism check: is the zero-inflated two-part Laplace objective consistent for betaz?
# At the TRUE parameters, on n simulated sites, compare the mean score in a common betaz shift δ for
#   (a) the package site marginal  GLLVModels.twopart_loglik_site(ZIPoisson()/ZINB(r), …)
#       (its log-det uses the EXPECTED count information `_zi_Icc_*`; `_tp_observed_Wc(::Any, …) = Wc`)
#   (b) the same Laplace with the OBSERVED cell curvature −∂²logf/∂η² in the log-det
#   (c) adaptive Gauss–Hermite quadrature (20×20 nodes per site) ≈ exact marginal
# δ* = −score/curvature is the betaz shift the objective's maximiser drifts to (other params at truth).
# usage: julia --project=<GLLVModels.jl> mechanism_zi.jl <ZIP|ZINB> <pi> <n_sites> <seed> [betaz|lambda]
# `lambda` scores a common loading scale Λ → (1+δ)Λ instead (δ* < 0 ⇒ attenuated loadings, ΛΛ' biased by ≈ 2δ*·ΛΛ').
using GLLVModels, Random, Distributions, LinearAlgebra, ForwardDiff
const G = GLLVModels
famname = ARGS[1]; π0 = parse(Float64, ARGS[2]); n = parse(Int, ARGS[3]); seed = parse(Int, ARGS[4])
which = length(ARGS) >= 5 ? ARGS[5] : "betaz"
p, K = 10, 2
Λ = [0.8 0.0; 0.5 0.6; -0.4 0.5; 0.6 -0.3; 0.3 0.6; -0.5 0.4; 0.7 -0.2; 0.2 -0.5; -0.3 0.5; 0.4 0.3]
βc = collect(range(0.5, 1.5; length = p)); r = 2.0
rng = Xoshiro(seed); Z = randn(rng, K, n); Y = zeros(Int, p, n)
for s in 1:n
    ηc = βc .+ Λ * view(Z, :, s)
    for t in 1:p
        μ = exp(ηc[t])
        Y[t, s] = rand(rng) < π0 ? 0 : famname == "ZIP" ? rand(rng, Poisson(μ)) :
                  rand(rng, NegativeBinomial(r, r / (r + μ)))
    end
end
fam = famname == "ZIP" ? G.ZIPoisson() : G.ZINB(r)
bz0 = fill(log(π0 / (1 - π0)), p)
logf(y, ηz, ηc) = G._tp_pieces(fam, y, ηz, ηc)[5]          # the package's own ZI log-density
const SC = Ref(1.0)
h(y, βz, z) = sum(logf(y[t], βz[t], βc[t] + SC[] * dot(view(Λ, t, :), z)) for t in 1:p) - 0.5 * dot(z, z)
function mode(y, βz)
    z = zeros(K)
    for _ in 1:100
        g = ForwardDiff.gradient(z -> h(y, βz, z), z); H = ForwardDiff.hessian(z -> h(y, βz, z), z)
        st = -(H \ g); a = 1.0; f0 = h(y, βz, z)
        while h(y, βz, z .+ a .* st) < f0 - 1e-12 && a > 1e-6; a /= 2; end
        z = z .+ a .* st; norm(g) < 1e-10 && break
    end
    return z
end
J = SymTridiagonal(zeros(20), [sqrt(i / 2) for i in 1:19]); E = eigen(J)
GX = E.values; GW = sqrt(π) .* E.vectors[1, :] .^ 2
function site_exact(y, βz)
    ẑ = mode(y, βz); L = inv(cholesky(Symmetric(-ForwardDiff.hessian(z -> h(y, βz, z), ẑ))).U)
    v = [log(GW[i] * GW[j]) + h(y, βz, ẑ .+ sqrt(2) .* (L * [GX[i], GX[j]])) + GX[i]^2 + GX[j]^2
         for i in 1:20 for j in 1:20]
    m = maximum(v)
    return m + log(sum(exp.(v .- m))) + logdet(L) + (K / 2) * log(2) - (K / 2) * log(2π)
end
function site_obslap(y, βz)
    ẑ = mode(y, βz)
    return h(y, βz, ẑ) - 0.5 * logdet(Symmetric(-ForwardDiff.hessian(z -> h(y, βz, z), ẑ)))
end
site_pkg(y, βz) = G.twopart_loglik_site(fam, y, zeros(p, K), SC[] .* Λ, βz, βc)
function total(f, δ)
    if which == "lambda"
        SC[] = 1 + δ; v = sum(f(view(Y, :, s), bz0) for s in 1:n); SC[] = 1.0; return v
    end
    return sum(f(view(Y, :, s), bz0 .+ δ) for s in 1:n)
end
println("$famname, pi = $π0, truth betaz = ", round(bz0[1]; digits = 4), ", n = $n sites, seed = $seed, direction = ", which == "lambda" ? "common loading scale (1+δ)Λ" : "common betaz shift")
hδ = 1e-3
for (name, f) in (("package Laplace (expected info)", site_pkg), ("Laplace, observed curvature", site_obslap),
                  ("AGHQ 20x20 (≈ exact)", site_exact))
    fp = total(f, hδ); f0 = total(f, 0.0); fm = total(f, -hδ)
    sc = (fp - fm) / (2hδ); cv = (fp - 2 * f0 + fm) / hδ^2
    println(rpad(name, 34), "mean score/site = ", rpad(round(sc / n; sigdigits = 3), 10),
            which == "lambda" ? "implied loading scale δ* = " : "implied betaz shift δ* = ", round(-sc / cv; sigdigits = 3))
end
