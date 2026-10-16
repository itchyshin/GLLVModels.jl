# ZINB joint pseudo-true shift in (common betaz shift δ, log r shift ρ) at the truth, other params fixed.
# One Newton step from the truth on the mean objective: (δ*, ρ*) = −H⁻¹ g, g/H by central differences.
# Compares the package objective (`twopart_loglik_site(ZINB(r), …)`, expected-info log-det) with the
# observed-curvature Laplace (≈ exact at p = 10; see mechanism_zi.jl). Standalone.
# usage: julia --project=<GLLVModels.jl> mechanism_zinb_joint.jl <pi> <n_sites> <seed>
using GLLVModels, Random, Distributions, LinearAlgebra, ForwardDiff
const G = GLLVModels
π0 = parse(Float64, ARGS[1]); n = parse(Int, ARGS[2]); seed = parse(Int, ARGS[3])
p, K = 10, 2
Λ = [0.8 0.0; 0.5 0.6; -0.4 0.5; 0.6 -0.3; 0.3 0.6; -0.5 0.4; 0.7 -0.2; 0.2 -0.5; -0.3 0.5; 0.4 0.3]
βc = collect(range(0.5, 1.5; length = p)); r0 = 2.0
rng = Xoshiro(seed); Z = randn(rng, K, n); Y = zeros(Int, p, n)
for s in 1:n
    ηc = βc .+ Λ * view(Z, :, s)
    for t in 1:p
        μ = exp(ηc[t])
        Y[t, s] = rand(rng) < π0 ? 0 : rand(rng, NegativeBinomial(r0, r0 / (r0 + μ)))
    end
end
bz0 = fill(log(π0 / (1 - π0)), p)
function h(fam, y, βz, z)
    sum(G._tp_pieces(fam, y[t], βz[t], βc[t] + dot(view(Λ, t, :), z))[5] for t in 1:p) - 0.5 * dot(z, z)
end
function mode(fam, y, βz)
    z = zeros(K)
    for _ in 1:100
        g = ForwardDiff.gradient(z -> h(fam, y, βz, z), z); H = ForwardDiff.hessian(z -> h(fam, y, βz, z), z)
        st = -(H \ g); a = 1.0; f0 = h(fam, y, βz, z)
        while h(fam, y, βz, z .+ a .* st) < f0 - 1e-12 && a > 1e-6; a /= 2; end
        z = z .+ a .* st; norm(g) < 1e-10 && break
    end
    return z
end
obslap(fam, y, βz) = (ẑ = mode(fam, y, βz); h(fam, y, βz, ẑ) - 0.5 * logdet(Symmetric(-ForwardDiff.hessian(z -> h(fam, y, βz, z), ẑ))))
pkg(fam, y, βz) = G.twopart_loglik_site(fam, y, zeros(p, K), Λ, βz, βc)
total(f, δ, ρ) = (fam = G.ZINB(r0 * exp(ρ)); sum(f(fam, view(Y, :, s), bz0 .+ δ) for s in 1:n))
println("ZINB joint (betaz shift δ, log r shift ρ); pi = $π0, n = $n sites, seed = $seed, truth r = $r0")
h1 = 1e-3; h2 = 1e-3
for (name, f) in (("package Laplace (expected info)", pkg), ("Laplace, observed curvature", obslap))
    f00 = total(f, 0, 0)
    fp0 = total(f, h1, 0); fm0 = total(f, -h1, 0); f0p = total(f, 0, h2); f0m = total(f, 0, -h2)
    fpp = total(f, h1, h2); fpm = total(f, h1, -h2); fmp = total(f, -h1, h2); fmm = total(f, -h1, -h2)
    g = [(fp0 - fm0) / (2h1), (f0p - f0m) / (2h2)]
    H = [(fp0 - 2 * f00 + fm0) / h1^2  (fpp - fpm - fmp + fmm) / (4h1 * h2);
         (fpp - fpm - fmp + fmm) / (4h1 * h2)  (f0p - 2 * f00 + f0m) / h2^2]
    st = -(H \ g)
    println(rpad(name, 34), "conditional δ* = ", round(-g[1] / H[1, 1]; sigdigits = 3),
            "   joint (δ*, ρ*) = (", round(st[1]; sigdigits = 3), ", ", round(st[2]; sigdigits = 3),
            ")  ⇒ r* ≈ ", round(r0 * exp(st[2]); sigdigits = 3))
end
