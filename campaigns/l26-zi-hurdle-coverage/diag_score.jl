# Mechanism check for the ZIP betaz bias: mean score in a common betaz shift δ at the TRUE parameters,
# on a large simulated sample, under three site-marginal approximations:
#   (a) package Laplace  (GLLVModels.zip_marginal_loglik_laplace; log-det uses EXPECTED count info)
#   (b) "exact"          (adaptive Gauss–Hermite, 20×20 nodes per site, centred at the mode)
#   (c) observed Laplace (same mode, log-det from the OBSERVED −∂²logf/∂η² of each cell)
# A consistent objective has E[score] = 0 at the truth; δ* ≈ −score/curvature approximates the
# pseudo-true shift the estimator converges to (other parameters held at truth).
include("/home/user/l26/l26core.jl")
using LinearAlgebra, ForwardDiff
famname = ARGS[1]; π0 = parse(Float64, ARGS[2]); n = parse(Int, ARGS[3]); seed = parse(Int, ARGS[4])
cellbase = famname == "ZIP" ? 0 : famname == "ZINB" ? 9 : 18
cell = cellbase + (findfirst(==(π0), PIS) - 1) * 3 + 1       # any cell with this family/pi; n overridden below
# generate with the sweep generator but a custom n and seed
function gen(fam, π0, n, seed)
    rng = Xoshiro(seed); Z = randn(rng, K, n); Y = zeros(Int, P, n)
    for s in 1:n
        ηc = BETAC_TRUE .+ LAMBDA_TRUE * view(Z, :, s)
        for t in 1:P
            μ = exp(ηc[t])
            if rand(rng) < π0
                Y[t, s] = 0
            elseif fam == "ZIP"
                Y[t, s] = rand(rng, Poisson(μ))
            elseif fam == "ZINB"
                Y[t, s] = rand(rng, NegativeBinomial(R_TRUE, R_TRUE / (R_TRUE + μ)))
            else
                Y[t, s] = rand_ztpois(rng, μ)
            end
        end
    end
    return Y
end
Y = gen(famname, π0, n, seed)
fam = famname == "ZIP" ? G.ZIPoisson() : famname == "ZINB" ? G.ZINB(R_TRUE) : G.HurdlePoisson()
bz0 = fill(betaz_true(famname, π0), P)
Λz0 = zeros(P, K)

logf(y, ηz, ηc) = G._tp_pieces(fam, y, ηz, ηc)[5]
h(y, βz, z) = sum(logf(y[t], βz[t], BETAC_TRUE[t] + dot(view(LAMBDA_TRUE, t, :), z)) for t in 1:P) - 0.5 * dot(z, z)

function site_mode(y, βz)
    z = zeros(K)
    for _ in 1:100
        g = ForwardDiff.gradient(z -> h(y, βz, z), z); H = ForwardDiff.hessian(z -> h(y, βz, z), z)
        step = -(H \ g)
        # backtracking on h
        a = 1.0; f0 = h(y, βz, z)
        while h(y, βz, z .+ a .* step) < f0 - 1e-12 && a > 1e-6; a /= 2; end
        z = z .+ a .* step
        norm(g) < 1e-10 && break
    end
    return z
end

# Gauss–Hermite (physicists') nodes/weights via Golub–Welsch
function gh(m)
    J = SymTridiagonal(zeros(m), [sqrt(i / 2) for i in 1:(m - 1)])
    E = eigen(J); x = E.values; w = sqrt(π) .* E.vectors[1, :] .^ 2
    return x, w
end
const GX, GW = gh(20)

function site_exact(y, βz)
    ẑ = site_mode(y, βz)
    H = -ForwardDiff.hessian(z -> h(y, βz, z), ẑ)
    L = inv(cholesky(Symmetric(H)).U)          # z = ẑ + √2 L x
    acc = -Inf; vals = Float64[]
    for i in eachindex(GX), j in eachindex(GX)
        x = [GX[i], GX[j]]
        z = ẑ .+ sqrt(2) .* (L * x)
        push!(vals, log(GW[i] * GW[j]) + h(y, βz, z) + dot(x, x))
    end
    m = maximum(vals)
    return m + log(sum(exp.(vals .- m))) + (K / 2) * log(2) + logdet(L) - (K / 2) * log(2π)
end

function site_obslap(y, βz)
    ẑ = site_mode(y, βz)
    H = -ForwardDiff.hessian(z -> h(y, βz, z), ẑ)  # = I + Λ' W_obs Λ
    return h(y, βz, ẑ) - 0.5 * logdet(Symmetric(H))
end

site_pkg(y, βz) = G.twopart_loglik_site(fam, y, Λz0, LAMBDA_TRUE, βz, BETAC_TRUE)

function total(f, δ)
    βz = bz0 .+ δ
    return sum(f(view(Y, :, s), βz) for s in 1:size(Y, 2))
end

# sanity: package site value vs our observed Laplace vs exact for a few sites
for s in 1:3
    y = view(Y, :, s)
    println("site $s y=", collect(y), "  pkg=", round(site_pkg(y, bz0), digits = 6), "  obsLap=",
            round(site_obslap(y, bz0), digits = 6), "  exact=", round(site_exact(y, bz0), digits = 6))
end
hδ = 1e-3
println("n = $n sites, family $famname, pi = $π0, truth betaz = ", round(bz0[1], digits = 4))
for (name, f) in (("package Laplace (expected W)", site_pkg), ("observed-W Laplace", site_obslap), ("exact AGHQ 20x20", site_exact))
    t = @elapsed begin
        fp = total(f, hδ); f0 = total(f, 0.0); fm = total(f, -hδ)
    end
    sc = (fp - fm) / (2hδ); cv = (fp - 2 * f0 + fm) / hδ^2
    println(rpad(name, 32), " mean score/site = ", round(sc / n, sigdigits = 4), "   curvature/site = ",
            round(cv / n, sigdigits = 4), "   implied shift δ* = ", round(-sc / cv, sigdigits = 4),
            "   (", round(t, digits = 1), " s)")
end
