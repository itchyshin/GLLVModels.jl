using GLLVModels
using Test
using Random
using Distributions
using SpecialFunctions: loggamma
using StableRNGs

# cv_gllvm scores each held-out cell with the fitted dispersion of that cell's species or
# group (#1211). The reference is built here from the fold fit's own fields and the stored
# predictions, with an independent density formula, so it does not call the code under test.

function _g1211_disp(fit, t)
    nm = fieldnames(typeof(fit))
    :r_group in nm && return fit.r_group[fit.group[t]]
    :r in nm && return fit.r
    for f in (:φ, :α)
        if f in nm
            v = getfield(fit, f)
            return v isa AbstractVector ? v[fit.group[t]] : v
        end
    end
    error("no dispersion field in $(typeof(fit))")
end

function _g1211_logpdf(kind, d, y, μ)
    if kind === :nb
        # integer y: log Γ(y + d) - log Γ(d) as a sum, and -d log(1 + μ/d), stay exact at the
        # Poisson-limit dispersions a grouped fit can end at
        μ = max(μ, 1e-12)
        return sum(log(d + i) for i in 0:(Int(y) - 1); init = 0.0) - loggamma(y + 1) -
               d * log1p(μ / d) + y * (log(μ) - log(d + μ))
    elseif kind === :gamma
        μ = max(μ, 1e-12); y = max(y, 1e-12)
        return d * log(d) - loggamma(d) + (d - 1) * log(y) - d * log(μ) - d * y / μ
    else
        μ = clamp(μ, 1e-6, 1 - 1e-6); y = clamp(y, 1e-6, 1 - 1e-6)
        return loggamma(d) - loggamma(μ * d) - loggamma((1 - μ) * d) +
               (μ * d - 1) * log(y) + ((1 - μ) * d - 1) * log(1 - y)
    end
end

# The fold assignment cv_gllvm makes from `rng` (first draw is the shuffle).
function _g1211_folds(split, p, n, k, seed)
    rng = StableRNG(seed)
    if split === :random
        sh = Random.shuffle(rng, [(t, s) for s in 1:n for t in 1:p])
        g = [Tuple{Int,Int}[] for _ in 1:k]
        for (i, c) in enumerate(sh); push!(g[mod1(i, k)], c); end
        return g
    end
    sites = Random.shuffle(rng, collect(1:n))
    sg = [Int[] for _ in 1:k]
    for (i, s) in enumerate(sites); push!(sg[mod1(i, k)], s); end
    return [[(t, s) for s in sg[j] for t in 1:p] for j in 1:k]
end

function _g1211_data(kind, rng, p, n, disp)
    b = 0.3 .* randn(rng, p); lam = 0.6 .* randn(rng, p)
    Y = kind === :nb ? zeros(Int, p, n) : zeros(Float64, p, n)
    for s in 1:n
        z = randn(rng)
        for t in 1:p
            if kind === :nb
                μ = exp(1.0 + b[t] + lam[t] * z)
                Y[t, s] = rand(rng, NegativeBinomial(disp[t], disp[t] / (disp[t] + μ)))
            elseif kind === :gamma
                μ = exp(1.0 + b[t] + lam[t] * z)
                Y[t, s] = rand(rng, Distributions.Gamma(disp[t], μ / disp[t]))
            else
                μ = 1 / (1 + exp(-(b[t] + lam[t] * z)))
                Y[t, s] = rand(rng, Distributions.Beta(μ * disp[t], (1 - μ) * disp[t]))
            end
        end
    end
    return Y
end

@testset "cv_gllvm held-out density uses the fitted dispersion (#1211)" begin
    p, k = 8, 3
    coarse = [1, 1, 1, 1, 2, 2, 3, 3]
    fams = Dict(:nb => NegativeBinomial(), :gamma => Distributions.Gamma(), :beta => Distributions.Beta())
    layouts = [
        ("NB2 shared",             :nb,    fill(3.0, p),                      (;), 80),
        ("NB2 per-species",        :nb,    [0.4, 0.8, 1.5, 3, 2, 4, 0.5, 3],   (; disp_group = :species), 300),
        ("NB2 grouped (3 groups)", :nb,    [0.5, 0.5, 0.5, 0.5, 2, 2, 4, 4],   (; disp_group = coarse), 200),
        ("Gamma shared",           :gamma, fill(2.0, p),                      (;), 80),
        ("Gamma per-species",      :gamma, [0.5, 1, 2, 4, 8, 16, 0.7, 30],    (; disp_group = :species), 80),
        ("Gamma grouped (3)",      :gamma, [0.5, 0.5, 0.5, 0.5, 4, 4, 30, 30], (; disp_group = coarse), 80),
        ("Beta per-species",       :beta,  [3, 6, 12, 24, 30, 5, 10, 20.0],   (; disp_group = :species), 80),
        ("Beta grouped (3)",       :beta,  [3, 3, 3, 3, 15, 15, 60, 60.0],    (; disp_group = coarse), 80),
    ]
    for (name, kind, disp, kw, n) in layouts, split in (:random, :site)
        @testset "$name, $split split" begin
            Y = _g1211_data(kind, StableRNG(11), p, n, disp)
            res = cv_gllvm(Y; k_folds = k, split = split, family = fams[kind], K = 1,
                           rng = StableRNG(5), kw...)
            # Out of scope here: a fitted r at the Poisson boundary (above 1e6) is scored by the
            # Float64 NegativeBinomial density, which is not exact there. The layouts keep clear of it.
            @test all(f -> !(:r_group in fieldnames(typeof(f))) || all(<(1e6), f.r_group), res.fits)
            g = _g1211_folds(split, p, n, k, 5)
            ref = [sum(_g1211_logpdf(kind, _g1211_disp(res.fits[j], t), Y[t, s], res.predictions[t, s])
                       for (t, s) in g[j]) for j in 1:k]
            @test isapprox(res.fold_logliks, ref; atol = 1e-8, rtol = 0)
            @test isapprox(res.loglik, sum(ref); atol = 1e-8, rtol = 0)
        end
    end
end
