using Pkg; Pkg.activate(@__DIR__; io=devnull)
using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels
for seed in (1, 5, 6)
    rng = MersenneTwister(seed)
    p, n, K = 8, 80, 2
    lam = seed <= 4 ? 1.0 : 1.8
    Λ = lam .* randn(rng, p, K); β = 0.5 .+ 0.5 .* randn(rng, p)
    x = randn(rng, n); γ = 0.4
    Z = randn(rng, K, n)
    η = β .+ Λ*Z .+ γ .* x'
    Y = [rand(rng, Poisson(min(exp(η[t,s]), 1e5))) for t in 1:p, s in 1:n]
    X = reshape(repeat(x', p), p, n, 1)
    O = G._build_offset(X, [γ]); N1 = ones(Int, p, n)
    ll_off  = G._marginal_loglik_offset(Poisson(), Y, N1, Λ, β, O, G.LogLink())
    ll_core = G.marginal_loglik_laplace(Poisson(), Y, N1, Λ, β, G.LogLink(); offset = O)
    println("seed=$seed  at simulation truth: fit_gllvm_cov kernel (_laplace_site_off) = $(round(ll_off, sigdigits=8))   backtracked core (_laplace_mode, same model) = $(round(ll_core, sigdigits=8))")
end
