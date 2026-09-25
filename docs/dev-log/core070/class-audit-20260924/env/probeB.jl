using Pkg; Pkg.activate(@__DIR__; io=devnull)
using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels

function fdgrad(f, θ; h=1e-5)
    g = similar(θ)
    for i in eachindex(θ)
        e = zeros(length(θ)); e[i] = h
        g[i] = (f(θ .+ e) - f(θ .- e)) / (2h)
    end
    g
end

function sim_pois(rng; p=8, n=80, K=2, lam=1.0)
    Λ = lam .* randn(rng, p, K); β = 0.5 .+ 0.5 .* randn(rng, p)
    x = randn(rng, n); γ = 0.4
    Z = randn(rng, K, n)
    η = β .+ Λ*Z .+ γ .* x'
    Y = [rand(rng, Poisson(min(exp(η[t,s]), 1e5))) for t in 1:p, s in 1:n]
    X = reshape(repeat(x', p), p, n, 1)
    Y, X
end

println("== fit_gllvm_cov(Poisson) : converged flag vs gradient at returned point")
for seed in 1:8
    rng = MersenneTwister(seed)
    Y, X = sim_pois(rng; lam = seed <= 4 ? 1.0 : 1.8)
    p, n = size(Y); K = 2
    t = @elapsed fit = G.fit_gllvm_cov(Y; family=Poisson(), X=X, K=K)
    θ = vcat(fit.β, fit.γ, G.pack_lambda(fit.Λ))
    rr = G.rr_theta_len(p, K)
    f = θ -> begin
        β = θ[1:p]; γ = θ[p+1:p+1]; Λ = G.unpack_lambda(θ[p+2:p+1+rr], p, K)
        O = G._build_offset(X, γ)
        v = try -G._marginal_loglik_offset(Poisson(), Y, ones(Int,p,n), Λ, β, O, G.LogLink()) catch; 1e12 end
        isfinite(v) ? v : 1e12
    end
    g = fdgrad(f, θ)
    println("seed=$seed  converged=$(fit.converged)  iters=$(fit.iterations)  loglik=$(round(fit.loglik, digits=3))  max|grad|=$(round(maximum(abs,g), sigdigits=3))  time=$(round(t,digits=1))s")
end

println("== fit_zip_gllvm : converged flag vs gradient at returned point")
for seed in 1:8
    rng = MersenneTwister(100+seed)
    p, n, K = 8, 80, 2
    Λ = (seed <= 4 ? 1.0 : 1.8) .* randn(rng, p, K); βc = 0.5 .+ 0.5 .* randn(rng, p)
    Z = randn(rng, K, n); η = βc .+ Λ*Z
    Y = [rand(rng) < 0.25 ? 0 : rand(rng, Poisson(min(exp(η[t,s]), 1e5))) for t in 1:p, s in 1:n]
    t = @elapsed fit = G.fit_zip_gllvm(Y; K=K)
    θ = vcat(fit.βz, fit.βc, G.pack_lambda(fit.Λc))
    rr = G.rr_theta_len(p, K)
    f = θ -> begin
        βz = θ[1:p]; βc_ = θ[p+1:2p]; Λc = G.unpack_lambda(θ[2p+1:2p+rr], p, K)
        v = try -G.zip_marginal_loglik_laplace(Y, Λc, βz, βc_; hessian=:observed) catch; 1e12 end
        isfinite(v) ? v : 1e12
    end
    g = fdgrad(f, θ)
    println("seed=$seed  converged=$(fit.converged)  iters=$(fit.iterations)  loglik=$(round(fit.loglik, digits=3))  max|grad|=$(round(maximum(abs,g), sigdigits=3))  time=$(round(t,digits=1))s")
end
