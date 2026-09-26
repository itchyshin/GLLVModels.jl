using GLLVModels, LinearAlgebra, Random, Statistics, Distributions, Printf
const GM = GLLVModels
const Optim = GM.Optim
logistic(x) = 1 / (1 + exp(-x))
function lowtri(rng, p, K, sd)
    Λ = sd .* randn(rng, p, K)
    for j in 1:K, i in 1:(j - 1); Λ[i, j] = 0.0; end
    for j in 1:K; Λ[j, j] = abs(Λ[j, j]) + 0.3 * sd; end
    return Λ
end
function rand_orderedbeta(rng, η, c0, c1, φ)
    p0 = logistic(c0 - η); p1 = logistic(η - c1)
    u = rand(rng)
    u < p0 && return 0.0
    u > 1 - p1 && return 1.0
    μ = clamp(logistic(η), 1e-10, 1 - 1e-10)
    return rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ))
end
function fdgrad(f, θ; h)
    g = similar(θ, Float64)
    for i in eachindex(θ)
        hi = h * max(1.0, abs(θ[i]))
        θp = copy(θ); θp[i] += hi
        θm = copy(θ); θm[i] -= hi
        g[i] = (f(θp) - f(θm)) / (2hi)
    end
    return g
end
function check(seed; p=5, n=40, K=2, sd=0.5, c0true=-1.0, c1true=1.0, φtrue=8.0, βlo=-0.3, βhi=0.3)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [rand_orderedbeta(rng, β[t] + dot(Λ[t, :], Z[:, i]), c0true, c1true, φtrue) for t in 1:p, i in 1:n]
    fit = fit_ordered_beta_gllvm(Y; K = K)
    rr = GM.rr_theta_len(p, K)
    negll = θ -> begin
        β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p+1):(p+rr)], p, K)
        c0_ = θ[p+rr+1]; c1_ = c0_ + exp(θ[p+rr+2]); φ_ = exp(θ[p+rr+3])
        v = try -GM.ordered_beta_marginal_loglik_laplace(Y, Λ_, β_, c0_, c1_, φ_; mask=nothing, maxiter=100, tol=1e-9)
        catch; return 1e12; end
        isfinite(v) ? v : 1e12
    end
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), fit.c0, log(fit.c1 - fit.c0), log(fit.φ))
    println("="^50); @printf("seed=%d (CLEAN)  ll=%.4f\n", seed, fit.loglik)
    for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
        g = fdgrad(negll, θhat; h=h)
        gs = abs.(g) .* (1 .+ abs.(θhat))
        imax = argmax(gs)
        @printf("  h=%.0e  gscaled_max=%.4e  coord=%d  raw_g=%.4e\n", h, gs[imax], imax, g[imax])
    end
end
for seed in (2001, 2008)
    check(seed)
end
