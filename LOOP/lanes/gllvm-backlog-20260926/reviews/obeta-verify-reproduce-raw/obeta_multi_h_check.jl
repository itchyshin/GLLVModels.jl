# Multi-step-size FD gradient check for the three flagged seeds, at BOTH the
# first-fit theta_hat and the true-start restart theta. A REAL non-stationary
# point should show a gradient that is STABLE (not erratic/coordinate-hopping)
# as h shrinks; an FD artifact from a branch/kink (as seen in compoisson) swings
# wildly and moves to different coordinates as h changes.
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
bt_ls() = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
perturb(rng, θ) = θ .+ (0.3 .* abs.(θ) .+ 0.1) .* randn(rng, length(θ))

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
    θtrue = vcat(β, GM.pack_lambda(Λ), c0true, log(c1true - c0true), log(φtrue))
    rt = Optim.optimize(negll, θtrue, bt_ls(), Optim.Options(g_tol=1e-5, iterations=500); autodiff=:finite)
    θr = Optim.minimizer(rt); llr = -Optim.minimum(rt)

    println("="^60); @printf("seed=%d  first-fit ll=%.4f  restart ll=%.4f (gap %+0.3f)\n", seed, fit.loglik, llr, llr - fit.loglik)
    for (tag, θ) in (("first-fit theta_hat", θhat), ("restart theta", θr))
        println("  -- $tag --")
        for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
            g = fdgrad(negll, θ; h=h)
            gs = abs.(g) .* (1 .+ abs.(θ))
            imax = argmax(gs)
            @printf("    h=%.0e  gscaled_max=%.4e  coord=%d  raw_g=%.4e\n", h, gs[imax], imax, g[imax])
        end
    end
end

for seed in (2003, 2005, 2010)
    check(seed)
end
