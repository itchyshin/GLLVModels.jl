# Which coordinate dominates gscaled at theta2 for seed 2003, and how does the FD
# step size (h) affect it -- is 0.128 (h=1e-6) real residual gradient, or an FD-noise
# artifact from a kink/branch in the objective (as seen in the compoisson family)?
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

seed = 2003
p, n, K, sd, c0true, c1true, φtrue, βlo, βhi = 5, 40, 2, 0.5, -1.0, 1.0, 8.0, -0.3, 0.3
rng = MersenneTwister(seed)
β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
Y = [rand_orderedbeta(rng, β[t] + dot(Λ[t, :], Z[:, i]), c0true, c1true, φtrue) for t in 1:p, i in 1:n]
rr = GM.rr_theta_len(p, K)
negll = θ -> begin
    β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p+1):(p+rr)], p, K)
    c0_ = θ[p+rr+1]; c1_ = c0_ + exp(θ[p+rr+2]); φ_ = exp(θ[p+rr+3])
    v = try -GM.ordered_beta_marginal_loglik_laplace(Y, Λ_, β_, c0_, c1_, φ_; mask=nothing, maxiter=100, tol=1e-9)
    catch; return 1e12; end
    isfinite(v) ? v : 1e12
end
θtrue = vcat(β, GM.pack_lambda(Λ), c0true, log(c1true - c0true), log(φtrue))
rt = Optim.optimize(negll, θtrue, bt_ls(), Optim.Options(g_tol=1e-5, iterations=500); autodiff=:finite)
θ1 = Optim.minimizer(rt)

names = vcat(["β$i" for i in 1:p], ["Λpk$i" for i in 1:(size(GM.unpack_lambda(θ1[(p+1):(p+rr)],p,K),1)*0+rr)],
             ["c0", "logΔc", "logφ"])
for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    g = fdgrad(negll, θ1; h = h)
    gs = abs.(g) .* (1 .+ abs.(θ1))
    imax = argmax(gs)
    @printf("h=%.0e  gscaled_max=%.4e  at coord %d (%s)  raw_g=%.4e theta=%.4e\n",
            h, gs[imax], imax, names[min(imax,length(names))], g[imax], θ1[imax])
end
@printf("negll(θ1) = %.10f\n", negll(θ1))
