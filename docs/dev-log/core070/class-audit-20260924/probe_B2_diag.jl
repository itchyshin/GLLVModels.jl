# Diagnose poisson_formula_X_harsh seed 1: evaluate the SAME θ with the backtracked generic kernel.
using GLLVModels, LinearAlgebra, Random, Distributions, StatsModels
const G = GLLVModels
seed = parse(Int, get(ENV, "SEED", "1"))
rng = MersenneTwister(200 + seed); p, n, K = 10, 80, 2
Λ = 2.0 .* randn(rng, p, K); β = randn(rng, p) .- 1.0; Z = randn(rng, K, n); x = randn(rng, n)
Y = [rand(rng, Poisson(exp(clamp(β[t] + dot(Λ[t, :], Z[:, s]) + 0.5 * x[s], -6, 6)))) for t in 1:p, s in 1:n]
X = reshape(repeat(x', p), p, n, 1)
f = gllvm(@formula(y ~ 1 + x), Y, (x = x,); family = Poisson(), K = 2)
O = G._build_offset(X, f.γ)
v_kernel = G._marginal_loglik_offset(Poisson(), Y, ones(Int, p, n), f.Λ, f.β, O, LogLink())
v_generic = G.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, n), f.Λ, f.β, LogLink(); offset = O)
# per-site discrepancy
d = [G._laplace_site_off(Poisson(), Y[:, s], ones(Int, p), f.Λ, f.β .+ O[:, s], LogLink()) -
     G.laplace_loglik_site(Poisson(), Y[:, s], ones(Int, p), f.Λ, f.β, LogLink(); offset = O[:, s]) for s in 1:n]
println("seed=$seed fit.loglik=$(f.loglik) converged=$(f.converged) gamma=$(round.(f.γ; sigdigits=4)) maxabsLambda=$(round(maximum(abs, f.Λ); sigdigits=4))")
println("  value at fitted θ: covariates kernel = $(v_kernel); backtracked generic kernel (same θ, same offset) = $(v_generic)")
println("  sites where kernels disagree by >1e-6: $(count(abs.(d) .> 1e-6)) of $n; largest |Δ| = $(maximum(abs, d))")
# honest comparator: default no-X Poisson fit with the covariate as an offset at the fitted γ
f2 = fit_poisson_gllvm(Y; K = 2, offset = O)
println("  comparator fit_poisson_gllvm(offset=Xγ̂): loglik=$(f2.loglik) converged=$(f2.converged)")
