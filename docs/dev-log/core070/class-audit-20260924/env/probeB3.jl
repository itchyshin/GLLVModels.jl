using Pkg; Pkg.activate(@__DIR__; io=devnull)
using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels
seed = 5
rng = MersenneTwister(100+seed)
p, n, K = 8, 80, 2
Λ = 1.8 .* randn(rng, p, K); βc = 0.5 .+ 0.5 .* randn(rng, p)
Z = randn(rng, K, n); η = βc .+ Λ*Z
Y = [rand(rng) < 0.25 ? 0 : rand(rng, Poisson(min(exp(η[t,s]), 1e5))) for t in 1:p, s in 1:n]
fit = G.fit_zip_gllvm(Y; K=K, iterations=150)
βz_true = fill(log(0.25/0.75), p)
ll_truth = G.zip_marginal_loglik_laplace(Y, Λ, βz_true, βc; hessian=:observed)
ll_fit = G.zip_marginal_loglik_laplace(Y, fit.Λc, fit.βz, fit.βc; hessian=:observed)
sites = [G.twopart_loglik_site(G.ZIPoisson(), Y[:,s], zeros(p,K), fit.Λc, fit.βz, fit.βc) for s in 1:n]
println("fit.converged=", fit.converged, " fit.iterations=", fit.iterations, " fit.loglik=", fit.loglik)
println("loglik at returned point (recomputed)=", ll_fit, "   loglik at simulation truth=", ll_truth)
println("site values at returned point: min=", minimum(sites), "  #sites < -1e4: ", count(<(-1e4), sites))
