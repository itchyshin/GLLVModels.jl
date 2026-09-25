using GLLVModels, Random, Distributions, LinearAlgebra
const G = GLLVModels
logistic(x) = 1/(1+exp(-x))
function simzip(seed; p=5, n=80, K=2, sd=0.8)
    rng = MersenneTwister(seed)
    Λ = sd .* randn(rng, p, K); βz = randn(rng, p) .* 0.5 .- 1.2; βc = randn(rng, p) .* 0.5 .+ 1.0
    Z = randn(rng, K, n)
    Y = [rand(rng) < logistic(βz[t]) ? 0 : rand(rng, Poisson(exp(βc[t] + dot(Λ[t,:], Z[:,s])))) for t in 1:p, s in 1:n]
    Y
end
Y = simzip(1)
t = @elapsed f = fit_zip_gllvm(Y; K=2); println("zip first ", t, " ", f, " it=", f.iterations)
t = @elapsed f = fit_zip_gllvm(simzip(2); K=2); println("zip ", t, " ", f, " it=", f.iterations)
t = @elapsed f = fit_zinb_gllvm(simzip(2); K=2); println("zinb ", t, " ", f, " it=", f.iterations)
t = @elapsed f = fit_zip_gllvm(simzip(3; sd=1.6); K=2); println("zip hard ", t, " ", f, " it=", f.iterations)
