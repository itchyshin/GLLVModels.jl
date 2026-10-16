# One large-n fit as an estimate of the pseudo-true parameters of the package objective.
# usage: julia --project=<GLLVModels.jl> largen_fit.jl <ZIP|ZINB> <pi> <n> <seed>
using GLLVModels, Random, Distributions, LinearAlgebra, Statistics
fam = ARGS[1]; π0 = parse(Float64, ARGS[2]); n = parse(Int, ARGS[3]); seed = parse(Int, ARGS[4])
p, K = 10, 2
Λ = [0.8 0.0; 0.5 0.6; -0.4 0.5; 0.6 -0.3; 0.3 0.6; -0.5 0.4; 0.7 -0.2; 0.2 -0.5; -0.3 0.5; 0.4 0.3]
βc = collect(range(0.5, 1.5; length = p)); r = 2.0
rng = Xoshiro(seed); Z = randn(rng, K, n); Y = zeros(Int, p, n)
for s in 1:n
    ηc = βc .+ Λ * view(Z, :, s)
    for t in 1:p
        μ = exp(ηc[t])
        Y[t, s] = rand(rng) < π0 ? 0 : fam == "ZIP" ? rand(rng, Poisson(μ)) : rand(rng, NegativeBinomial(r, r / (r + μ)))
    end
end
t0 = time()
fit = fam == "ZIP" ? fit_zip_gllvm(Y; K = K) : fit_zinb_gllvm(Y; K = K)
el = time() - t0
bz = log(π0 / (1 - π0))
S = Λ * Λ'; Ŝ = fit.Λc * fit.Λc'
println("$fam pi=$π0 n=$n seed=$seed  converged=$(fit.converged) iterations=$(fit.iterations) loglik=$(fit.loglik)  fit time $(round(el; digits = 0)) s")
println("betaz − truth: mean ", round(mean(fit.βz .- bz); digits = 4), "  per species ", round.(fit.βz .- bz; digits = 3))
println("betac − truth: mean ", round(mean(fit.βc .- βc); digits = 4), "  per species ", round.(fit.βc .- βc; digits = 3))
fam == "ZINB" && println("r̂ = ", round(fit.r; digits = 4), " (truth 2)")
println("ΛΛ' diag − truth: mean ", round(mean(diag(Ŝ) .- diag(S)); digits = 4), "; relative ", round(mean(diag(Ŝ) ./ diag(S)) - 1; digits = 4))
println("ΛΛ' offdiag − truth: mean abs ", round(mean(abs.((Ŝ .- S)[[i > j for i in 1:p, j in 1:p]])); digits = 4))
