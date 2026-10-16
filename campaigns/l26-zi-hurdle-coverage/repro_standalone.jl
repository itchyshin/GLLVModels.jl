# Standalone L26 repro: one simulated replicate -> fit -> public Wald confint + vcov -> coverage of the truth.
# GLLVModels.jl @ 033d4e1da4c20f67e24d7308f442506065a8d768, Julia 1.11.9, single thread.
# usage: julia --project=<GLLVModels.jl> repro_standalone.jl <family: ZIP|ZINB|HurdlePoisson> <pi> <n> <seed>
using GLLVModels, Random, Distributions, LinearAlgebra, ForwardDiff
family = ARGS[1]; π0 = parse(Float64, ARGS[2]); n = parse(Int, ARGS[3]); seed = parse(Int, ARGS[4])
p, K = 10, 2
Λ = [0.8 0.0; 0.5 0.6; -0.4 0.5; 0.6 -0.3; 0.3 0.6; -0.5 0.4; 0.7 -0.2; 0.2 -0.5; -0.3 0.5; 0.4 0.3]
βc = collect(range(0.5, 1.5; length = p)); r = 2.0
function rand_ztpois(rng, μ)                     # exact inverse-CDF zero-truncated Poisson
    t = exp(-μ) + rand(rng) * (1 - exp(-μ)); k = 0; pk = exp(-μ); F = pk
    while F < t; k += 1; pk *= μ / k; F += pk; end
    return max(k, 1)
end
rng = Xoshiro(seed)
Z = randn(rng, K, n); Y = zeros(Int, p, n)
for s in 1:n                                      # sites outer, species inner
    ηc = βc .+ Λ * view(Z, :, s)
    for t in 1:p
    μ = exp(ηc[t])
    if rand(rng) < π0
        Y[t, s] = 0
    elseif family == "ZIP"
        Y[t, s] = rand(rng, Poisson(μ))
    elseif family == "ZINB"
        Y[t, s] = rand(rng, NegativeBinomial(r, r / (r + μ)))
    else
        Y[t, s] = rand_ztpois(rng, μ)
    end
    end
end
fit = family == "ZIP" ? fit_zip_gllvm(Y; K = K) : family == "ZINB" ? fit_zinb_gllvm(Y; K = K) :
      fit_hurdle_poisson_gllvm(Y; K = K)
ci = confint(fit, Y; method = :wald); V = vcov(fit, Y)
println(fit, "  converged = ", fit.converged, "  pd_hessian = ", ci.pd_hessian)
logit(x) = log(x / (1 - x))
bz = family == "HurdlePoisson" ? logit(1 - π0) : logit(π0)   # hurdle betaz = logit P(y > 0)
terms = String.(ci.term)
for t in 1:p
    i = findfirst(==("betaz[$t]"), terms); j = findfirst(==("betac[$t]"), terms)
    println(rpad("betaz[$t]", 10), round(ci.estimate[i]; digits = 3), "  (", round(ci.lower[i]; digits = 3), ", ",
            round(ci.upper[i]; digits = 3), ")  truth ", round(bz; digits = 3), "  covers: ", ci.lower[i] <= bz <= ci.upper[i],
            "    betac[$t] ", round(ci.estimate[j]; digits = 3), "  covers ", round(βc[t]; digits = 3), ": ",
            ci.lower[j] <= βc[t] <= ci.upper[j])
end
# Delta-method Wald for the unique entries of ΛΛ' from the public vcov (pack_lambda order = confint Lambda terms)
iΛ = findall(s -> startswith(s, "Lambda["), terms)
θΛ = ci.estimate[iΛ]
llt(θ) = (L = GLLVModels.unpack_lambda(θ, p, K); S = L * L'; [S[i, j] for j in 1:p for i in j:p])
J = ForwardDiff.jacobian(llt, θΛ); se = sqrt.(max.(diag(J * Matrix(V)[iΛ, iΛ] * J'), 0))
truth = llt(GLLVModels.pack_lambda(Λ)); est = llt(θΛ)
cov = abs.(est .- truth) .<= 1.959964 .* se
println("LLt: ", count(cov), "/", length(cov), " unique entries covered; diag covered ",
        count(cov[[k for (k, (i, j)) in enumerate([(i, j) for j in 1:p for i in j:p]) if i == j]]), "/", p)
