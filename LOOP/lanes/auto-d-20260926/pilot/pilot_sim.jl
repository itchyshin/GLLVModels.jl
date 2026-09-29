# simulate() copied from pilot.jl for reuse
function simulate(fam, n, p, K, rng)
    β = fam == "binomial" ? zeros(p) : (fam == "gaussian" ? zeros(p) : fill(log(4.0), p))
    Λ = 0.8 .* randn(rng, p, K)
    η = β .+ Λ * randn(rng, K, n)                       # p × n (species × sites)
    fam == "gaussian" && return η .+ randn(rng, p, n)
    fam == "poisson"  && return [rand(rng, Poisson(exp(x))) for x in η]
    fam == "binomial" && return [rand(rng, Bernoulli(1 / (1 + exp(-x)))) for x in η]
    return [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(x)))) for x in η]
end
