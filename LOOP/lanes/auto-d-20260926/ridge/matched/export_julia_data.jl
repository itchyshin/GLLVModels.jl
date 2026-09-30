# Write the exact Julia binary-ridge datasets (ridge_binary_julia.jl DGP and seeds) for R to fit.
using Random, Distributions, DelimitedFiles
L = 1.5; out = ARGS[1]
for (n, p, K) in [(60, 10, 2), (120, 10, 2), (120, 20, 2), (120, 20, 3)], r in 1:10
    rng = MersenneTwister(hash((n, p, K, r, L)))
    Λ = L .* randn(rng, p, K); η = Λ * randn(rng, K, n)
    Y = [rand(rng, Bernoulli(1 / (1 + exp(-x)))) for x in η]
    writedlm(joinpath(out, "Y_n$(n)_p$(p)_K$(K)_r$(r).csv"), Y, ',')
end
