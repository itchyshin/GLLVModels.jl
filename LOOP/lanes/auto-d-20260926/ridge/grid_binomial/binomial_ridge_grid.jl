# Binomial grid cells (pilot.jl DGP and seeds) with select_lv's default binary ridge.
# Usage: julia --project=<repo> binomial_ridge_grid.jl <out.csv> <n> <reps>
using GLLVModels, Random, Distributions, Printf
out, n, reps = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3])
open(out, "w") do io
    println(io, "n,p,K_true,rep,best_k_bic_sites,best_k_bic,statuses,secs")
    for p in (10, 20), K in (1, 2, 3), r in 1:reps
        rng = MersenneTwister(hash(("binomial", n, p, K, r)))              # the grid's seed
        Λ = 0.8 .* randn(rng, p, K); η = Λ * randn(rng, K, n)               # the grid's DGP (β = 0)
        Y = [rand(rng, Bernoulli(1 / (1 + exp(-x)))) for x in η]
        t = time()
        s = try select_lv(Y; family = Binomial(), Kmax = min(K + 2, p - 1)) catch e; nothing end
        if s === nothing
            @printf(io, "%d,%d,%d,%d,-1,-1,ERROR,%.1f\n", n, p, K, r, time() - t)
        else
            @printf(io, "%d,%d,%d,%d,%d,%d,%s,%.1f\n", n, p, K, r, s.best_k, s.K[argmin(s.bic)],
                    join(string.(getfield.(s.attempts, :status)), "/"), time() - t)
        end
        flush(io)
    end
end
