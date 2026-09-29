# Julia twin of ridge_binary_scaled.R: does the loading ridge rescue K recovery on Bernoulli data?
# Usage: julia --project=. ridge_binary_julia.jl <reps> <loading_scale> <out.csv>
using GLLVModels, Distributions, Random, Printf
reps, L, out = parse(Int, ARGS[1]), parse(Float64, ARGS[2]), ARGS[3]
cells = [(60, 10, 2), (120, 10, 2), (120, 20, 2), (120, 20, 3)]
open(out, "w") do io
    println(io, "n,p,K_true,rep,ridge,selected,statuses,secs")
    for (n, p, K) in cells, r in 1:reps
        rng = MersenneTwister(hash((n, p, K, r, L)))
        Λ = L .* randn(rng, p, K); η = Λ * randn(rng, K, n)
        Y = [rand(rng, Bernoulli(1 / (1 + exp(-x)))) for x in η]
        for ridge in (Inf, 2.0)
            t = time()
            sel = try
                select_lv(Y; family = Binomial(), Kmax = K + 2, binary_ridge = ridge)
            catch e
                e isa InterruptException && rethrow()
                nothing
            end
            st = sel === nothing ? "ERROR" : join(string.(getfield.(sel.attempts, :status)), "/")
            @printf(io, "%d,%d,%d,%d,%s,%s,%s,%.1f\n", n, p, K, r, ridge, sel === nothing ? "NA" : sel.best_k, st, time() - t)
            flush(io)
        end
    end
end
