# Pre-run for the Gaussian grid re-run after #519. Usage: julia --project=<tree> gaussian_prerun.jl <label> <centred|uncentred> <out.csv>
using GLLVModels, Random, Distributions, Printf
label, mode, out = ARGS
open(out, "w") do io
    println(io, "label,data,n,p,K_true,rep,best_k,secs")
    for (n, p, K) in [(30, 10, 2), (60, 20, 3), (120, 10, 1), (120, 20, 2), (300, 20, 3)], r in 1:10
        rng = MersenneTwister(hash(("gaussian", n, p, K, r)))            # the grid's seed
        Λ = 0.8 .* randn(rng, p, K); Y = Λ * randn(rng, K, n) .+ randn(rng, p, n)   # the grid's DGP (β = 0)
        if mode == "uncentred"
            Y .+= 3.0 .+ randn(MersenneTwister(hash(("gaussian-mean", n, p, K, r))), p)
        end
        t = time()
        k = try select_lv(Y; family = Normal(), Kmax = min(K + 2, p - 1)).best_k catch e; -1 end
        @printf(io, "%s,%s,%d,%d,%d,%d,%d,%.2f\n", label, mode, n, p, K, r, k, time() - t); flush(io)
    end
end
