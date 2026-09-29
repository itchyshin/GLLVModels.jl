# Full Gaussian grid on uncentred data with trait intercepts (#518 + #519 tree).
# Usage: julia --project=<tree> gaussian_full.jl <out.csv> <n>   (one n per process)
using GLLVModels, Random, Distributions, Printf
out, nsel = ARGS[1], parse(Int, ARGS[2])
open(out, "w") do io
    println(io, "n,p,K_true,rep,best_k_bic_sites,best_k_bic,secs")
    for p in (10, 20), K in (1, 2, 3), r in 1:200
        n = nsel
        rng = MersenneTwister(hash(("gaussian", n, p, K, r)))
        Λ = 0.8 .* randn(rng, p, K); Y = Λ * randn(rng, K, n) .+ randn(rng, p, n)
        Y .+= 3.0 .+ randn(MersenneTwister(hash(("gaussian-mean", n, p, K, r))), p)
        t = time()
        s = try select_lv(Y; family = Normal(), Kmax = min(K + 2, p - 1)) catch e; nothing end
        if s === nothing
            @printf(io, "%d,%d,%d,%d,-1,-1,%.2f\n", n, p, K, r, time() - t)
        else
            kb = s.K[argmin(s.bic)]
            @printf(io, "%d,%d,%d,%d,%d,%d,%.2f\n", n, p, K, r, s.best_k, kb, time() - t)
        end
        flush(io)
    end
end
