# A2 recovery pilot — existing API only (no src edits). One row per (family, n, p, K_true, rep, K_fit).
# Criteria are computed afterwards from the rows: AIC, BIC log(p·n), BIC log(n); failures kept as rows.
# Usage: julia --project=<repo> pilot.jl <out.csv> <reps> [grid=pre|heavy|full] [task_id]
# With task_id (full grid, 96 cells, 960 tasks): cost-balanced map, see below; <reps> is ignored.
using GLLVModels, Distributions, Random, Statistics, Printf
import GLLVModels.StatsAPI: loglikelihood, dof, aic, bic

out, reps, grid = ARGS[1], parse(Int, ARGS[2]), (length(ARGS) >= 3 ? ARGS[3] : "pre")
fams  = Dict("gaussian" => Normal(), "poisson" => Poisson(), "binomial" => Binomial(), "nb" => NegativeBinomial())
cells = grid == "heavy" ? [(f, 300, 20, 3) for f in ("nb", "binomial")] :
        grid == "pre" ? [(f, 60, 10, k) for f in ("gaussian", "poisson") for k in (1, 2)] :
        [(f, n, p, k) for f in ("gaussian", "poisson", "binomial", "nb") for n in (30, 60, 120, 300) for p in (10, 20) for k in (1, 2, 3)]

function simulate(fam, n, p, K, rng)
    β = fam == "binomial" ? zeros(p) : (fam == "gaussian" ? zeros(p) : fill(log(4.0), p))
    Λ = 0.8 .* randn(rng, p, K)
    η = β .+ Λ * randn(rng, K, n)                       # p × n (species × sites)
    fam == "gaussian" && return η .+ randn(rng, p, n)
    fam == "poisson"  && return [rand(rng, Poisson(exp(x))) for x in η]
    fam == "binomial" && return [rand(rng, Bernoulli(1 / (1 + exp(-x)))) for x in η]
    return [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(x)))) for x in η]
end

reprange = 1:reps
if length(ARGS) >= 4          # cost-balanced task map: slow cells (n·p ≥ 2400) 10 reps/task, others 50; 200 reps each
    t = parse(Int, ARGS[4])
    tasks = [(c, (b * r + 1):((b + 1) * r)) for c in cells for r in ((c[2] * c[3] >= 2400) ? 10 : 50) for b in 0:(200 ÷ r - 1)]
    c, reprange = tasks[t]; cells = [c]
end

open(out, "w") do io
    println(io, "family,n,p,K_true,rep,K_fit,status,converged,loglik,dof,aic,bic_pn,bic_n,secs")
    for (fam, n, p, K) in cells, r in reprange
        rng = MersenneTwister(hash((fam, n, p, K, r)))
        Y = simulate(fam, n, p, K, rng)
        for k in 1:min(K + 2, p - 1)
            t = time()
            try
                fit = fit_gllvm(Y; family = fams[fam], K = k)
                ll, d = loglikelihood(fit), dof(fit)
                cv = hasproperty(fit, :converged) ? fit.converged : missing
                @printf(io, "%s,%d,%d,%d,%d,%d,ok,%s,%.6f,%d,%.6f,%.6f,%.6f,%.2f\n", fam, n, p, K, r, k, cv, ll, d,
                        aic(fit), bic(fit, Y), bic(fit, n), time() - t)
            catch e
                e isa InterruptException && rethrow()
                @printf(io, "%s,%d,%d,%d,%d,%d,fail:%s,,,,,,,%.2f\n", fam, n, p, K, r, k, nameof(typeof(e)), time() - t)
            end
            flush(io)
        end
    end
end
