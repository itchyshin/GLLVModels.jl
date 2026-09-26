const LOG = Ref{Vector{Any}}(Any[])
@eval GLLVModels function _fit_verdict(res)
    gres = Optim.g_residual(res); gt = Optim.g_tol(res); nll = Optim.minimum(res)
    thr = max(gt, gt * abs(nll))
    old = Optim.converged(res); new = old && _gradient_criterion_met(res)
    push!($(LOG)[], (; nll, gres, gt, thr, xc = res.stopped_by.x_converged,
        fc = res.stopped_by.f_converged, gc = res.stopped_by.g_converged,
        it = Optim.iterations(res), old, new))
    return _fit_verdict(Optim.minimum(res), new, Optim.iterations(res))
end
function lowtri(rng, p, K, sd)
    Λ = sd .* randn(rng, p, K)
    for j in 1:K, i in 1:(j-1); Λ[i, j] = 0.0; end
    for j in 1:K; Λ[j, j] = abs(Λ[j, j]) + 0.3sd; end
    Λ
end
function sim(fam, seed; p = 8, n = 100, K = 2)
    rng = MersenneTwister(seed)
    β = 0.5 .+ rand(rng, p); Λ = lowtri(rng, p, K, 0.6); Z = randn(rng, K, n)
    η = β .+ Λ * Z
    fam === :pois && return [rand(rng, Poisson(exp(η[t,i]))) for t in 1:p, i in 1:n]
    fam === :binom && return [rand(rng, Bernoulli(1/(1+exp(-(η[t,i]-1))))) ? 1 : 0 for t in 1:p, i in 1:n]
    fam === :gamma && return [rand(rng, Gamma(2.0, exp(η[t,i])/2.0)) for t in 1:p, i in 1:n]
    fam === :gauss && return η .+ 0.5 .* randn(rng, p, n)
end
function run(label, f)
    empty!(LOG[]); t = @elapsed fit = try f() catch e; println(label, " ERROR ", sprint(showerror, e)[1:min(end,200)]); return end
    for r in LOG[]
        @printf("%-28s nll=%10.3f gres=%9.2e thr=%9.2e ratio=%9.2e x/f/g=%d%d%d it=%4d old=%d new=%d t=%.1fs\n",
            label, r.nll, r.gres, r.thr, r.gres / r.thr, r.xc, r.fc, r.gc, r.it, r.old, r.new, t)
    end
    flush(stdout)
end
