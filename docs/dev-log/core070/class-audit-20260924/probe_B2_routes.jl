using GLLVModels, LinearAlgebra, Random, Distributions, StatsModels
const G = GLLVModels
function fdgrad(f, θ; h = 1e-5)
    g = similar(θ)
    for i in eachindex(θ)
        e = h * max(1.0, abs(θ[i])); θp = copy(θ); θm = copy(θ); θp[i] += e; θm[i] -= e
        g[i] = (f(θp) - f(θm)) / (2e)
    end
    return g
end
report(route, seed, fit, gn, el) = println(rpad(route, 26), " seed=$seed  type=$(nameof(typeof(fit)))  converged=$(fit.converged)  loglik=$(round(hasproperty(fit, :loglik) ? fit.loglik : fit.logLik; digits=3))  max|grad|=$(round(gn; sigdigits=3))  t=$(round(el; digits=1))s  ",
    fit.converged && gn > 1e-3 ? "DISHONEST(conv=true, |g|>100*g_tol)" : "")
for seed in 1:6
    rng = MersenneTwister(100 + seed); p, n, K = 10, 80, 2
    Λ = 1.5 .* randn(rng, p, K); β = randn(rng, p); Z = randn(rng, K, n)
    # Gaussian default route (bridge "gaussian", fit_gllvm(Normal())): gradient of its own packed nll
    Y = β .+ Λ * Z .+ 0.3 .* randn(rng, p, n)
    t0 = time(); f = fit_gllvm(Y; family = Normal(), K = 2); el = time() - t0
    report("gaussian_default", seed, f, G._bridge_gradient_max_gaussian(f, Y, nothing, nothing), el)
    flush(stdout)
end
for seed in 1:6
    # harsher Poisson + X (formula route -> fit_gllvm_cov): larger loadings, sparse counts
    rng = MersenneTwister(200 + seed); p, n, K = 10, 80, 2
    Λ = 2.0 .* randn(rng, p, K); β = randn(rng, p) .- 1.0; Z = randn(rng, K, n); x = randn(rng, n)
    Y = [rand(rng, Poisson(exp(clamp(β[t] + dot(Λ[t, :], Z[:, s]) + 0.5 * x[s], -6, 6)))) for t in 1:p, s in 1:n]
    X = reshape(repeat(x', p), p, n, 1)
    t0 = time(); f = gllvm(@formula(y ~ 1 + x), Y, (x = x,); family = Poisson(), K = 2); el = time() - t0
    ci = G._family_ci(f, Float64.(Y); X = X)
    report("poisson_formula_X_harsh", seed, f, maximum(abs, fdgrad(ci.nll, ci.θ)), el)
    flush(stdout)
end
