using GLLVModels, LinearAlgebra, Random, Distributions
const G = GLLVModels
function fdgrad(f, θ; h = 1e-5)
    g = similar(θ)
    for i in eachindex(θ)
        e = h * max(1.0, abs(θ[i])); θp = copy(θ); θm = copy(θ); θp[i] += e; θm[i] -= e
        g[i] = (f(θp) - f(θm)) / (2e)
    end
    return g
end
logistic(x) = 1 / (1 + exp(-x))
function one(route, seed)
    rng = MersenneTwister(300 + seed); p, n, K = 8, 60, 2
    Λ = randn(rng, p, K); β = randn(rng, p) .* 0.5; Z = randn(rng, K, n); η = β .+ Λ * Z
    πz = logistic.(randn(rng, p) .- 0.5)
    t0 = time()
    f, Y = if route == :gamma_default
        Y = [rand(rng, Gamma(2.0, exp(clamp(η[t, s], -5, 5)) / 2.0)) for t in 1:p, s in 1:n]
        fit_gllvm(Y; family = Gamma(), K = 2), Y
    elseif route == :exponential_default
        Y = [rand(rng, Exponential(exp(clamp(η[t, s], -5, 5)))) for t in 1:p, s in 1:n]
        fit_gllvm(Y; family = Exponential(), K = 2), Y
    elseif route == :hurdlepoisson_default
        Y = [rand(rng) < πz[t] ? 0 : (x = 0; while x == 0; x = rand(rng, Poisson(max(exp(clamp(η[t, s], -5, 5)), 0.05))); end; x) for t in 1:p, s in 1:n]
        fit_gllvm(Y; family = G.HurdlePoisson(), K = 2), Y
    elseif route == :deltagamma_default
        Y = [rand(rng) < πz[t] ? 0.0 : rand(rng, Gamma(2.0, exp(clamp(η[t, s], -5, 5)) / 2.0)) for t in 1:p, s in 1:n]
        fit_gllvm(Y; family = G.DeltaGamma(), K = 2), Y
    elseif route == :zinb_default
        Y = [rand(rng) < πz[t] ? 0 : rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(clamp(η[t, s], -5, 5))))) for t in 1:p, s in 1:n]
        fit_gllvm(Y; family = G.ZINegBin(), K = 2), Y
    end
    el = time() - t0
    ci = G._family_ci(f, Float64.(Y))
    gn = maximum(abs, fdgrad(ci.nll, ci.θ))
    println(rpad(string(route), 24), " seed=$seed  type=$(nameof(typeof(f)))  converged=$(f.converged)  iters=$(f.iterations)  loglik=$(round(f.loglik; digits=3))  nll_adapter=$(round(ci.nll(ci.θ); digits=3))  max|grad|=$(round(gn; sigdigits=3))  t=$(round(el; digits=1))s  ",
            f.converged && gn > 1e-3 ? "DISHONEST(conv=true, |g|>100*g_tol)" : "")
    flush(stdout)
end
for r in (:gamma_default, :exponential_default, :hurdlepoisson_default, :deltagamma_default, :zinb_default), s in 1:3
    try one(r, s) catch e; println(r, " seed=$s ERROR ", sprint(showerror, e)[1:min(end, 250)]); end
end
