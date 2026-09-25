# Replicate fit_nb1_gllvm_grouped's optimisation (grouped_dispersion.jl:1558-1612) verbatim
# to read Optim's stopped_by flags, which the fit object does not retain.
using GLLVModels, LinearAlgebra, Random, Distributions, Optim
const G = GLLVModels
function sim(seed; p = 8, n = 60, K = 2, sc = 1.0)
    rng = MersenneTwister(seed)
    Λ = sc .* randn(rng, p, K); β = randn(rng, p) .* 0.5; Z = randn(rng, K, n); x = randn(rng, n)
    η = β .+ Λ * Z
    φ = exp.(randn(rng, p) .* 0.5)
    return [rand(rng, NegativeBinomial(exp(clamp(η[t, s], -5, 5)) / φ[t], 1 / (1 + φ[t]))) for t in 1:p, s in 1:n]
end
for seed in (1, 2)
    Y = sim(seed); p, n = size(Y); K = 2; link = LogLink(); rr = G.rr_theta_len(p, K); group = collect(1:p); G_ = p
    gidx = collect(1:p)
    Yc = Integer.(Y)
    Zemp = [G.linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    β0 = vec(sum(Zemp; dims = 2)) ./ n; Zc = Zemp .- β0; F = svd(Zc)
    Λ0 = zeros(p, K); for j in 1:min(K, length(F.S)); Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n)); end
    θ0 = vcat(β0, G.pack_lambda(Λ0), fill(log(1.0), G_))
    negll = θ -> begin
        β = θ[1:p]; Λ = G.unpack_lambda(θ[(p + 1):(p + rr)], p, K); φg = exp.(θ[(p + rr + 1):(p + rr + G_)])
        v = try -G.nb1_grouped_marginal_loglik_laplace(Yc, Λ, β, φg; link = link, hessian = :observed, maxiter = 100, tol = 1e-9) catch; return 1e12 end
        isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = 1e-5, iterations = 500); autodiff = :finite)
    fit = fit_gllvm(Y; family = G.NB1(), K = 2)
    θ̂ = Optim.minimizer(res)
    println("seed=$seed  Optim.converged=$(Optim.converged(res))  x_conv=$(Optim.x_converged(res))  f_conv=$(Optim.f_converged(res))  g_conv=$(Optim.g_converged(res))  g_residual=$(Optim.g_residual(res))  iters=$(Optim.iterations(res))  stopped_by=$(res.stopped_by)")
    println("   replicate matches public fit: loglik $( -Optim.minimum(res)) vs $(fit.loglik); fit.converged=$(fit.converged)")
    println("   x_abschange=$(Optim.x_abschange(res))  f_abschange=$(Optim.f_abschange(res))")
    # which coordinates carry the gradient
    g = similar(θ̂); h = 1e-5
    for i in eachindex(θ̂); e = h*max(1, abs(θ̂[i])); a = copy(θ̂); b = copy(θ̂); a[i] += e; b[i] -= e; g[i] = (negll(a) - negll(b)) / (2e); end
    idx = sortperm(abs.(g); rev = true)[1:4]
    println("   largest |grad| coords: ", [(i, i <= p ? "beta" : i <= p + rr ? "Lambda" : "logphi", round(g[i]; sigdigits = 3)) for i in idx])
    # is the objective locally noisy? evaluate along the top coordinate
    i = idx[1]; vals = [negll((a = copy(θ̂); a[i] += d; a)) - negll(θ̂) for d in (-1e-4, -1e-5, -1e-6, 1e-6, 1e-5, 1e-4)]
    println("   Δnll along coord $i at steps ±1e-4,±1e-5,±1e-6: ", round.(vals; sigdigits = 3))
end
