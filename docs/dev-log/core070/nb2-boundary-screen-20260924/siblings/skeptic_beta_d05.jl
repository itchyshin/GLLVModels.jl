# Skeptic re-check of the claim: fit_beta_gllvm_grouped stalls on dataset d05
# (fitter logLik 265.976379 vs alternative a_fresh_logphi0 272.609413, same objective).
# Written independently of /tmp/claude-503/sibling-screen/beta.jl; only the dataset
# definition (design + MersenneTwister(20260924 + 5)) is taken from the claim, because
# the claim is about that specific dataset.
#
# Checks:
#  (1) replicated objective == -fit.loglik at the fitter's returned θ (tol 1e-6), and
#      == the public beta_grouped_marginal_loglik_laplace evaluated directly;
#  (2) alternative point: all finite, not the 1e12 penalty, φ finite and inside [1e-6,1e6];
#  (3) gain > 1e-3.
# Extra (skeptic) checks: inner Newton converged at both points (tighter tol does not move
# the value); an INDEPENDENT Laplace computation (own mode search + finite-difference
# Hessian using Distributions.Beta directly) reproduces the package value at both points
# and the Laplace Hessian is PD; exact marginal by adaptive Gauss-Hermite at both points;
# gradient at the fitter point.

using GLLVModels, LinearAlgebra, Random, Printf
import Distributions
const GM = GLLVModels
const Optim = GM.Optim

const p, K, n = 5, 2, 80
const Λtrue = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
const βtrue = [-1.5, -0.75, 0.0, 0.75, 1.5]
const φtrue = 2.0
logistic(x) = 1 / (1 + exp(-x))

function sim_d05()
    rng = MersenneTwister(20260924 + 5)
    Y = Matrix{Float64}(undef, p, n)
    for i in 1:n
        z = randn(rng, K)
        η = βtrue .+ Λtrue * z
        for t in 1:p
            μ = logistic(η[t])
            Y[t, i] = clamp(rand(rng, Distributions.Beta(μ * φtrue, (1 - μ) * φtrue)), 1e-12, 1 - 1e-12)
        end
    end
    Y
end

Y = sim_d05()
@printf("data: size=%s  min=%.3e  max=1-%.3e  all in (0,1)=%s\n", string(size(Y)),
        minimum(Y), 1 - maximum(Y), all(0 .< Y .< 1))

# ---------- objective, written from src/families/grouped_dispersion.jl:748-793 ----------
const rr = GM.rr_theta_len(p, K)
const G = p
const gidx = collect(1:p)
const link = GM.LogitLink()
msk = GM._resolve_obs_mask(nothing, Y)
Yc = GM._sanitize_missing(Y, 0.5)
Zemp = [GM.linkfun(link, clamp(float(Yc[t, i]), 1e-6, 1 - 1e-6)) for t in 1:p, i in 1:n]
GM._mask_warmstart!(Zemp, msk)
β0 = vec(sum(Zemp; dims = 2)) ./ n
F = svd(Zemp .- β0)
Λ0 = zeros(p, K)
for j in 1:min(K, length(F.S))
    Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
end
θ0 = vcat(β0, GM.pack_lambda(Λ0), fill(log(10.0), G))
const IL = (p + rr + 1):(p + rr + G)   # log-φ block

function negll(θ; maxiter = 100, tol = 1e-9)
    β = θ[1:p]
    Λ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
    φg = exp.(θ[IL])
    φvec = [φg[gidx[t]] for t in 1:p]
    v = try
        -GM.beta_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link = link, mask = msk,
                                                 offset = nothing, hessian = :observed,
                                                 maxiter = maxiter, tol = tol)
    catch
        return 1e12
    end
    isfinite(v) ? v : 1e12
end
ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
opts = Optim.Options(g_tol = 1e-5, iterations = 500)

# ---------- public fitter ----------
fit = GM.fit_beta_gllvm_grouped(Y; K = K, group = collect(1:p))
θF = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.φ))
@printf("\nFITTER: loglik=%.6f converged=%s iters=%d phi=%s boundary=%s\n", fit.loglik,
        fit.converged, fit.iterations, string(round.(fit.φ; sigdigits = 5)),
        string(findall(fit.dispersion_boundary)))
@printf("  unpack(pack(Λ̂)) == Λ̂: %s\n", GM.unpack_lambda(GM.pack_lambda(fit.Λ), p, K) == fit.Λ)
nF = negll(θF)
direct_F = GM.beta_grouped_marginal_loglik_laplace(Y, fit.Λ, fit.β, fit.φ[gidx])
@printf("  CHECK1 negll(θF)=%.9f  -fit.loglik=%.9f  |diff|=%.3e  (direct public call: %.9f)\n",
        nF, -fit.loglik, abs(nF + fit.loglik), direct_F)

# ---------- alternative (a): fitter θ0 with all log φ = 0 ----------
θa0 = copy(θ0); θa0[IL] .= 0.0
ra = Optim.optimize(negll, θa0, ls, opts; autodiff = :finite)
θA = Optim.minimizer(ra)
llA = -Optim.minimum(ra)
ΛA = GM.unpack_lambda(θA[(p + 1):(p + rr)], p, K); βA = θA[1:p]; φA = exp.(θA[IL])
direct_A = GM.beta_grouped_marginal_loglik_laplace(Y, ΛA, βA, φA[gidx])
@printf("\nALT (a): loglik=%.6f  conv=%s iters=%d\n", llA, Optim.converged(ra), Optim.iterations(ra))
@printf("  CHECK2 all finite=%s  negll(θA)=%.6f (<1e11: %s)  phi=%s  in[1e-6,1e6]=%s\n",
        all(isfinite, θA), negll(θA), negll(θA) < 1e11, string(round.(φA; sigdigits = 5)),
        !any(GM._dispersion_group_boundary(φA)))
@printf("  direct public call at θA: %.9f\n", direct_A)
@printf("  beta=%s\n  Lambda=%s\n", string(round.(βA; digits = 4)), string(round.(ΛA; digits = 4)))
gain = llA - fit.loglik
@printf("  CHECK3 gain = %.6f  (> 1e-3: %s)\n", gain, gain > 1e-3)

# ---------- inner-Newton convergence at both points ----------
for (lab, θ) in (("fitter", θF), ("alt", θA))
    v0 = -negll(θ); v1 = -negll(θ; maxiter = 2000, tol = 1e-13)
    @printf("  inner Newton [%s]: default=%.9f  tight=%.9f  |diff|=%.2e\n", lab, v0, v1, abs(v0 - v1))
end

# ---------- independent Laplace + exact marginal (adaptive GH) ----------
function gh(m)
    J = SymTridiagonal(zeros(m), [sqrt(k / 2) for k in 1:(m - 1)])
    E = eigen(J)
    E.values, sqrt(pi) .* E.vectors[1, :] .^ 2
end
const GX, GW = gh(30)

function site_ell(z, y, β, Λ, φv)
    η = clamp.(β .+ Λ * z, -30.0, 30.0)
    μ = clamp.(logistic.(η), 1e-6, 1 - 1e-6)
    s = 0.0
    for t in 1:p
        s += Distributions.logpdf(Distributions.Beta(μ[t] * φv[t], (1 - μ[t]) * φv[t]), y[t])
    end
    s
end

function indep_eval(θ)
    β = θ[1:p]; Λ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K); φv = exp.(θ[IL])[gidx]
    lap = 0.0; ex = 0.0; minev = Inf; maxgz = 0.0
    for i in 1:n
        y = Y[:, i]
        q(z) = site_ell(z, y, β, Λ, φv) - 0.5 * dot(z, z)
        r = Optim.optimize(z -> -q(z), zeros(K), Optim.LBFGS(),
                           Optim.Options(g_tol = 1e-11, iterations = 1000); autodiff = :finite)
        ẑ = Optim.minimizer(r)
        h = 1e-4
        H = zeros(K, K)
        for a in 1:K, b in 1:K
            ea = zeros(K); ea[a] = h; eb = zeros(K); eb[b] = h
            H[a, b] = -(q(ẑ + ea + eb) - q(ẑ + ea - eb) - q(ẑ - ea + eb) + q(ẑ - ea - eb)) / (4h^2)
        end
        H = Symmetric((H + H') / 2)
        ev = eigvals(H); minev = min(minev, minimum(ev))
        lap += q(ẑ) - 0.5 * logdet(H)
        # exact: log ∫ exp(ℓ(z)) N(z;0,I) dz, adaptive GH around ẑ with L = chol(H^{-1})
        L = cholesky(inv(H)).L
        terms = Float64[]
        for j1 in eachindex(GX), j2 in eachindex(GX)
            x = [GX[j1], GX[j2]]
            z = ẑ .+ sqrt(2) .* (L * x)
            push!(terms, log(GW[j1]) + log(GW[j2]) + dot(x, x) + q(z) - (K / 2) * log(2pi))
        end
        mx = maximum(terms)
        ex += mx + log(sum(exp.(terms .- mx))) + logdet(L) + (K / 2) * log(2)
    end
    lap, ex, minev
end

println("\nINDEPENDENT Laplace (own mode search + FD Hessian, Distributions.Beta) and exact (adaptive GH 30x30):")
res = Dict{String, Any}()
for (lab, θ) in (("fitter", θF), ("alt", θA))
    lap, ex, minev = indep_eval(θ)
    res[lab] = (lap, ex)
    @printf("  [%s] pkg=%.6f  indep_laplace=%.6f (|diff|=%.2e)  min eig(H)=%.4g  exact_GH=%.6f\n",
            lab, -negll(θ), lap, abs(lap + negll(θ)), minev, ex)
end
@printf("  gain under indep Laplace = %.6f ; gain under exact GH marginal = %.6f\n",
        res["alt"][1] - res["fitter"][1], res["alt"][2] - res["fitter"][2])

# ---------- gradient at fitter point, and a continuation ----------
function cgrad(f, θ; h = 1e-5)
    g = similar(θ)
    for j in eachindex(θ)
        e = zeros(length(θ)); e[j] = h
        g[j] = (f(θ .+ e) - f(θ .- e)) / (2h)
    end
    g
end
gF = cgrad(negll, θF); gA = cgrad(negll, θA)
@printf("\nmax|grad negll| at fitter=%.4g (component %d)  at alt=%.4g\n", maximum(abs, gF),
        argmax(abs.(gF)), maximum(abs, gA))
rc = Optim.optimize(negll, θF, Optim.LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 500);
                    autodiff = :finite)
@printf("continuation from fitter point with default LBFGS (HagerZhang): loglik=%.6f iters=%d phi=%s\n",
        -Optim.minimum(rc), Optim.iterations(rc), string(round.(exp.(Optim.minimizer(rc)[IL]); sigdigits = 4)))
@printf("\nVERDICT: check1=%s check2=%s check3=%s\n", abs(nF + fit.loglik) <= 1e-6,
        all(isfinite, θA) && negll(θA) < 1e11 && !any(GM._dispersion_group_boundary(φA)), gain > 1e-3)
