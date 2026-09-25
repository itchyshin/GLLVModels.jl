# SKEPTIC re-check of the claimed fit_nb_gllvm_grouped_cov stall on "d6 (r_true = 2, seed 20260930)".
# Written independently of nb2-cov.jl (not included); only the stated design is reused.
#
# Claim under test: public fitter logLik -848.018612027; alternative (a) reaches -847.475409
# under the SAME objective.
#
# Checks:
#  (0) the regenerated dataset is the claimed one (public fit reproduces -848.018612027)
#  (1) my replicated negll (written from src/families/grouped_dispersion.jl:506-570) equals
#      -fit.loglik at the fitter's returned parameters (to 1e-6)
#  (2) the alternative point is a valid parameter vector: all finite, not the 1e12 penalty,
#      r finite; plus robustness: clamp every boundary log r to a MODERATE finite value
#      (log r = 15, 20) and re-evaluate -- the gain must persist at finite dispersions;
#      inner Newton mode converged (re-evaluate with maxiter 2000, tol 1e-13); no eta clamp active.
#  (3) the gain is real: > 1e-3 under the package objective; cross-checked with an
#      independent Laplace implementation (own mode search + O(y) exact NB2 log-pmf) and,
#      as extra information only, an adaptive Gauss-Hermite marginal (not the objective).
using GLLVModels, LinearAlgebra, Random, Printf
import Distributions
const GM = GLLVModels
const Optim = GM.Optim

const p, K, n = 5, 2, 80
const Λtrue = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
const βtrue = [0.0, 0.5, 1.0, 1.5, 2.0]
const γtrue = 0.3
const SEED = 20260930
const RTRUE = 2.0

function simulate_d6()
    rng = MersenneTwister(SEED)
    x = randn(rng, n)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        z = randn(rng, K)
        η = βtrue .+ γtrue * x[s] .+ Λtrue * z
        for t in 1:p
            μ = exp(η[t])
            Y[t, s] = rand(rng, Distributions.NegativeBinomial(RTRUE, RTRUE / (RTRUE + μ)))
        end
    end
    X = reshape(repeat(x', p), p, n, 1) .* 1.0     # X[t,s,1] = x[s]
    return Y, X
end

Y, X = simulate_d6()
@assert all(X[t, s, 1] == X[1, s, 1] for t in 1:p, s in 1:n)

# ---------------- replicated objective (read from the source) ----------------
link = GM.LogLink()
γmask = GM._fixed_zero_mask(nothing, size(X, 3), "γ_fixed")
X_fit, _ = GM._slice_fixed_X(X, γmask)
q = size(X_fit, 3)
rr = GM.rr_theta_len(p, K)
group = collect(1:p)
labels = sort(unique(group)); G = length(labels)
gidx = [findfirst(==(group[t]), labels) for t in 1:p]
msk = GM._resolve_obs_mask(nothing, Y)
Yc = Integer.(GM._sanitize_missing(Y, 0))
Zemp = [GM.linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
GM._mask_warmstart!(Zemp, msk)
β0 = vec(sum(Zemp; dims = 2)) ./ n
F = svd(Zemp .- β0); kk = min(K, length(F.S))
Λ0 = zeros(p, K)
for j in 1:kk
    Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
end
θ0 = vcat(β0, zeros(q), GM.pack_lambda(Λ0), fill(log(10.0), G))
ilr = (p + q + rr + 1):(p + q + rr + G)        # log r indices

unpackθ(θ) = (β = θ[1:p], γ = θ[(p + 1):(p + q)],
              Λ = GM.unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K),
              rg = exp.(θ[ilr]))

function make_negll(; maxiter = 100, tol = 1e-9)
    return function (θ)
        u = unpackθ(θ)
        rvec = [u.rg[gidx[t]] for t in 1:p]
        O = GM._build_offset(X_fit, u.γ)
        v = try
            -GM.nb_grouped_marginal_loglik_laplace(Yc, u.Λ, u.β, rvec; link = link, mask = msk,
                offset = O, hessian = :observed, maxiter = maxiter, tol = tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
end
negll = make_negll()
negll_tight = make_negll(maxiter = 2000, tol = 1e-13)

ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
opts = Optim.Options(g_tol = 1e-5, iterations = 500)
runopt(θs) = Optim.optimize(negll, θs, ls, opts; autodiff = :finite)

# ---------------- independent Laplace + GH (own code) ----------------
# exact NB2 log-pmf via O(y) rising factorial, stable for any r > 0
function nb2_lpmf(y::Int, μ::Float64, r::Float64)
    a = log1p(μ / r)
    s = 0.0
    for j in 0:(y - 1)
        s += log1p(j / r) - a
    end
    return s + y * log(μ) - Distributions.SpecialFunctions.loggamma(y + 1.0) - r * a
end
# BigFloat reference for spot checks
function nb2_lpmf_big(y::Int, μ::Float64, r::Float64)
    setprecision(BigFloat, 2048) do
        R = BigFloat(r); M = BigFloat(μ)
        v = Distributions.SpecialFunctions.loggamma(y + R) - Distributions.SpecialFunctions.loggamma(R) -
            Distributions.SpecialFunctions.loggamma(BigFloat(y) + 1) + R * log(R / (R + M)) + y * log(M / (R + M))
        Float64(v)
    end
end

# site log-joint g(z) = Σ_t log f(y_t | μ_t) - z'z/2 ; mode by damped Newton on exact Hessian
function site_mode(y, β, off, Λ, rv)
    z = zeros(K)
    g(z) = sum(nb2_lpmf(y[t], exp(β[t] + off[t] + dot(Λ[t, :], z)), rv[t]) for t in 1:p) - 0.5 * dot(z, z)
    for it in 1:500
        η = β .+ off .+ Λ * z; μ = exp.(η)
        s = (y .- μ) ./ (1 .+ μ ./ rv)                       # dℓ/dη
        W = μ .* (1 .+ y ./ rv) ./ (1 .+ μ ./ rv) .^ 2        # -d²ℓ/dη²
        grad = Λ' * s .- z
        H = Symmetric(Λ' * (W .* Λ) + I)
        Δ = H \ grad
        step = 1.0; g0 = g(z)
        while g(z .+ step .* Δ) < g0 - 1e-14 && step > 1e-10
            step /= 2
        end
        z = z .+ step .* Δ
        norm(grad) < 1e-12 && break
    end
    η = β .+ off .+ Λ * z; μ = exp.(η)
    W = μ .* (1 .+ y ./ rv) ./ (1 .+ μ ./ rv) .^ 2
    H = Symmetric(Λ' * (W .* Λ) + I)
    return z, g(z), H, maximum(abs, η)
end

function indep_laplace(θ)
    u = unpackθ(θ); rv = [u.rg[gidx[t]] for t in 1:p]
    O = GM._build_offset(X_fit, u.γ)
    tot = 0.0; ηmax = 0.0
    for s in 1:n
        z, gz, H, em = site_mode(view(Yc, :, s), u.β, view(O, :, s), u.Λ, rv)
        tot += gz - 0.5 * logdet(H); ηmax = max(ηmax, em)
    end
    return tot, ηmax
end

# Gauss-Hermite nodes (physicists') by Golub-Welsch
function gh(m)
    J = SymTridiagonal(zeros(m), [sqrt(k / 2) for k in 1:(m - 1)])
    E = eigen(J)
    return E.values, sqrt(pi) .* E.vectors[1, :] .^ 2
end
const GHX, GHW = gh(20)
function agh_marginal(θ)
    u = unpackθ(θ); rv = [u.rg[gidx[t]] for t in 1:p]
    O = GM._build_offset(X_fit, u.γ)
    tot = 0.0
    for s in 1:n
        y = view(Yc, :, s); off = view(O, :, s)
        ẑ, gẑ, H, _ = site_mode(y, u.β, off, u.Λ, rv)
        L = cholesky(H).L
        Linv_t = inv(Matrix(L))'                # z = ẑ + √2 L^{-T} x
        acc = 0.0
        for i in eachindex(GHX), j in eachindex(GHX)
            x = [GHX[i], GHX[j]]
            z = ẑ .+ sqrt(2) .* (Linv_t * x)
            gz = sum(nb2_lpmf(y[t], exp(u.β[t] + off[t] + dot(u.Λ[t, :], z)), rv[t]) for t in 1:p) - 0.5 * dot(z, z)
            acc += GHW[i] * GHW[j] * exp(gz - gẑ + dot(x, x))
        end
        # ∫ e^{g(z)} (2π)^{-K/2} dz = (2π)^{-K/2} 2^{K/2} det(L)^{-1} e^{g(ẑ)} Σ w e^{g-gẑ+x'x}
        tot += gẑ + log(acc) - (K / 2) * log(2pi) + (K / 2) * log(2) - logdet(L)
    end
    return tot
end

θ_of(fit) = vcat(fit.β, fit.γ[.!fit.γ_fixed], GM.pack_lambda(fit.Λ), log.(fit.r_group))
fmt(v) = join([@sprintf("%.4g", x) for x in v], ", ")

println("SKEPTIC d6 re-check  Julia ", VERSION, " threads=", Threads.nthreads())
println("Y column sums per trait: ", vec(sum(Y; dims = 2)), "   mean x = ", @sprintf("%.6f", sum(X[1, :, 1]) / n))

# ---------------- (0) public fit ----------------
t_fit = @elapsed fit = GM.fit_nb_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
@printf("(0) public fit loglik = %.9f  (claimed -848.018612027; |diff| = %.2e)  conv=%s iters=%d  %.1fs\n",
        fit.loglik, abs(fit.loglik + 848.018612027), fit.converged, fit.iterations, t_fit)
println("    r_group = [", fmt(fit.r_group), "]  boundary = ", findall(fit.dispersion_boundary),
        "  γ = ", fmt(fit.γ))
θf = θ_of(fit)

# ---------------- (1) harness validity ----------------
vf = -negll(θf)
@printf("(1) replicated -negll(θ̂_fit) = %.9f   |diff vs fit.loglik| = %.3e  -> %s\n",
        vf, abs(vf - fit.loglik), abs(vf - fit.loglik) <= 1e-6 ? "HARNESS VALID" : "HARNESS INVALID")
# also the optimizer: replicate the whole run from θ0
t_rep = @elapsed res0 = runopt(θ0)
@printf("    replicated optimizer from θ0: loglik = %.9f (|diff| %.2e)  converged=%s iters=%d  %.1fs\n",
        -Optim.minimum(res0), abs(-Optim.minimum(res0) - fit.loglik), Optim.converged(res0),
        Optim.iterations(res0), t_rep)

# ---------------- alternatives ----------------
θa0 = copy(θ0); θa0[ilr] .= 0.0
t_a = @elapsed resa = runopt(θa0)
θa = Optim.minimizer(resa)
bd = findall(fit.dispersion_boundary)
θb0 = copy(θf); θb0[ilr[bd]] .= 0.0
t_b = @elapsed resb = runopt(θb0)
θb = Optim.minimizer(resb)
for (nm, rs, θx, tt) in (("a (θ0, all log r = 0)", resa, θa, t_a), ("b (θ̂, boundary log r = 0)", resb, θb, t_b))
    @printf("alt %s: loglik = %.9f  gain = %+.6f  conv=%s iters=%d  %.1fs\n", nm, -Optim.minimum(rs),
            -Optim.minimum(rs) - fit.loglik, Optim.converged(rs), Optim.iterations(rs), tt)
    println("    r = [", fmt(exp.(θx[ilr])), "]  boundary = ",
            findall(GM._dispersion_group_boundary(exp.(θx[ilr]))), "  γ = ", fmt(θx[(p + 1):(p + q)]))
end
θalt = -Optim.minimum(resa) >= -Optim.minimum(resb) ? θa : θb
ll_alt = -negll(θalt)

# ---------------- (2) validity of the alternative point ----------------
println("\n(2) validity of alternative point")
u = unpackθ(θalt)
@printf("    all θ finite: %s; all r finite: %s; negll = %.6f (penalty 1e12? %s)\n",
        all(isfinite, θalt), all(isfinite, u.rg), negll(θalt), negll(θalt) >= 1e11)
println("    β = [", fmt(u.β), "]  Λ = ", fmt(vec(u.Λ)), "  max|Λ| = ", @sprintf("%.3g", maximum(abs, u.Λ)))
@printf("    tight inner Newton (maxiter 2000, tol 1e-13): alt %.9f (Δ %.2e); fit %.9f (Δ %.2e)\n",
        -negll_tight(θalt), -negll_tight(θalt) - ll_alt, -negll_tight(θf), -negll_tight(θf) - vf)
for c in (10.0, 15.0, 20.0, 25.0)
    θac = copy(θalt); θfc = copy(θf)
    for g in 1:G
        θac[ilr[g]] > c && (θac[ilr[g]] = c)
        θfc[ilr[g]] > c && (θfc[ilr[g]] = c)
    end
    la = -negll(θac); lf = -negll(θfc)
    @printf("    clamp log r ≤ %4.1f (r ≤ %.3g):  alt %.6f  fit %.6f  gain %+.6f   alt r=[%s]\n",
            c, exp(c), la, lf, la - lf, fmt(exp.(θac[ilr])))
end

# ---------------- (3) independent objective + extra ----------------
println("\n(3) independent Laplace implementation (own mode search, O(y) exact NB2 log-pmf)")
# spot-check own lpmf vs BigFloat at extreme r
for (yy, μμ, rr_) in ((0, 2.3, 1e44), (7, 4.1, 1e44), (13, 6.0, 4.2e9), (3, 1.2, 1.353), (25, 9.0, 5.5e22))
    @printf("    lpmf check y=%d μ=%.2f r=%.3g: own %.12f  big %.12f  pkg %.12f\n", yy, μμ, rr_,
            nb2_lpmf(yy, μμ, rr_), nb2_lpmf_big(yy, μμ, rr_), GM._nb2_logpdf_mean(μμ, rr_, yy))
end
lf_ind, ηf = indep_laplace(θf); la_ind, ηa = indep_laplace(θalt)
@printf("    fit point: indep Laplace %.9f (pkg %.9f, Δ %.2e), max|η| %.2f\n", lf_ind, vf, lf_ind - vf, ηf)
@printf("    alt point: indep Laplace %.9f (pkg %.9f, Δ %.2e), max|η| %.2f\n", la_ind, ll_alt, la_ind - ll_alt, ηa)
@printf("    indep gain = %+.6f   pkg gain = %+.6f\n", la_ind - lf_ind, ll_alt - vf)
t_gh = @elapsed begin
    gf = agh_marginal(θf); ga = agh_marginal(θalt)
end
@printf("    [extra, NOT the objective] 20x20 adaptive GH marginal: fit %.6f  alt %.6f  gain %+.6f  (%.1fs)\n",
        gf, ga, ga - gf, t_gh)

# gradient norms (finite difference) at both points, non-boundary coordinates
function fdgrad(f, θ; h = 1e-5)
    g = similar(θ)
    for i in eachindex(θ)
        e = zeros(length(θ)); e[i] = h
        g[i] = (f(θ .+ e) - f(θ .- e)) / (2h)
    end
    g
end
for (nm, θx) in (("fit", θf), ("alt", θalt))
    gx = fdgrad(negll, θx)
    nb = findall(GM._dispersion_group_boundary(exp.(θx[ilr])))
    keep = setdiff(1:length(θx), ilr[nb])
    @printf("    %s: max|∇negll| over non-boundary coords = %.2e; over boundary log r = %s\n",
            nm, maximum(abs, gx[keep]), fmt(gx[ilr[nb]]))
end

println("\nVERDICT INPUTS: harness_valid=", abs(vf - fit.loglik) <= 1e-6,
        "  alt_valid=", all(isfinite, θalt) && negll(θalt) < 1e11,
        "  gain=", @sprintf("%.6f", ll_alt - fit.loglik), "  gain>1e-3=", ll_alt - fit.loglik > 1e-3)
