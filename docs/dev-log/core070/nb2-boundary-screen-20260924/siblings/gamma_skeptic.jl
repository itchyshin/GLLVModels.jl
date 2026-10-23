# Skeptic re-check of the claimed fit_gamma_gllvm_grouped stall on dataset "seed 2 (alpha_true = 2)".
# Independent of /tmp/claude-503/sibling-screen/gamma.jl: own replication of the fitter's negll,
# plus an INDEPENDENT Laplace evaluator (own damped Newton + Distributions.logpdf) and an
# importance-sampling estimate of the true marginal log-likelihood at both points.
using GLLVModels, Distributions, Random, LinearAlgebra, Printf, Logging
const GM = GLLVModels
const Optim = GM.Optim
println("GLLVModels loaded from: ", pathof(GLLVModels))
println("julia ", VERSION, "; threads ", Threads.nthreads(), "; BLAS threads ", BLAS.get_num_threads())

# ---------------- data: the claim's dataset 2 (p=5, K=2, n=80, alpha_true=2, MersenneTwister(2)) ----
const p, K, n = 5, 2, 80
const Λtrue = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
const βtrue = [0.5, 1.0, -0.5, 1.5, 0.0]
function simulate(seed, α)
    rng = MersenneTwister(seed)
    Z = randn(rng, K, n)
    μ = exp.(βtrue .+ Λtrue * Z)
    return [rand(rng, Gamma(α, μ[t, i] / α)) for t in 1:p, i in 1:n]
end
Y = simulate(2, 2.0)
@printf("Y: size=%s min=%.4e max=%.4e all finite=%s all positive=%s\n",
        string(size(Y)), minimum(Y), maximum(Y), all(isfinite, Y), all(>(0), Y))

# ---------------- my own replication of fit_gamma_gllvm_grouped (defaults, group = 1:p) ----------
group = collect(1:p)
rr = GM.rr_theta_len(p, K)
G = p
gidx = collect(1:p)
msk = GM._resolve_obs_mask(nothing, Y)
Yc = GM._sanitize_missing(Y, 1.0)
Zemp = log.(max.(Yc, 1e-6))
GM._mask_warmstart!(Zemp, msk)
β0 = vec(sum(Zemp; dims = 2)) ./ n
F = svd(Zemp .- β0)
Λ0 = zeros(p, K)
for j in 1:min(K, length(F.S)); Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n)); end
θ0 = vcat(β0, GM.pack_lambda(Λ0), fill(log(2.0), G))
const FIRST = p + rr + 1

function negll(θ)
    β = θ[1:p]; Λ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
    αg = exp.(θ[FIRST:end]); αvec = [αg[gidx[t]] for t in 1:p]
    v = try
        -GM.gamma_grouped_marginal_loglik_laplace(Yc, Λ, β, αvec; link = GM.LogLink(), mask = msk,
                                                  offset = nothing, hessian = :observed,
                                                  maxiter = 100, tol = 1e-9)
    catch
        return 1e12
    end
    return isfinite(v) ? v : 1e12
end
ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
opts = Optim.Options(g_tol = 1e-5, iterations = 500)
runopt(θs) = Optim.optimize(negll, θs, ls, opts; autodiff = :finite)
unpack(θ) = (θ[1:p], GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K), exp.(θ[FIRST:end]))

# ---------------- public fitter ----------------
fit = with_logger(NullLogger()) do
    GM.fit_gamma_gllvm_grouped(Y; K = K, group = group)
end
@printf("\nPUBLIC FIT: loglik=%.6f converged=%s iterations=%d α̂=%s boundary=%s\n",
        fit.loglik, fit.converged, fit.iterations, string(round.(fit.α; sigdigits = 5)),
        string(findall(fit.dispersion_boundary)))
θfit = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.α))

# ---------------- CHECK 1: replicated objective at the fitter's returned parameters ----------------
nf = negll(θfit)
chk1 = abs(nf - (-fit.loglik))
@printf("\nCHECK 1: negll(θ̂_fit)=%.9f  -fit.loglik=%.9f  |diff|=%.3e  -> %s\n",
        nf, -fit.loglik, chk1, chk1 < 1e-6 ? "PASS" : "FAIL")
rep = runopt(θ0)
@printf("         replicated optimizer from θ0: -min=%.9f  |Δ vs fit.loglik|=%.3e  max|θ diff|=%.3e\n",
        -Optim.minimum(rep), abs(-Optim.minimum(rep) - fit.loglik),
        maximum(abs, Optim.minimizer(rep) .- θfit))
println("         replicated run: iterations=", Optim.iterations(rep), " converged=", Optim.converged(rep),
        " g_residual=", Optim.g_residual(rep), " x_abschange=", rep.x_abschange)
@printf("         negll(θ0) = %.6e (the warm start itself)\n", negll(θ0))

# ---------------- alternative (a): θ0 with all log-dispersions = 0 ----------------
θa0 = copy(θ0); θa0[FIRST:end] .= 0.0
ra = runopt(θa0)
θa = Optim.minimizer(ra)
βa, Λa, αa = unpack(θa)
lla = -negll(θa)
println("\nALT (a) fresh logα=0: iterations=", Optim.iterations(ra), " converged=", Optim.converged(ra),
        " g_residual=", Optim.g_residual(ra))
@printf("         ll(a)=%.6f  (claimed -567.232613)  α̂=%s\n", lla, string(round.(αa; sigdigits = 5)))

# ---------------- alternative (c): plain restart from the fitter's own returned point -------------
rc = runopt(θfit)
llc = -Optim.minimum(rc)
@printf("ALT (c) restart from θ̂_fit: ll=%.6f iterations=%d α̂=%s\n", llc, Optim.iterations(rc),
        string(round.(exp.(Optim.minimizer(rc)[FIRST:end]); sigdigits = 5)))

# ---------------- CHECK 2: validity of the alternative point ----------------
v_finite = all(isfinite, θa)
v_pen = negll(θa) < 1e11
v_disp = all(isfinite, αa) && !any(GM._dispersion_group_boundary(αa))
@printf("\nCHECK 2: θ̂_a finite=%s  negll not penalty (%.6f < 1e11)=%s  α finite & inside [1e-6,1e6]=%s\n",
        v_finite, negll(θa), v_pen, v_disp)
@printf("         max|β̂_a|=%.3f  max|Λ̂_a|=%.3f\n", maximum(abs, βa), maximum(abs, Λa))

# inner Laplace mode-search health, per site, replicating the package loop but recording convergence
function inner_health(β, Λ, αvec)
    fams = [Gamma(float(αvec[t]), 1.0) for t in 1:p]
    worst_step = 0.0; worst_stat = 0.0; nonconv = Int[]
    for i in 1:n
        y = Yc[:, i]; z = zeros(K); last = Inf; ok = false
        for it in 1:100
            η = GM._clamp_eta.(β .+ Λ * z); μ = GM._clamp_mu.(fams, GM.linkinv.(Ref(GM.LogLink()), η))
            me = GM.mu_eta.(Ref(GM.LogLink()), η)
            s = GM._glm_score.(fams, μ, ones(Int, p), me, y)
            W = GM._gamma_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(GM.LogLink()))
            Δ = GM._safe_solve(Symmetric(Λ' * (W .* Λ) + I), Λ' * s .- z)
            (Δ === nothing || !all(isfinite, Δ)) && break
            z = z .+ Δ; last = maximum(abs, Δ)
            last < 1e-9 && (ok = true; break)
        end
        # stationarity of the log-posterior at the returned z (analytic Gamma/log score)
        μ = exp.(β .+ Λ * z)
        stat = maximum(abs, Λ' * (αvec .* (y ./ μ .- 1)) .- z)
        ok || push!(nonconv, i)
        worst_step = max(worst_step, last); worst_stat = max(worst_stat, stat)
    end
    return (; nonconv, worst_step, worst_stat)
end
bf, Λf, αf = unpack(θfit)
hf = inner_health(bf, Λf, αf); ha = inner_health(βa, Λa, αa)
println("         inner mode search at θ̂_fit: non-converged sites=", hf.nonconv,
        "  worst last step=", hf.worst_step, "  worst |∇ log-post|=", hf.worst_stat)
println("         inner mode search at θ̂_a  : non-converged sites=", ha.nonconv,
        "  worst last step=", ha.worst_step, "  worst |∇ log-post|=", ha.worst_stat)

# ---------------- independent Laplace evaluator (own damped Newton, Distributions.logpdf) ---------
function laplace_indep(β, Λ, αvec)
    tot = 0.0; zs = Vector{Vector{Float64}}(undef, n); Hs = Vector{Matrix{Float64}}(undef, n)
    for i in 1:n
        y = Yc[:, i]
        h(z) = sum(logpdf(Gamma(αvec[t], exp(β[t] + dot(Λ[t, :], z)) / αvec[t]), y[t]) for t in 1:p) -
               0.5 * dot(z, z)
        z = zeros(K)
        for it in 1:200
            μ = exp.(β .+ Λ * z)
            g = Λ' * (αvec .* (y ./ μ .- 1)) .- z
            H = Λ' * ((αvec .* y ./ μ) .* Λ) + I        # negative Hessian (SPD)
            d = H \ g; step = 1.0; h0 = h(z)
            while h(z .+ step .* d) < h0 - 1e-14 && step > 1e-10; step /= 2; end
            z = z .+ step .* d
            maximum(abs, step .* d) < 1e-12 && break
        end
        μ = exp.(β .+ Λ * z)
        H = Λ' * ((αvec .* y ./ μ) .* Λ) + I
        tot += h(z) - 0.5 * logdet(H)
        zs[i] = z; Hs[i] = H
    end
    return tot, zs, Hs
end
Lf, zf, Hf = laplace_indep(bf, Λf, αf)
La, za, Ha = laplace_indep(βa, Λa, αa)
@printf("\nINDEPENDENT Laplace: at θ̂_fit %.6f (pkg %.6f, |d|=%.2e); at θ̂_a %.6f (pkg %.6f, |d|=%.2e)\n",
        Lf, fit.loglik, abs(Lf - fit.loglik), La, lla, abs(La - lla))

# ---------------- importance-sampling estimate of the TRUE marginal log-likelihood ----------------
function is_marginal(β, Λ, αvec, zs, Hs; M = 4000, seed = 99)
    rng = MersenneTwister(seed); tot = 0.0; ν = 5.0
    for i in 1:n
        y = Yc[:, i]; Σ = Symmetric(inv(Hs[i])); q = MvTDist(ν, zs[i], Matrix(Σ))
        lw = Vector{Float64}(undef, M)
        for m in 1:M
            z = rand(rng, q)
            lw[m] = sum(logpdf(Gamma(αvec[t], exp(β[t] + dot(Λ[t, :], z)) / αvec[t]), y[t]) for t in 1:p) +
                    logpdf(MvNormal(zeros(K), I(K)), z) - logpdf(q, z)
        end
        mx = maximum(lw); tot += mx + log(mean(exp.(lw .- mx)))
    end
    return tot
end
ISf = is_marginal(bf, Λf, αf, zf, Hf); ISa = is_marginal(βa, Λa, αa, za, Ha)
@printf("IMPORTANCE-SAMPLED true marginal loglik: θ̂_fit %.4f ; θ̂_a %.4f ; gain %.4f\n", ISf, ISa, ISa - ISf)

# ---------------- CHECK 3: the gain ----------------
gain = lla - fit.loglik
@printf("\nCHECK 3: ll(a) - fit.loglik = %.6f  (> 1e-3: %s); restart-from-fit gain = %.6f\n",
        gain, gain > 1e-3, llc - fit.loglik)
@printf("fit gradient check: |∇negll(θ̂_fit)|∞ via central FD = ")
fd(f, θ; h = 1e-6) = [(f(θ .+ h .* (1:length(θ) .== j)) - f(θ .- h .* (1:length(θ) .== j))) / (2h) for j in eachindex(θ)]
@printf("%.3e ; at θ̂_a = %.3e\n", maximum(abs, fd(negll, θfit)), maximum(abs, fd(negll, θa)))
println("\nVERDICT inputs: check1=", chk1 < 1e-6, " check2=", v_finite && v_pen && v_disp && isempty(ha.nonconv),
        " check3=", gain > 1e-3)
