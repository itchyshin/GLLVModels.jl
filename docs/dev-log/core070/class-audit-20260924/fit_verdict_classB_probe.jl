# Audit harness: _fit_verdict(res) (src/fit_verdict.jl:53-55), the shared verdict
# behind ~85 fitters. Suspected CLASS B (converged=true while gradient >> g_tol),
# with a CLASS A per-site mode-search census on the same fits.
#
# Routes probed (bridge defaults, complete data, no X): gaussian (fit_gaussian_gllvm on
# centred Y), poisson, binomial (Bernoulli), zip, nb1 (fit_nb1_gllvm_grouped, group=1:p),
# ordinal (fit_ordinal_gllvm_pertrait, logit), truncated_poisson.
#
# For each fit:
#   VALIDITY  replicate the fitter's negll (same kernel, same kwargs) and its warm start,
#             re-run the SAME Optim call, check (i) replicated minimizer == fitter's params,
#             (ii) negll_rep(theta_hat) == -fit.loglik.
#   CLASS B   stopped_by flags (x/f/g), Optim.g_residual, FD gradient (two steps) and the
#             analytic gradient where the fitter has one; restart from theta_hat and a fresh
#             start from the true parameters; flag converged=true with max|g| > 100*g_tol.
#   CLASS A   instrumented copy of the per-site mode search (bitwise-checked against the
#             package's own mode), run at theta0 and theta_hat; status per site + the
#             stationarity residual |Lambda's(z) - z|_inf at the package's own mode.
#
# Run: cd <worktree> && JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#      julia +1.10.12 --startup-file=no --project=test/parity <this file>

using GLLVModels, LinearAlgebra, Random, Statistics, Distributions, Printf
const GM = GLLVModels
const Optim = GM.Optim
const FD = GM.ForwardDiff
const T0 = time()
const SOFT = parse(Float64, get(ENV, "AUDIT_SOFT_SECONDS", "480"))
elapsed() = time() - T0

maxabs(x) = isempty(x) ? 0.0 : maximum(abs, x)
logistic(x) = 1 / (1 + exp(-x))

function fdgrad(f, θ; h = 1e-5)
    g = similar(θ, Float64)
    for i in eachindex(θ)
        hi = h * max(1.0, abs(θ[i]))
        θp = copy(θ); θp[i] += hi
        θm = copy(θ); θm[i] -= hi
        g[i] = (f(θp) - f(θm)) / (2hi)
    end
    return g
end

bt_ls() = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
fd_run(negll, θ0; g_tol = 1e-5, iterations = 500) =
    Optim.optimize(negll, θ0, bt_ls(), Optim.Options(g_tol = g_tol, iterations = iterations);
                   autodiff = :finite)
an_run(negll, ag, θ0; g_tol = 1e-5, iterations = 500) =
    GM._optimize_with_analytic(negll, ag, θ0, bt_ls(),
                               Optim.Options(g_tol = g_tol, iterations = iterations))

stopflags(res) = (x = res.stopped_by.x_converged, f = res.stopped_by.f_converged,
                  g = res.stopped_by.g_converged, ls_failed = res.stopped_by.ls_failed,
                  gres = Optim.g_residual(res), it = Optim.iterations(res),
                  conv = Optim.converged(res))

function lowtri(rng, p, K, sd)
    Λ = sd .* randn(rng, p, K)
    for j in 1:K, i in 1:(j - 1)
        Λ[i, j] = 0.0
    end
    for j in 1:K
        Λ[j, j] = abs(Λ[j, j]) + 0.3 * sd
    end
    return Λ
end

# --------------------------------------------------------------------------------------
# CLASS A: instrumented mode searches (status per site).
# --------------------------------------------------------------------------------------

# Copy of families/laplace.jl::_laplace_mode (no workspace), with an exit status.
function im_generic(fam, y, n, Λ, β, link; maxiter = 100, tol = 1e-9)
    p, K = size(Λ); z = zeros(K); restarted = false; status = :maxiter; it = 0
    while it < maxiter
        it += 1
        η = GM._clamp_eta.(β .+ Λ * z)
        μ = GM._clamp_mu.(Ref(fam), GM.linkinv.(Ref(link), η))
        me = GM.mu_eta.(Ref(link), η)
        s = GM._glm_score.(Ref(fam), μ, n, me, y)
        W = GM._glm_weight.(Ref(fam), μ, n, me)
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = GM._safe_solve(A, Λ' * s .- z)
        if Δ === nothing || !all(isfinite, Δ)
            if !restarted
                z = zeros(K); restarted = true; continue
            end
            status = :solve_fail; break
        end
        step_taken = 1.0
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        elseif !GM._laplace_mode_should_backtrack(fam)
            z = z .+ Δ
        else
            q0 = GM._laplace_mode_logpost(fam, y, n, Λ, β, link, z)
            if isfinite(q0)
                accepted = false; step = 1.0
                for _ in 1:30
                    zt = z .+ step .* Δ
                    q1 = GM._laplace_mode_logpost(fam, y, n, Λ, β, link, zt)
                    if isfinite(q1) && q1 >= q0
                        z = zt; step_taken = step; accepted = true; break
                    end
                    step *= 0.5
                end
                if !accepted
                    status = :backtrack_fail; break
                end
            else
                z = z .+ Δ
            end
        end
        if step_taken * maximum(abs, Δ) < tol
            status = step_taken < 1e-3 ? :tiny_step_exit : :converged; break
        end
    end
    return z, status
end
function resid_generic(fam, y, n, Λ, β, link, z)
    η = GM._clamp_eta.(β .+ Λ * z)
    μ = GM._clamp_mu.(Ref(fam), GM.linkinv.(Ref(link), η))
    me = GM.mu_eta.(Ref(link), η)
    s = GM._glm_score.(Ref(fam), μ, n, me, y)
    return maxabs(Λ' * s .- z)
end

# Copy of families/twopart.jl::_twopart_mode for ZIP (Λz = 0), with status.
function im_zip(y, Λc, βz, βc; maxiter = 100, tol = 1e-9)
    p, K = size(Λc); z = zeros(K); status = :maxiter
    for _ in 1:maxiter
        ηz = GM._clamp_eta.(βz)
        ηc = GM._clamp_eta.(βc .+ Λc * z)
        sc = zeros(p); Wc = zeros(p)
        for t in 1:p
            _, s_c, _, W_c, _ = GM._tp_pieces(GM.ZIPoisson(), y[t], ηz[t], ηc[t])
            sc[t] = s_c; Wc[t] = W_c
        end
        A = Symmetric(Λc' * (Wc .* Λc) + I)
        Δ = GM._safe_solve(A, Λc' * sc .- z)
        if Δ === nothing || !all(isfinite, Δ)
            status = :solve_fail; break
        end
        z = z .+ Δ
        if maximum(abs, Δ) < tol
            status = :converged; break
        end
    end
    return z, status
end
function resid_zip(y, Λc, βz, βc, z)
    p = size(Λc, 1)
    ηz = GM._clamp_eta.(βz); ηc = GM._clamp_eta.(βc .+ Λc * z)
    sc = [GM._tp_pieces(GM.ZIPoisson(), y[t], ηz[t], ηc[t])[2] for t in 1:p]
    return maxabs(Λc' * sc .- z)
end

# Copy of the inline loop in grouped_dispersion.jl::_nb1_grouped_loglik_site, with status.
function im_nb1(fams, y, n, Λ, β, link; maxiter = 100, tol = 1e-9)
    p, K = size(Λ); z = zeros(K); status = :maxiter
    for _ in 1:maxiter
        η = GM._clamp_eta.(β .+ Λ * z)
        μ = GM._clamp_mu.(fams, GM.linkinv.(Ref(link), η))
        me = GM.mu_eta.(Ref(link), η)
        s = GM._glm_score.(fams, μ, n, me, y)
        W = GM._nb1_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(link))
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = GM._safe_solve(A, Λ' * s .- z)
        if Δ === nothing || !all(isfinite, Δ)
            status = :solve_fail; break
        end
        z = z .+ Δ
        if maximum(abs, Δ) < tol
            status = :converged; break
        end
    end
    return z, status
end
# post-mode value, same formula as _nb1_grouped_loglik_site (to bitwise-check the copy)
function nb1_site_value(fams, y, n, Λ, β, link, z; hessian = :observed)
    p = size(Λ, 1)
    η = GM._clamp_eta.(β .+ Λ * z)
    μ = GM._clamp_mu.(fams, GM.linkinv.(Ref(link), η))
    me = GM.mu_eta.(Ref(link), η)
    W = GM._nb1_grouped_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link))
    A = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = 0.0
    for t in 1:p
        ℓ += GM._glm_logpdf(fams[t], μ[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end
function resid_nb1(fams, y, n, Λ, β, link, z)
    η = GM._clamp_eta.(β .+ Λ * z)
    μ = GM._clamp_mu.(fams, GM.linkinv.(Ref(link), η))
    me = GM.mu_eta.(Ref(link), η)
    s = GM._glm_score.(fams, μ, n, me, y)
    return maxabs(Λ' * s .- z)
end

# Copy of ordinal.jl::_ordinal_laplace_mode_pertrait, with status.
function im_ord(y, Λ, β, τ, C, link; maxiter = 100, tol = 1e-9)
    p, K = size(Λ); z = zeros(K); status = :maxiter
    for _ in 1:maxiter
        η = GM._clamp_eta.(β .+ Λ * z)
        s = zeros(p); W = zeros(p)
        for t in 1:p
            st, wt = GM._ord_score_weight(Int(y[t]), η[t], GM._trait_cutpoints(τ, C, t), link)
            s[t] = st; W[t] = wt
        end
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = GM._safe_solve(A, Λ' * s .- z)
        if Δ === nothing || !all(isfinite, Δ)
            status = :solve_fail; break
        end
        z = z .+ Δ
        if maximum(abs, Δ) < tol
            status = :converged; break
        end
    end
    return z, status
end
function resid_ord(y, Λ, β, τ, C, link, z)
    p = size(Λ, 1)
    η = GM._clamp_eta.(β .+ Λ * z)
    s = [GM._ord_score_weight(Int(y[t]), η[t], GM._trait_cutpoints(τ, C, t), link)[1] for t in 1:p]
    return maxabs(Λ' * s .- z)
end

# Site census: (n_nonconv, statuses seen, max residual at package mode, max |z_copy - z_pkg|)
function census(sitefun, n)
    nonconv = 0; kinds = Dict{Symbol,Int}(); rmax = 0.0; dz = 0.0; idx = Int[]
    for i in 1:n
        st, r, d = sitefun(i)
        kinds[st] = get(kinds, st, 0) + 1
        if st != :converged || r > 1e-6
            nonconv += 1; push!(idx, i)
        end
        rmax = max(rmax, r); dz = max(dz, d)
    end
    return (nonconv = nonconv, kinds = kinds, rmax = rmax, dzcopy = dz, idx = idx)
end

# --------------------------------------------------------------------------------------
# Routes. Each returns a NamedTuple row.
# --------------------------------------------------------------------------------------

const ROWS = NamedTuple[]

function finish_row!(; route, setting, tfit, fit_conv, fit_ll, negll, θhat, θtruth, rep,
                     rep_match, valid, g_tol, ga, runner, classA0, classA1, extraA = "")
    g1 = fdgrad(negll, θhat; h = 1e-5)
    g2 = fdgrad(negll, θhat; h = 1e-4)
    gfd = maxabs(g1); gfd2 = maxabs(g2)
    gmax = ga === nothing ? gfd : ga
    flagB = fit_conv && gmax > 100 * g_tol
    # restart from theta_hat (fresh L-BFGS memory, same settings)
    t = time()
    rr = runner(θhat)
    ll_rs = -Optim.minimum(rr)
    # fresh start from the generating parameters
    rf = runner(θtruth)
    ll_fr = -Optim.minimum(rf)
    trs = time() - t
    dll = max(ll_rs, ll_fr) - fit_ll
    fl = stopflags(rep)
    row = (route = route, setting = setting, tfit = tfit, conv = fit_conv, ll = fit_ll,
           valid = valid, rep_match = rep_match,
           x = fl.x, f = fl.f, g = fl.g, ls_failed = fl.ls_failed, gres = fl.gres, it = fl.it,
           gfd = gfd, gfd2 = gfd2, gan = ga === nothing ? NaN : ga,
           flagB = flagB, ll_restart = ll_rs, ll_fresh = ll_fr, dll = dll,
           restart_conv = Optim.converged(rr), restart_gres = Optim.g_residual(rr),
           fresh_conv = Optim.converged(rf), fresh_gres = Optim.g_residual(rf),
           A0 = classA0, A1 = classA1, extraA = extraA, trs = trs)
    push!(ROWS, row)
    @printf("  %-9s %-14s t=%.1fs conv=%s ll=%.4f | valid|Δ|=%.1e repΔθ=%.1e | stop x=%d f=%d g=%d lsf=%d it=%d gres=%.2e | |g|fd=%.2e/%.2e an=%s | B=%s | restart %.4f (Δ%+.2e) fresh %.4f (Δ%+.2e) | A0 %d %s rmax=%.1e | A1 %d %s rmax=%.1e dz=%.1e %s\n",
            route, setting, tfit, fit_conv, fit_ll, valid, rep_match, fl.x, fl.f, fl.g,
            fl.ls_failed, fl.it, fl.gres, gfd, gfd2,
            ga === nothing ? "-" : @sprintf("%.2e", ga), flagB, ll_rs, ll_rs - fit_ll, ll_fr,
            ll_fr - fit_ll, classA0.nonconv, string(classA0.kinds), classA0.rmax,
            classA1.nonconv, string(classA1.kinds), classA1.rmax, classA1.dzcopy, extraA)
    flush(stdout)
    return row
end

# ---------------- Poisson ----------------
function route_poisson(rng, setting; p = 5, n = 80, K = 2, sd = 0.7, βlo = 0.0, βhi = 1.5)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd)
    Z = randn(rng, K, n)
    Y = [rand(rng, Poisson(exp(clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30)))) for t in 1:p, i in 1:n]
    link = GM.LogLink()
    t = time(); fit = fit_poisson_gllvm(Y; K = K); tfit = time() - t
    rr = GM.rr_theta_len(p, K); N1 = ones(Int, p, n)
    negll = θ -> begin
        v = try
            -GM.marginal_loglik_laplace(Poisson(), Y, N1, GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                                        θ[1:p], link; maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    ag = θ -> try
        -GM.poisson_laplace_grad(Y, GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K), θ[1:p])
    catch
        nothing
    end
    # warm start (copy of _fit_poisson_gllvm_laplace)
    Zemp = [GM.linkfun(link, max(Y[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    F = svd(Zemp .- β0); L0 = zeros(p, K)
    for j in 1:min(K, length(F.S))
        L0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, GM.pack_lambda(L0))
    runner = θs -> an_run(negll, ag, θs)
    rep = runner(θ0)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ))
    rep_match = maxabs(Optim.minimizer(rep) .- θhat)
    valid = abs(negll(θhat) + fit.loglik)
    gan = let g = ag(θhat); g === nothing ? NaN : maxabs(g) end
    classA(θ) = begin
        b = θ[1:p]; L = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        census(n) do i
            y = view(Y, :, i); nn = ones(Int, p)
            zc, st = im_generic(Poisson(), y, nn, L, b, link)
            zp = GM._laplace_mode(Poisson(), y, nn, L, b, link)
            (st, resid_generic(Poisson(), y, nn, L, b, link, zp), maxabs(zc .- zp))
        end
    end
    finish_row!(route = "poisson", setting = setting, tfit = tfit, fit_conv = fit.converged,
                fit_ll = fit.loglik, negll = negll, θhat = θhat,
                θtruth = vcat(β, GM.pack_lambda(Λ)), rep = rep, rep_match = rep_match,
                valid = valid, g_tol = 1e-5, ga = gan, runner = runner,
                classA0 = classA(θ0), classA1 = classA(θhat))
end

# ---------------- Binomial (Bernoulli, logit) ----------------
function route_binomial(rng, setting; p = 5, n = 80, K = 2, sd = 0.7, βlo = -1.0, βhi = 1.0, Ntr = 1)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd)
    Z = randn(rng, K, n)
    Nm = fill(Ntr, p, n)
    Y = [rand(rng, Binomial(Ntr, logistic(β[t] + dot(Λ[t, :], Z[:, i])))) for t in 1:p, i in 1:n]
    link = GM.LogitLink()
    t = time(); fit = fit_binomial_gllvm(Y; K = K, N = Nm); tfit = time() - t
    rr = GM.rr_theta_len(p, K)
    negll = θ -> begin
        v = try
            -GM.binomial_marginal_loglik_laplace(Y, Nm, GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                                                 θ[1:p], link; hessian = fit.hessian,
                                                 maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    ag = θ -> try
        -GM.binomial_laplace_grad(Y, Nm, GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K), θ[1:p])
    catch
        nothing
    end
    Zemp = [GM.linkfun(link, clamp((Y[t, i] + 0.5) / (Nm[t, i] + 1), 1e-4, 1 - 1e-4)) for t in 1:p, i in 1:n]
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    F = svd(Zemp .- β0); L0 = zeros(p, K)
    for j in 1:min(K, length(F.S))
        L0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, GM.pack_lambda(L0))
    runner = θs -> an_run(negll, ag, θs)
    rep = runner(θ0)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ))
    rep_match = maxabs(Optim.minimizer(rep) .- θhat)
    valid = abs(negll(θhat) + fit.loglik)
    gan = let g = ag(θhat); g === nothing ? NaN : maxabs(g) end
    classA(θ) = begin
        b = θ[1:p]; L = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        census(n) do i
            y = view(Y, :, i); nn = view(Nm, :, i)
            zc, st = im_generic(Binomial(), y, nn, L, b, link)
            zp = GM._laplace_mode(Binomial(), y, nn, L, b, link)
            (st, resid_generic(Binomial(), y, nn, L, b, link, zp), maxabs(zc .- zp))
        end
    end
    finish_row!(route = "binomial", setting = setting, tfit = tfit, fit_conv = fit.converged,
                fit_ll = fit.loglik, negll = negll, θhat = θhat,
                θtruth = vcat(β, GM.pack_lambda(Λ)), rep = rep, rep_match = rep_match,
                valid = valid, g_tol = 1e-5, ga = gan, runner = runner,
                classA0 = classA(θ0), classA1 = classA(θhat))
end

# ---------------- ZIP ----------------
function route_zip(rng, setting; p = 5, n = 80, K = 2, sd = 0.7, π0 = 0.25, βlo = 0.5, βhi = 1.5)
    βz = fill(log(π0 / (1 - π0)), p); βc = βlo .+ (βhi - βlo) .* rand(rng, p)
    Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [rand(rng) < π0 ? 0 : rand(rng, Poisson(exp(clamp(βc[t] + dot(Λ[t, :], Z[:, i]), -30, 30))))
         for t in 1:p, i in 1:n]
    t = time(); fit = fit_zip_gllvm(Y; K = K); tfit = time() - t
    rr = GM.rr_theta_len(p, K); Λz0 = zeros(p, K)
    negll = θ -> begin
        v = try
            -GM.zip_marginal_loglik_laplace(Y, GM.unpack_lambda(θ[(2p + 1):(2p + rr)], p, K),
                                            θ[1:p], θ[(p + 1):(2p)]; offsetc = nothing,
                                            hessian = :observed, maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    βz0, βc0, Λc0 = GM._zi_warmstart(Y, K)
    θ0 = vcat(βz0, βc0, GM.pack_lambda(Λc0))
    runner = θs -> fd_run(negll, θs)
    rep = runner(θ0)
    θhat = vcat(fit.βz, fit.βc, GM.pack_lambda(fit.Λc))
    rep_match = maxabs(Optim.minimizer(rep) .- θhat)
    valid = abs(negll(θhat) + fit.loglik)
    classA(θ) = begin
        bz = θ[1:p]; bc = θ[(p + 1):(2p)]; L = GM.unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        census(n) do i
            y = view(Y, :, i)
            zc, st = im_zip(y, L, bz, bc)
            zp = GM._twopart_mode(GM.ZIPoisson(), y, Λz0, L, bz, bc)
            (st, resid_zip(y, L, bz, bc, zp), maxabs(zc .- zp))
        end
    end
    finish_row!(route = "zip", setting = setting, tfit = tfit, fit_conv = fit.converged,
                fit_ll = fit.loglik, negll = negll, θhat = θhat,
                θtruth = vcat(βz, βc, GM.pack_lambda(Λ)), rep = rep, rep_match = rep_match,
                valid = valid, g_tol = 1e-5, ga = nothing, runner = runner,
                classA0 = classA(θ0), classA1 = classA(θhat))
end

# ---------------- NB1 (per-trait dispersion, grouped route) ----------------
function route_nb1(rng, setting; p = 5, n = 80, K = 2, sd = 0.7, φ = 1.0, βlo = 0.5, βhi = 1.5)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [begin
            μ = exp(clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30))
            rand(rng, NegativeBinomial(μ / φ, 1 / (1 + φ)))
         end for t in 1:p, i in 1:n]
    link = GM.LogLink()
    t = time(); fit = fit_nb1_gllvm_grouped(Y; K = K, group = collect(1:p)); tfit = time() - t
    rr = GM.rr_theta_len(p, K)
    negll = θ -> begin
        v = try
            -GM.nb1_grouped_marginal_loglik_laplace(Y, GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                                                    θ[1:p], exp.(θ[(p + rr + 1):(p + rr + p)]);
                                                    link = link, hessian = :observed,
                                                    maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    Zemp = [GM.linkfun(link, max(Y[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    F = svd(Zemp .- β0); L0 = zeros(p, K)
    for j in 1:min(K, length(F.S))
        L0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, GM.pack_lambda(L0), zeros(p))
    runner = θs -> fd_run(negll, θs)
    rep = runner(θ0)
    θrep = Optim.minimizer(rep)
    θhat_fit = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.φ))
    rep_match = maxabs(θrep .- θhat_fit)
    θhat = θrep   # exact minimizer (φ round-trip through exp/log avoided)
    valid = abs(negll(θhat_fit) + fit.loglik)
    classA(θ) = begin
        b = θ[1:p]; L = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        fams = [GM.NB1(exp(θ[p + rr + t])) for t in 1:p]
        census(n) do i
            y = view(Y, :, i); nn = ones(Int, p)
            zc, st = im_nb1(fams, y, nn, L, b, link)
            vpkg = GM._nb1_grouped_loglik_site(fams, y, nn, L, b, link; hessian = :observed)
            vcopy = nb1_site_value(fams, y, nn, L, b, link, zc)
            (st, resid_nb1(fams, y, nn, L, b, link, zc), abs(vpkg - vcopy))
        end
    end
    finish_row!(route = "nb1", setting = setting, tfit = tfit, fit_conv = fit.converged,
                fit_ll = fit.loglik, negll = negll, θhat = θhat,
                θtruth = vcat(β, GM.pack_lambda(Λ), fill(log(φ), p)), rep = rep,
                rep_match = rep_match, valid = valid, g_tol = 1e-5, ga = nothing,
                runner = runner, classA0 = classA(θ0), classA1 = classA(θhat),
                extraA = "boundary=" * string(any(fit.dispersion_boundary)))
end

# ---------------- Ordinal (per-trait cutpoints, logit) ----------------
function route_ordinal(rng, setting; p = 5, n = 80, K = 2, sd = 0.7, Ccat = 4, gap = 1.0)
    Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    β = 0.3 .* randn(rng, p)
    τtrue = [0.0 + gap * (c - 1) for c in 1:(Ccat - 1)]
    Y = [begin
            η = β[t] + dot(Λ[t, :], Z[:, i]); u = rand(rng); c = Ccat
            for k in 1:(Ccat - 1)
                if u <= logistic(τtrue[k] - η)
                    c = k; break
                end
            end
            c
         end for t in 1:p, i in 1:n]
    link = GM.LogitLink()
    t = time(); fit = fit_ordinal_gllvm_pertrait(Y; K = K, link = link); tfit = time() - t
    rr = GM.rr_theta_len(p, K)
    C = fit.C; ncut = sum(C .- 2)
    negll = θ -> begin
        v = try
            -GM.ordinal_marginal_loglik_laplace_pertrait(Y, GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                θ[1:p], GM._unpack_cutpoints_pertrait(θ[(p + rr + 1):(p + rr + ncut)], C), C;
                link = link, mask = nothing, maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    obs = trues(p, n)
    Zproxy = [quantile(Normal(), clamp((Y[t, i] - 0.5) / C[t], 1e-3, 1 - 1e-3)) for t in 1:p, i in 1:n]
    F = svd(Zproxy .- (sum(Zproxy; dims = 2) ./ n)); L0 = zeros(p, K)
    for j in 1:min(K, length(F.S))
        L0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    β0, ψ0 = GM._pack_initial_ordinal_pertrait(Y, obs, C, link)
    θ0 = vcat(β0, GM.pack_lambda(L0), ψ0)
    runner = θs -> fd_run(negll, θs)
    rep = runner(θ0)
    θrep = Optim.minimizer(rep)
    τrep = GM._unpack_cutpoints_pertrait(θrep[(p + rr + 1):(p + rr + ncut)], C)
    rep_match = max(maxabs(θrep[1:p] .- fit.β), maxabs(θrep[(p + 1):(p + rr)] .- GM.pack_lambda(fit.Λ)),
                    maxabs(filter(isfinite, τrep) .- filter(isfinite, fit.τ)))
    θhat = θrep
    valid = abs(negll(θhat) + fit.loglik)
    ψtruth = Float64[]
    for tt in 1:p, c in 2:(C[tt] - 1)
        push!(ψtruth, log(gap))
    end
    classA(θ) = begin
        b = θ[1:p]; L = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        τm = GM._unpack_cutpoints_pertrait(θ[(p + rr + 1):(p + rr + ncut)], C)
        census(n) do i
            y = view(Y, :, i)
            zc, st = im_ord(y, L, b, τm, C, link)
            zp = GM._ordinal_laplace_mode_pertrait(y, L, b, τm, C, link)
            (st, resid_ord(y, L, b, τm, C, link, zp), maxabs(zc .- zp))
        end
    end
    finish_row!(route = "ordinal", setting = setting, tfit = tfit, fit_conv = fit.converged,
                fit_ll = fit.loglik, negll = negll, θhat = θhat,
                θtruth = vcat(β, GM.pack_lambda(Λ), ψtruth), rep = rep, rep_match = rep_match,
                valid = valid, g_tol = 1e-5, ga = nothing, runner = runner,
                classA0 = classA(θ0), classA1 = classA(θhat), extraA = "C=" * string(C))
end

# ---------------- Truncated Poisson ----------------
function route_truncpois(rng, setting; p = 5, n = 80, K = 2, sd = 0.7, βlo = 0.3, βhi = 1.2)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [begin
            μ = exp(clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30)); y = 0
            while y == 0
                y = rand(rng, Poisson(μ))
            end
            y
         end for t in 1:p, i in 1:n]
    link = GM.LogLink()
    t = time(); fit = fit_truncated_poisson_gllvm(Y; K = K); tfit = time() - t
    rr = GM.rr_theta_len(p, K); N1 = ones(Int, p, n); fam = GM.TruncatedPoisson()
    negll = θ -> begin
        v = try
            -GM.marginal_loglik_laplace(fam, Y, N1, GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
                                        θ[1:p], link; mask = nothing, offset = nothing,
                                        maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    Zemp = [GM.linkfun(link, max(Float64(Y[t, i]), 1.0)) for t in 1:p, i in 1:n]
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    F = svd(Zemp .- β0); L0 = zeros(p, K)
    for j in 1:min(K, length(F.S))
        L0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, GM.pack_lambda(L0))
    runner = θs -> fd_run(negll, θs)
    rep = runner(θ0)
    θhat = fit.theta_packed
    rep_match = maxabs(Optim.minimizer(rep) .- θhat)
    valid = abs(negll(θhat) + fit.loglik)
    classA(θ) = begin
        b = θ[1:p]; L = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        census(n) do i
            y = view(Y, :, i); nn = ones(Int, p)
            zc, st = im_generic(fam, y, nn, L, b, link)
            zp = GM._laplace_mode(fam, y, nn, L, b, link)
            (st, resid_generic(fam, y, nn, L, b, link, zp), maxabs(zc .- zp))
        end
    end
    finish_row!(route = "truncpois", setting = setting, tfit = tfit, fit_conv = fit.converged,
                fit_ll = fit.loglik, negll = negll, θhat = θhat,
                θtruth = vcat(β, GM.pack_lambda(Λ)), rep = rep, rep_match = rep_match,
                valid = valid, g_tol = 1e-5, ga = nothing, runner = runner,
                classA0 = classA(θ0), classA1 = classA(θhat))
end

# ---------------- Gaussian (bridge default: centred Y, profiled objective) ----------------
function route_gaussian(rng, setting; p = 5, n = 80, K = 2, sd = 0.7, σ = 0.5)
    Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [dot(Λ[t, :], Z[:, i]) + σ * randn(rng) for t in 1:p, i in 1:n]
    Yc = Y .- mean(Y; dims = 2)
    t = time(); fit = fit_gaussian_gllvm(Yc; K = K); tfit = time() - t
    res = fit.optim_result
    spec = (q = 0, p = p, K_B = K, K_W = 0, has_diag = false, K_phy = 0, has_phy_unique = false)
    negll = θ -> GM.gaussian_profile_nll(θ, Yc; spec = spec, X = nothing, Σ_phy = nothing,
                                         profile_beta = false)
    opts = Optim.Options(x_abstol = 1e-8, f_reltol = 1e-10, g_tol = 1e-6, iterations = 500,
                         show_trace = false)
    runner = θs -> Optim.optimize(negll, θs, Optim.LBFGS(), opts; autodiff = :forward)
    θhat = Optim.minimizer(res)
    valid = abs(negll(θhat) + fit.logLik)
    rep = res               # the fit's own Optim result (stored on GllvmFit)
    rep_match = 0.0
    ga = maxabs(FD.gradient(negll, θhat))
    empty = (nonconv = 0, kinds = Dict(:closed_form => n), rmax = 0.0, dzcopy = 0.0, idx = Int[])
    finish_row!(route = "gaussian", setting = setting, tfit = tfit, fit_conv = fit.converged,
                fit_ll = fit.logLik, negll = negll, θhat = θhat,
                θtruth = GM.pack_lambda(Λ ./ σ), rep = rep, rep_match = rep_match,
                valid = valid, g_tol = 1e-6, ga = ga, runner = runner,
                classA0 = empty, classA1 = empty)
end

# --------------------------------------------------------------------------------------
# Settings: 8 datasets per route (seeded), base + harder settings near family limits.
# --------------------------------------------------------------------------------------
const SETTINGS = [
    ("base-s1", 101, (;)),
    ("base-s2", 102, (;)),
    ("base-s3", 103, (;)),
    ("strongL", 104, (sd = 1.5,)),
    ("small-n30", 105, (n = 30,)),
    ("extreme", 106, :extreme),
    ("p8", 107, (p = 8,)),
    ("extreme+L", 108, :extremeL),
]
const EXTREME = Dict(
    "poisson"   => (βlo = -1.8, βhi = -1.0),
    "binomial"  => (βlo = -2.5, βhi = 2.5, sd = 1.2),
    "zip"       => (π0 = 0.6, βlo = -0.2, βhi = 0.3),
    "nb1"       => (φ = 5.0, βlo = -0.2, βhi = 0.5),
    "ordinal"   => (Ccat = 5, gap = 0.6),
    "truncpois" => (βlo = -1.0, βhi = -0.3),
    "gaussian"  => (σ = 0.08,),
)
const ROUTES = [("zip", route_zip), ("poisson", route_poisson), ("binomial", route_binomial),
                ("nb1", route_nb1), ("ordinal", route_ordinal), ("truncpois", route_truncpois),
                ("gaussian", route_gaussian)]

only = get(ENV, "AUDIT_ONLY", "")
println("Optim ", pkgversion(Optim), "  GLLVModels at ", pathof(GLLVModels))
println("soft budget ", SOFT, " s")
stopped_early = false
for (sname, seed, kw) in SETTINGS
    for (rname, rf) in ROUTES
        (isempty(only) || occursin(rname, only)) || continue
        if elapsed() > SOFT
            global stopped_early = true
            @printf("BUDGET: soft limit reached at %.0f s; skipping %s/%s and the rest\n", elapsed(), rname, sname)
            break
        end
        kws = kw === :extreme ? EXTREME[rname] :
              kw === :extremeL ? merge(EXTREME[rname], (sd = 1.5,)) : kw
        rng = MersenneTwister(seed)
        try
            rf(rng, sname; kws...)
        catch e
            @printf("  %-9s %-14s ERROR %s\n", rname, sname, sprint(showerror, e)[1:min(end, 300)])
        end
    end
    stopped_early && break
end

println("\n==== SUMMARY by route ====")
for r in unique(getfield.(ROWS, :route))
    rows = filter(x -> x.route == r, ROWS)
    nconv = count(x -> x.conv, rows)
    nB = count(x -> x.flagB, rows)
    nBx = count(x -> x.flagB && !x.g, rows)
    nstopxf = count(x -> x.conv && !x.g, rows)
    nimp = count(x -> x.dll > 1e-3, rows)
    nBimp = count(x -> x.flagB && x.dll > 1e-3, rows)
    nA1 = count(x -> x.A1.nonconv > 0, rows)
    nA0 = count(x -> x.A0.nonconv > 0, rows)
    @printf("%-9s fits=%d conv=%d | conv&&stop-not-g=%d | flagB(conv,|g|>100g_tol)=%d | restart/fresh +>1e-3: %d (of flagged %d) | max|g| on conv fits=%.2e | max valid |Δ|=%.1e max repΔθ=%.1e | ClassA nonconv sites: start %d fits (%d sites), hat %d fits (%d sites)\n",
            r, length(rows), nconv, nstopxf, nB, nimp, nBimp,
            maximum([max(x.gfd, isnan(x.gan) ? 0.0 : x.gan) for x in rows if x.conv]; init = 0.0),
            maximum(x.valid for x in rows), maximum(x.rep_match for x in rows),
            nA0, sum(x.A0.nonconv for x in rows), nA1, sum(x.A1.nonconv for x in rows))
end
@printf("total elapsed %.1f s%s\n", elapsed(), stopped_early ? " (stopped early on soft budget)" : "")
