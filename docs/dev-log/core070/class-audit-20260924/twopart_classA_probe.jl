# Audit harness: CLASS A (silent inner-loop failure) and CLASS B (dishonest converged flag)
# for the zero-inflated two-part fitters in src/families/twopart.jl:
#   fit_zip_gllvm, fit_zip_gllvm_cov, fit_zinb_gllvm, fit_zinb_gllvm_cov,
#   fit_zib_gllvm, fit_zib_gllvm_cov.
# READ-ONLY with respect to the package: every package function is called, never redefined.
# Run:  cd <worktree> && JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#       julia +1.10.12 --startup-file=no --project=test/parity <this file>
using GLLVModels, Random, Distributions, LinearAlgebra, Printf
const G = GLLVModels
const Optim = G.Optim
const T0 = time()
elapsed() = time() - T0
# Wall-clock guards (the task budget is 10 min for the whole harness).
const T_FIT_MAX   = 390.0   # do not START a public fit after this
const T_EXTRA_MAX = 500.0   # do not START a restart after this
const T_HARD      = 570.0   # stop everything after this
const SMOKE = get(ENV, "ZI_SMOKE", "0") == "1"
const WARM_IT  = SMOKE ? 2 : 25         # warm restart from θ̂ (same LBFGS, capped)
const FRESH_IT = SMOKE ? 2 : 30         # fresh start (loadings × 0.3), capped
const GTOL = 1e-5           # fitter default g_tol
const MAXIT_NEWTON = 100; const TOL_NEWTON = 1e-9   # fitter defaults
logistic(x) = 1 / (1 + exp(-x))

# --------------------------------------------------------------------------------------
# Model descriptions. `unpack(θ)` mirrors the fitter's negll unpacking EXACTLY and returns
# the kernel inputs; `negll(θ)` mirrors the fitter's closure (same wrapper, same kwargs,
# same try/catch sentinel).
# --------------------------------------------------------------------------------------
struct Model
    name::String
    Y::Matrix{Int}
    p::Int; n::Int; K::Int; q::Int; rr::Int
    unpack::Function          # θ -> (fam, Λz, Λc, βz, βc, Oz, Oc)
    negll::Function           # replicated fitter objective
    θ0::Vector{Float64}       # replicated fitter start
    θtrue::Vector{Float64}
    fitfn::Function           # () -> public fit
    θhat::Function            # fit -> θ̂ (packed)
    fresh::Function           # θ0 -> fresh start
end

# Rotate true loadings to the lower-trapezoidal packing (z ~ N(0,I) is rotation invariant).
function lower_rot(Λ)
    F = qr(Matrix(Λ'))                 # Λ' = Q R  ⇒  Λ Q = R'
    L = Matrix(Λ * Matrix(F.Q))
    for k in 1:size(L, 2)              # positive diagonal (sign flip of a column is also invariant)
        if L[k, k] < 0; L[:, k] .*= -1; end
    end
    for k in 1:size(L, 2), i in 1:(k - 1); L[i, k] = 0.0; end
    return L
end

function sim(kind; seed, p = 5, n = 80, K = 2, sd = 0.8, r = 3.0, N = 10, βc_mean = 1.0, cov = false)
    rng = MersenneTwister(seed)
    Λ = sd .* randn(rng, p, K)
    βz = randn(rng, p) .* 0.5 .- 1.2                   # structural-zero share ≈ 0.23
    βc = kind === :zib ? randn(rng, p) .* 0.5 : randn(rng, p) .* 0.5 .+ βc_mean
    Z = randn(rng, K, n)
    x = randn(rng, n)
    γz = cov ? [0.4] : Float64[]; γc = cov ? [0.5] : Float64[]
    ηz = [βz[t] + (cov ? γz[1] * x[s] : 0.0) for t in 1:p, s in 1:n]
    ηc = [βc[t] + (cov ? γc[1] * x[s] : 0.0) + dot(Λ[t, :], Z[:, s]) for t in 1:p, s in 1:n]
    Y = zeros(Int, p, n)
    for t in 1:p, s in 1:n
        rand(rng) < logistic(ηz[t, s]) && continue
        if kind === :zip
            Y[t, s] = rand(rng, Poisson(exp(ηc[t, s])))
        elseif kind === :zinb
            μ = exp(ηc[t, s]); Y[t, s] = rand(rng, NegativeBinomial(r, r / (r + μ)))
        else
            Y[t, s] = rand(rng, Binomial(N, logistic(ηc[t, s])))
        end
    end
    X = reshape(repeat(x', p), p, n, 1)                  # shared site covariate, q = 1
    return (; Y, X, βz, βc, γz, γc, L = lower_rot(Λ), r, N)
end

function build(kind; cov, seed, sd, r = 3.0, N = 10, βc_mean = 1.0, tag = "")
    d = sim(kind; seed, sd, r, N, βc_mean, cov)
    Y = d.Y; p, n = size(Y); K = 2; rr = G.rr_theta_len(p, K)
    q = cov ? 1 : 0
    Λz0 = zeros(p, K)
    name = string(kind === :zip ? "fit_zip_gllvm" : kind === :zinb ? "fit_zinb_gllvm" : "fit_zib_gllvm",
                  cov ? "_cov" : "", tag)
    if !cov
        # ---- no-X fitters: θ = [βz; βc; pack(Λc)] (+ log r for ZINB) ----
        unpack = θ -> begin
            βz = θ[1:p]; βc = θ[(p + 1):(2p)]
            Λc = G.unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
            fam = kind === :zip ? G.ZIPoisson() : kind === :zinb ? G.ZINB(float(exp(θ[2p + rr + 1]))) : G.ZIB(Int(N))
            (fam, zeros(p, K), Λc, βz, βc, nothing, nothing)
        end
        negll = if kind === :zip
            θ -> begin
                βz = θ[1:p]; βc = θ[(p + 1):(2p)]
                Λc = G.unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
                v = try
                    -G.zip_marginal_loglik_laplace(Y, Λc, βz, βc; offsetc = nothing, hessian = :observed,
                                                   maxiter = MAXIT_NEWTON, tol = TOL_NEWTON)
                catch
                    return 1e12
                end
                isfinite(v) ? v : 1e12
            end
        elseif kind === :zinb
            θ -> begin
                βz = θ[1:p]; βc = θ[(p + 1):(2p)]
                Λc = G.unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
                rθ = exp(θ[2p + rr + 1])
                v = try
                    -G.zinb_marginal_loglik_laplace(Y, Λc, βz, βc, rθ; offsetc = nothing, hessian = :observed,
                                                    maxiter = MAXIT_NEWTON, tol = TOL_NEWTON)
                catch
                    return 1e12
                end
                isfinite(v) ? v : 1e12
            end
        else
            θ -> begin
                βz = θ[1:p]; βc = θ[(p + 1):(2p)]
                Λc = G.unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
                v = try
                    -G.zib_marginal_loglik_laplace(Y, Λc, βz, βc, N; offsetc = nothing, hessian = :observed,
                                                   maxiter = MAXIT_NEWTON, tol = TOL_NEWTON)
                catch
                    return 1e12
                end
                isfinite(v) ? v : 1e12
            end
        end
        βz0, βc0, Λc0 = kind === :zib ? G._zib_warmstart(Y, N, K) : G._zi_warmstart(Y, K)
        θ0 = kind === :zinb ? vcat(βz0, βc0, G.pack_lambda(Λc0), log(10.0)) : vcat(βz0, βc0, G.pack_lambda(Λc0))
        θtrue = kind === :zinb ? vcat(d.βz, d.βc, G.pack_lambda(d.L), log(d.r)) : vcat(d.βz, d.βc, G.pack_lambda(d.L))
        fitfn = kind === :zip ? ((; kw...) -> fit_zip_gllvm(Y; K = K, kw...)) :
                kind === :zinb ? ((; kw...) -> fit_zinb_gllvm(Y; K = K, kw...)) : ((; kw...) -> fit_zib_gllvm(Y; K = K, N = N, kw...))
        θhat = f -> kind === :zinb ? vcat(f.βz, f.βc, G.pack_lambda(f.Λc), log(f.r)) : vcat(f.βz, f.βc, G.pack_lambda(f.Λc))
        fresh = θ -> (θf = copy(θ); θf[(2p + 1):(2p + rr)] .*= 0.3; θf)
        return Model(name, Y, p, n, K, q, rr, unpack, negll, θ0, θtrue, fitfn, θhat, fresh)
    else
        # ---- cov fitters: θ = [βz; γz; βc; γc; pack(Λc)] (+ log r for ZINB); γ_fixed = nothing ----
        X = d.X
        mask = G._fixed_zero_mask(nothing, size(X, 3), "γ_fixed")
        X_fit, _ = G._slice_fixed_X(X, mask)
        unpack = θ -> begin
            βz = θ[1:p]; γz = θ[(p + 1):(p + q)]; βc = θ[(p + q + 1):(2p + q)]; γc = θ[(2p + q + 1):(2p + 2q)]
            Λc = G.unpack_lambda(θ[(2p + 2q + 1):(2p + 2q + rr)], p, K)
            fam = kind === :zip ? G.ZIPoisson() : kind === :zinb ? G.ZINB(float(exp(θ[2p + 2q + rr + 1]))) : G.ZIB(Int(N))
            (fam, zeros(p, K), Λc, βz, βc, G._build_offset(X_fit, γz), G._build_offset(X_fit, γc))
        end
        negll = θ -> begin
            βz = θ[1:p]; γz = θ[(p + 1):(p + q)]; βc = θ[(p + q + 1):(2p + q)]; γc = θ[(2p + q + 1):(2p + 2q)]
            Λc = G.unpack_lambda(θ[(2p + 2q + 1):(2p + 2q + rr)], p, K)
            Oz = G._build_offset(X_fit, γz); Oc = G._build_offset(X_fit, γc)
            v = try
                if kind === :zip
                    -G.zip_marginal_loglik_laplace(Y, Λc, βz, βc; offsetz = Oz, offsetc = Oc, hessian = :observed,
                                                   maxiter = MAXIT_NEWTON, tol = TOL_NEWTON)
                elseif kind === :zinb
                    rθ = exp(θ[2p + 2q + rr + 1])
                    -G.zinb_marginal_loglik_laplace(Y, Λc, βz, βc, rθ; hessian = :observed, offsetz = Oz, offsetc = Oc,
                                                    maxiter = MAXIT_NEWTON, tol = TOL_NEWTON)
                else
                    -G.zib_marginal_loglik_laplace(Y, Λc, βz, βc, N; hessian = :observed, offsetz = Oz, offsetc = Oc,
                                                   maxiter = MAXIT_NEWTON, tol = TOL_NEWTON)
                end
            catch
                return 1e12
            end
            isfinite(v) ? v : 1e12
        end
        βz0, βc0, Λc0 = kind === :zib ? G._zib_warmstart(Y, N, K) : G._zi_warmstart(Y, K)
        base = vcat(βz0, zeros(q), βc0, zeros(q), G.pack_lambda(Λc0))
        θ0 = kind === :zinb ? vcat(base, log(10.0)) : base
        tb = vcat(d.βz, d.γz, d.βc, d.γc, G.pack_lambda(d.L))
        θtrue = kind === :zinb ? vcat(tb, log(d.r)) : tb
        fitfn = kind === :zip ? ((; kw...) -> fit_zip_gllvm_cov(Y; X = X, K = K, kw...)) :
                kind === :zinb ? ((; kw...) -> fit_zinb_gllvm_cov(Y; X = X, K = K, kw...)) : ((; kw...) -> fit_zib_gllvm_cov(Y; X = X, K = K, N = N, kw...))
        θhat = f -> begin
            b = vcat(f.βz, f.γz[.!f.γ_fixed], f.βc, f.γc[.!f.γ_fixed], G.pack_lambda(f.Λc))
            kind === :zinb ? vcat(b, log(f.r)) : b
        end
        fresh = θ -> (θf = copy(θ); θf[(2p + 2q + 1):(2p + 2q + rr)] .*= 0.3; θf)
        return Model(name, Y, p, n, K, q, rr, unpack, negll, θ0, θtrue, fitfn, θhat, fresh)
    end
end

# --------------------------------------------------------------------------------------
# CLASS A instrumentation.
# mode_instr: verbatim copy of _twopart_mode's arithmetic plus an exit status.
# robust_mode: damped (backtracking) Fisher scoring on q(z) = Σ logf − ½z'z, the reference.
# --------------------------------------------------------------------------------------
function pieces(fam, y, Λz, Λc, βz, βc, offz, offc, z)
    p = length(y)
    ηz = G._clamp_eta.(βz .+ offz .+ Λz * z)
    ηc = G._clamp_eta.(βc .+ offc .+ Λc * z)
    sz = zeros(p); sc = zeros(p); Wz = zeros(p); Wc = zeros(p); ℓ = 0.0
    for t in 1:p
        a, b, c, e, lf = G._tp_pieces_at(fam, t, y[t], ηz[t], ηc[t])
        sz[t] = a; sc[t] = b; Wz[t] = c; Wc[t] = e; ℓ += lf
    end
    return sz, sc, Wz, Wc, ℓ, ηz, ηc
end

function mode_instr(fam, y, Λz, Λc, βz, βc, offz, offc)
    K = size(Λc, 2); z = zeros(K); status = :maxiter; it = 0
    for i in 1:MAXIT_NEWTON
        it = i
        sz, sc, Wz, Wc, _, _, _ = pieces(fam, y, Λz, Λc, βz, βc, offz, offc, z)
        Amat = Λz' * (Wz .* Λz); Amat .+= Λc' * (Wc .* Λc)
        for d in 1:K; Amat[d, d] += 1.0; end
        g = Λz' * sz; g .= g .+ Λc' * sc .- z
        Δ = G._safe_solve(Symmetric(Amat), g)
        if Δ === nothing || !all(isfinite, Δ); status = :nonfinite_step; break; end
        z = z .+ Δ
        if maximum(abs, Δ) < TOL_NEWTON; status = :converged; break; end
    end
    sz, sc, _, _, _, _, _ = pieces(fam, y, Λz, Λc, βz, βc, offz, offc, z)
    gz = Λz' * sz .+ Λc' * sc .- z
    return z, status, it, maximum(abs, gz)
end

# Log posterior q(z) = Σ logf − ½z'z; generic in eltype so ForwardDiff can drive the BFGS fallback.
function qpost(fam, y, Λz, Λc, βz, βc, offz, offc, z)
    ηz = G._clamp_eta.(βz .+ offz .+ Λz * z)
    ηc = G._clamp_eta.(βc .+ offc .+ Λc * z)
    ℓ = zero(eltype(ηc))
    for t in eachindex(y); ℓ += G._tp_pieces_at(fam, t, y[t], ηz[t], ηc[t])[5]; end
    return ℓ - 0.5 * dot(z, z)
end

function laplace_at(fam, y, Λz, Λc, βz, βc, offz, offc, z)
    p = length(y); K = size(Λc, 2)
    _, _, Wz, Wc, ℓ, _, ηc = pieces(fam, y, Λz, Λc, βz, βc, offz, offc, z)
    Wo = [G._tp_observed_Wc_at(fam, t, y[t], ηc[t], Wc[t]) for t in 1:p]
    A = Λz' * (Wz .* Λz) + Λc' * (Wo .* Λc) + I
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(Symmetric(Matrix(A)))
end

function robust_mode(fam, y, Λz, Λc, βz, βc, offz, offc)
    z, ok, gmax = damped_mode(fam, y, Λz, Λc, βz, βc, offz, offc)
    ok && return z, ok, gmax
    # Fallback: BFGS (ForwardDiff) on −q from the damped point; keep it only if it is no worse.
    f = zz -> -qpost(fam, y, Λz, Λc, βz, βc, offz, offc, zz)
    r = try Optim.optimize(f, z, Optim.BFGS(), Optim.Options(g_tol = 1e-10, iterations = 1000); autodiff = :forward) catch; nothing end
    if r !== nothing && all(isfinite, Optim.minimizer(r)) && -Optim.minimum(r) >= qpost(fam, y, Λz, Λc, βz, βc, offz, offc, z)
        z = Optim.minimizer(r)
    end
    sz, sc, _, _, _, _, _ = pieces(fam, y, Λz, Λc, βz, βc, offz, offc, z)
    gmax = maximum(abs, Λz' * sz .+ Λc' * sc .- z)
    return z, gmax < 1e-6, gmax
end

function damped_mode(fam, y, Λz, Λc, βz, βc, offz, offc; maxiter = 400)
    K = size(Λc, 2); z = zeros(K); q = qpost(fam, y, Λz, Λc, βz, βc, offz, offc, z); gmax = Inf
    for _ in 1:maxiter
        sz, sc, Wz, Wc, _, _, _ = pieces(fam, y, Λz, Λc, βz, βc, offz, offc, z)
        g = Λz' * sz .+ Λc' * sc .- z; gmax = maximum(abs, g)
        gmax < 1e-9 && return z, true, gmax
        A = Symmetric(Λz' * (Wz .* Λz) + Λc' * (Wc .* Λc) + I)
        Δ = A \ g; α = 1.0; zn = z; qn = -Inf
        while α > 1e-14
            zn = z .+ α .* Δ; qn = qpost(fam, y, Λz, Λc, βz, βc, offz, offc, zn)
            (isfinite(qn) && qn >= q) && break
            α /= 2
        end
        α <= 1e-14 && return z, gmax < 1e-6, gmax
        maximum(abs, zn .- z) < 1e-13 && (z = zn; break)
        z = zn; q = qn
    end
    sz, sc, _, _, _, _, _ = pieces(fam, y, Λz, Λc, βz, βc, offz, offc, z)
    gmax = maximum(abs, Λz' * sz .+ Λc' * sc .- z)
    return z, gmax < 1e-6, gmax
end

sitecol(O, s) = O === nothing ? false : view(O, :, s)
sitekw(O, s) = O === nothing ? nothing : view(O, :, s)

# Per-site census at θ: package value, instrumented status, reference (robust) value.
function census(m::Model, θ)
    fam, Λz, Λc, βz, βc, Oz, Oc = m.unpack(θ)
    nonconv = 0; nmaxit = 0; nnonfin = 0; nfinite_nonconv = 0; zmismatch = 0; robfail = 0; nmat = 0; nhigh = 0
    ll_pkg = 0.0; ll_rob = 0.0; worst_site_gap = 0.0; worst_site = (0, 0.0, 0.0, 0.0)
    for s in 1:m.n
        y = view(m.Y, :, s)
        offz = sitecol(Oz, s); offc = sitecol(Oc, s)
        v = G.twopart_loglik_site(fam, y, Λz, Λc, βz, βc; offsetz = sitekw(Oz, s), offsetc = sitekw(Oc, s),
                                  hessian = :observed, maxiter = MAXIT_NEWTON, tol = TOL_NEWTON)
        zp = G._twopart_mode(fam, y, Λz, Λc, βz, βc; offsetz = sitekw(Oz, s), offsetc = sitekw(Oc, s),
                             maxiter = MAXIT_NEWTON, tol = TOL_NEWTON)
        zi, st, it, gz = mode_instr(fam, y, Λz, Λc, βz, βc, offz, offc)
        zi == zp || (zmismatch += 1)
        zr, okr, _ = robust_mode(fam, y, Λz, Λc, βz, βc, offz, offc)
        okr || (robfail += 1)
        vr = laplace_at(fam, y, Λz, Λc, βz, βc, offz, offc, zr)
        ll_pkg += v; ll_rob += vr
        if st !== :converged
            nonconv += 1
            st === :maxiter ? (nmaxit += 1) : (nnonfin += 1)
            isfinite(v) && (nfinite_nonconv += 1)
            gap = abs(vr - v)
            if isfinite(v) && gap > 1e-3
                nmat += 1
                v > vr && (nhigh += 1)       # package value spuriously HIGHER than at the true mode
            end
            if gap > worst_site_gap
                worst_site_gap = gap; worst_site = (s, v, vr, gz)
            end
        end
    end
    return (; nonconv, nmaxit, nnonfin, nfinite_nonconv, nmat, nhigh, zmismatch, robfail, ll_pkg, ll_rob,
            gap = ll_rob - ll_pkg, worst_site)
end

# Robust (reference) objective, same sentinel convention.
function negll_robust(m::Model, θ)
    fam, Λz, Λc, βz, βc, Oz, Oc = m.unpack(θ)
    acc = 0.0
    for s in 1:m.n
        y = view(m.Y, :, s); offz = sitecol(Oz, s); offc = sitecol(Oc, s)
        zr, _, _ = robust_mode(fam, y, Λz, Λc, βz, βc, offz, offc)
        acc += laplace_at(fam, y, Λz, Λc, βz, βc, offz, offc, zr)
    end
    v = -acc
    return isfinite(v) ? v : 1e12
end

# FiniteDiff-style central gradient (relstep = absstep = cbrt(eps)), as Optim's autodiff=:finite.
function fdgrad(f, θ; rel = cbrt(eps()))
    g = similar(θ); nsent = 0
    for i in eachindex(θ)
        h = max(rel * abs(θ[i]), rel)
        θp = copy(θ); θm = copy(θ); θp[i] += h; θm[i] -= h
        fp = f(θp); fm = f(θm)
        (fp >= 1e11 || fm >= 1e11) && (nsent += 1)
        g[i] = (fp - fm) / (2h)
    end
    return g, nsent
end

function lbfgs(f, θ, iters)
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    return Optim.optimize(f, θ, ls, Optim.Options(g_tol = GTOL, iterations = iters); autodiff = :finite)
end
flags(res) = @sprintf("it=%d conv=%s g_conv=%s f_conv=%s x_conv=%s g_residual=%.3g",
                      Optim.iterations(res), Optim.converged(res), Optim.g_converged(res),
                      Optim.f_converged(res), Optim.x_converged(res), Optim.g_residual(res))

fmtc(c) = @sprintf("nonconv=%d/%d (maxiter %d, nonfinite-step %d) finite-valued-nonconv=%d material(|Δsite|>1e-3)=%d [pkg>ref: %d] | ll_pkg=%.6g ll_ref=%.6g gap(ref-pkg)=%.4g | worst site s=%d pkg=%.4g ref=%.4g |g(z)|=%.3g | z-mismatch=%d ref-fail=%d",
                   c.nonconv, 80, c.nmaxit, c.nnonfin, c.nfinite_nonconv, c.nmat, c.nhigh, c.ll_pkg, c.ll_rob, c.gap,
                   c.worst_site[1], c.worst_site[2], c.worst_site[3], c.worst_site[4], c.zmismatch, c.robfail)

# --------------------------------------------------------------------------------------
# Datasets (priority order; the budget guard skips the tail if needed).
# --------------------------------------------------------------------------------------
const SPECS = [
    (kind = :zip,  cov = false, seed = 101, sd = 0.8, r = 3.0, N = 10, βc_mean = 1.0, tag = "  [realistic sd0.8]"),
    (kind = :zip,  cov = false, seed = 102, sd = 1.6, r = 3.0, N = 10, βc_mean = 1.5, tag = "  [HARD sd1.6 βc1.5]"),
    (kind = :zinb, cov = false, seed = 103, sd = 0.8, r = 3.0, N = 10, βc_mean = 1.0, tag = "  [realistic sd0.8 r3]"),
    (kind = :zib,  cov = false, seed = 104, sd = 0.8, r = 3.0, N = 10, βc_mean = 0.0, tag = "  [realistic sd0.8 N10]"),
    (kind = :zip,  cov = true,  seed = 105, sd = 0.8, r = 3.0, N = 10, βc_mean = 1.0, tag = "  [realistic sd0.8 q1]"),
    (kind = :zinb, cov = true,  seed = 106, sd = 0.8, r = 3.0, N = 10, βc_mean = 1.0, tag = "  [realistic sd0.8 r3 q1]"),
    (kind = :zib,  cov = true,  seed = 107, sd = 0.8, r = 3.0, N = 10, βc_mean = 0.0, tag = "  [realistic sd0.8 N10 q1]"),
    (kind = :zinb, cov = false, seed = 108, sd = 1.5, r = 1.5, N = 10, βc_mean = 1.5, tag = "  [HARD sd1.5 r1.5 βc1.5]"),
    (kind = :zib,  cov = false, seed = 109, sd = 1.8, r = 3.0, N = 20, βc_mean = 0.0, tag = "  [HARD sd1.8 N20]"),
    (kind = :zip,  cov = true,  seed = 110, sd = 1.6, r = 3.0, N = 10, βc_mean = 1.5, tag = "  [HARD sd1.6 βc1.5 q1]"),
]

summary_rows = String[]
println("harness start; Julia threads = ", Threads.nthreads(), "; BLAS threads = ", LinearAlgebra.BLAS.get_num_threads())
for sp in (SMOKE ? SPECS[[1, 3, 4, 6]] : SPECS)
    if elapsed() > T_FIT_MAX
        println("\nSKIP (budget) ", sp); push!(summary_rows, "SKIPPED(budget) $(sp.kind) cov=$(sp.cov) seed=$(sp.seed)"); continue
    end
    m = build(sp.kind; cov = sp.cov, seed = sp.seed, sd = sp.sd, r = sp.r, N = sp.N, βc_mean = sp.βc_mean, tag = sp.tag)
    println("\n==== ", m.name, "  seed=", sp.seed, "  zeros=", round(count(==(0), m.Y) / length(m.Y); digits = 3),
            " max y=", maximum(m.Y), "  nθ=", length(m.θ0), "  t=", round(elapsed(); digits = 1), "s")
    tf = @elapsed fit = SMOKE ? m.fitfn(; iterations = 2) : m.fitfn()
    θ̂ = m.θhat(fit)
    println(@sprintf("  public fit: loglik=%.6f converged=%s iterations=%d  (%.1fs)", fit.loglik, fit.converged, fit.iterations, tf))
    # ---- HARNESS VALIDITY ----
    n̂ = m.negll(θ̂)
    valid = isfinite(fit.loglik) && abs(n̂ + fit.loglik) <= 1e-8 * max(1.0, abs(fit.loglik))
    println(@sprintf("  VALIDITY: replicated negll(θ̂)=%.10f  -fit.loglik=%.10f  |diff|=%.3g  -> %s",
                     n̂, -fit.loglik, abs(n̂ + fit.loglik), valid ? "OK" : "MISMATCH"))
    # ---- CLASS A census ----
    c0 = census(m, m.θ0); ch = census(m, θ̂); ct = census(m, m.θtrue)
    println("  CLASS A @θ0    : ", fmtc(c0))
    println("  CLASS A @θ̂     : ", fmtc(ch))
    println("  CLASS A @truth : ", fmtc(ct))
    println(@sprintf("  objective @truth: pkg negll=%.6g  ref negll=%.6g ;  fitted pkg loglik=%.6f vs pkg loglik@truth=%.6g",
                     m.negll(m.θtrue), negll_robust(m, m.θtrue), fit.loglik, -m.negll(m.θtrue)))
    # ---- CLASS B ----
    g, ns = fdgrad(m.negll, θ̂); gmax = maximum(abs, g)
    gc, nsc = fdgrad(m.negll, θ̂; rel = 1e-4); gcmax = maximum(abs, gc)
    gr, _ = fdgrad(θ -> negll_robust(m, θ), θ̂); grmax = maximum(abs, gr)
    flagB = fit.converged && gmax > 100 * GTOL
    println(@sprintf("  CLASS B: FD max|g| pkg objective @θ̂ = %.4g (coarse h=1e-4: %.4g; sentinel neighbours %d/%d) ; ref objective FD max|g| = %.4g ; converged=%s -> FLAG=%s",
                     gmax, gcmax, ns + nsc, 4 * length(θ̂), grmax, fit.converged, flagB))
    warm = "not run"; fresh = "not run"; dwarm = NaN; dfresh = NaN
    if elapsed() < T_EXTRA_MAX
        tw = @elapsed rw = lbfgs(m.negll, θ̂, WARM_IT)
        dwarm = -Optim.minimum(rw) - fit.loglik
        warm = @sprintf("Δloglik=%+.6g (%s, %.1fs)", dwarm, flags(rw), tw)
        println("  warm restart from θ̂ (cap $(WARM_IT)): ", warm)
    end
    if elapsed() < T_EXTRA_MAX
        tfr = @elapsed rf = lbfgs(m.negll, m.fresh(m.θ0), FRESH_IT)
        dfresh = -Optim.minimum(rf) - fit.loglik
        fresh = @sprintf("Δloglik=%+.6g (%s, %.1fs)", dfresh, flags(rf), tfr)
        println("  fresh start Λ0×0.3 (cap $(FRESH_IT)): ", fresh)
        cf = census(m, Optim.minimizer(rf))
        println("  CLASS A @fresh-end: ", fmtc(cf))
    end
    push!(summary_rows, @sprintf("%-26s seed=%d valid=%s ll=%.3f conv=%s it=%d | A: nonconv θ0/θ̂/truth=%d/%d/%d gap@θ̂=%.3g gap@truth=%.3g | B: max|g|=%.3g ref|g|=%.3g FLAG=%s warmΔ=%+.4g freshΔ=%+.4g",
                                 strip(m.name), sp.seed, valid, fit.loglik, fit.converged, fit.iterations,
                                 c0.nonconv, ch.nonconv, ct.nonconv, ch.gap, ct.gap, gmax, grmax, flagB, dwarm, dfresh))
    elapsed() > T_HARD && (println("HARD STOP (budget)"); break)
end
println("\n==== SUMMARY (", round(elapsed(); digits = 1), "s) ====")
foreach(println, summary_rows)
