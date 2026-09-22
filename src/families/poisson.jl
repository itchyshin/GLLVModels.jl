# Poisson family pieces for the generic Laplace core (src/families/laplace.jl).
# y_t ~ Poisson(μ_t), μ = linkinv(link, η) (log link ⇒ μ = exp η). E[y]=μ, Var=μ.
# Score/weight wrt η: with the log link (me = μ) these reduce to (y − μ) and μ.
# Poisson has no trial count, so `n` is ignored.
_clamp_mu(::Poisson, μ) = max(μ, 1e-12)
_glm_score(::Poisson, μ, n, me, y) = (y - μ) / μ * me
# Canonical link: y enters η linearly, so −∂²ℓ/∂η² = μ is y-free and the
# observed curvature coincides with the Fisher weight POINTWISE. This family is
# therefore bit-for-bit unaffected by the log-det curvature selector.
_glm_weight_matches_observed(::Poisson, ::LogLink) = true

_glm_weight(::Poisson, μ, n, me)   = me^2 / μ
_glm_logpdf(::Poisson, μ, n, y)    = logpdf(Poisson(μ), Int(y))

"""
    poisson_marginal_loglik_laplace(Y, Λ, β, link=LogLink(); kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a Poisson GLLVM — a
thin wrapper over the family-generic `marginal_loglik_laplace` with `Poisson()`.
`Y` is the p×n integer count matrix; `Λ` p×K; `β` length-p. Poisson has no trial
counts, so a unit `N` is supplied internally.
"""
poisson_marginal_loglik_laplace(Y::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, link::Link = LogLink(); kwargs...) =
    marginal_loglik_laplace(Poisson(), Y, ones(Int, size(Y)), Λ, β, link; kwargs...)

# ---------------------------------------------------------------------------
# Fit driver (Poisson slice 2).
# ---------------------------------------------------------------------------

"""
    PoissonFit

Result of [`fit_poisson_gllvm`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the `link`, the fitted marginal `loglik`, the optimiser `converged`
flag, and `iterations`. Fits using `X_lv` additionally retain `alpha_lv`, the
raw latent-axis coefficients for the predictor-informed score mean; use
[`extract_lv_effects`](@ref) for the rotation-stable trait-scale product
`Λ * alpha_lv'`. Optional `integration` records requested/actual AGHQ, node
counts, controls, final caches and retained start diagnostics. It is `nothing`
for default Laplace and legacy constructors. AGHQ convergence is with respect
to the frozen-node surrogate, not derivatives through changing adaptation.
"""
struct PoissonFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    alpha_lv::Union{Nothing, Matrix{Float64}}
    theta_packed::Vector{Float64}
    hessian::Symbol   # Laplace log-det curvature; AGHQ is recorded separately
    integration::Union{Nothing,AGHQFitInfo}
end

PoissonFit(β, Λ, link, loglik, converged, iterations, alpha_lv, theta_packed, hessian) =
    PoissonFit(β, Λ, link, loglik, converged, iterations, alpha_lv, theta_packed, hessian, nothing)

# Positional compatibility constructor (2026-08-28): every pre-existing
# construction site builds a default-curvature fit; the `hessian` field
# records the objective identity so `confint`/bootstrap can rebuild THE
# SAME objective instead of guessing (the audit's confint-consistency class).
PoissonFit(β, Λ, link, loglik, converged, iterations, alpha_lv, theta_packed) =
    PoissonFit(β, Λ, link, loglik, converged, iterations, alpha_lv, theta_packed, _default_hessian(Poisson(), link))

PoissonFit(β::Vector{Float64}, Λ::Matrix{Float64}, link::Link,
           loglik::Float64, converged::Bool, iterations::Int) =
    PoissonFit(β, Λ, link, loglik, converged, iterations, nothing, Float64[])

function Base.show(io::IO, f::PoissonFit)
    p, K = size(f.Λ)
    print(io, "PoissonFit(p=", p, ", K=", K, ", link=", nameof(typeof(f.link)),
          f.alpha_lv === nothing ? "" : ", X_lv=true",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.integration===nothing ? "" : ", "*string(f.integration.actual)*"(k="*string(f.integration.k)*")",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    poisson_lv_nll_packed(params, Y, p, K, link; X_lv, q_lv, kwargs...) -> Real

Negative Laplace log-likelihood for the predictor-informed latent-score Poisson
model. Parameter layout:

- `params[1:p]` = per-trait intercepts `β`;
- next `q_lv * K` entries = `alpha_lv`, reshaped as `q_lv × K`;
- remaining entries = packed reduced-rank loadings `Λ`.

The conditional latent variable is the zero-mean innovation. The predictor mean
enters the Laplace core as the parameter-dependent offset
`Λ * alpha_lv' * X_lv[s, :]` (the same offset trick as the binomial X_lv route).
"""
function poisson_lv_nll_packed(params::AbstractVector, Y::AbstractMatrix,
        p::Integer, K::Integer, link::Link;
        X_lv::AbstractMatrix, q_lv::Integer,
        mask = nothing, offset = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    size(Y, 1) == p ||
        throw(ArgumentError("Y first dim ($(size(Y, 1))) must equal p ($p)"))
    n = size(Y, 2)
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
    size(X_lv, 2) == q_lv ||
        throw(ArgumentError("X_lv second dim ($(size(X_lv, 2))) must equal q_lv ($q_lv)"))
    q_lv > 0 || throw(ArgumentError("q_lv must be positive"))

    rr = rr_theta_len(p, K)
    n_expected = p + q_lv * K + rr
    length(params) == n_expected || throw(ArgumentError(
        "params length ($(length(params))) must equal $n_expected " *
        "(p=$p + alpha_lv=$(q_lv * K) + rr=$rr)"))

    cursor = 0
    β = @view params[(cursor + 1):(cursor + p)]
    cursor += p
    alpha_vec = @view params[(cursor + 1):(cursor + q_lv * K)]
    alpha_lv = reshape(alpha_vec, q_lv, K)
    cursor += q_lv * K
    θ_rr = @view params[(cursor + 1):(cursor + rr)]
    Λ = unpack_lambda(θ_rr, p, K)

    lv_offset = _lv_mean_eta(Λ, X_lv, alpha_lv)
    off = offset === nothing ? lv_offset : offset .+ lv_offset
    return -poisson_marginal_loglik_laplace(Y, Λ, β, link;
                                            mask = mask, offset = off,
                                            maxiter = maxiter, tol = tol)
end

"""
    fit_poisson_gllvm(Y; K, link=LogLink(), mask=nothing, …) -> PoissonFit

Fit a Poisson GLLVM by L-BFGS on the Laplace marginal log-likelihood
(`poisson_marginal_loglik_laplace`). `Y` is a p×n integer count matrix
(responses × sites) that may contain `missing` (gllvm-style NA); `K` the latent
dimension. Optimises intercepts `β` and loadings `Λ`. The default analytic
Laplace gradient is used on the plain no-mask/no-offset path, with an internal
finite-difference fallback; masked or offset fits use finite differences. Warm
start = empirical log-mean intercepts + an SVD (PPCA-style) loadings init.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or simply include
`missing` entries in `Y` — either way the masked cells are dropped from the
marginal *and* from the warm start, so the fit depends only on the observed cells
(it is invariant to whatever sits in the masked positions).

Offset: pass a p×n `offset` (known additive term in `η = β + offset + Λz`, e.g.
log-exposure/effort/area). It is subtracted from the link-scale warm start so `β`
estimates the offset-free intercept.

`hessian` selects the Laplace log-det curvature only (`:fisher` expected /
`:observed` joint — TMB's choice); the inner mode search is always
Fisher-scored. Default: canonical log link — the two coincide. Omitting it is exactly the pre-kwarg
behaviour.
"""
function _fit_poisson_gllvm_laplace(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        gradient::Symbol = :analytic,
        hessian::Symbol = _default_hessian(Poisson(), link),
        β_init = nothing, Λ_init = nothing,
        X_lv::Union{Nothing, AbstractMatrix} = nothing,
        alpha_lv_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    rr = rr_theta_len(p, K)
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_poisson_gllvm: hessian must be :fisher or :observed; got :$hessian"))
    if X_lv !== nothing && hessian !== _default_hessian(Poisson(), link)
        throw(ArgumentError("fit_poisson_gllvm: a non-default hessian is not yet supported " *
                            "together with X_lv (the packed X_lv objective does not thread it)"))
    end

    # Predictor-informed latent-score mean (Design 73 / gllvmTMB C1): X_lv (n×q_lv)
    # activates joint estimation of alpha_lv via the parameter-dependent offset
    # Λ * alpha_lv' * X_lv[s, :]. Point-estimate route only.
    q_lv = 0
    X_lv_fit = nothing
    if X_lv !== nothing
        K > 0 || throw(ArgumentError("X_lv requires a positive latent dimension K"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
        q_lv = size(X_lv, 2)
        q_lv > 0 || throw(ArgumentError("X_lv must have at least one predictor column"))
        X_lv_fit = Matrix{Float64}(X_lv)
        if alpha_lv_init !== nothing
            size(alpha_lv_init, 1) == q_lv ||
                throw(ArgumentError(
                    "alpha_lv_init first dim ($(size(alpha_lv_init, 1))) must equal size(X_lv, 2) ($q_lv)"))
            size(alpha_lv_init, 2) == K ||
                throw(ArgumentError(
                    "alpha_lv_init second dim ($(size(alpha_lv_init, 2))) must equal K ($K)"))
        end
    elseif alpha_lv_init !== nothing
        throw(ArgumentError("alpha_lv_init requires X_lv"))
    end

    # NA handling: derive the observation mask (explicit `mask`, else from `missing`)
    # and a sanitized count matrix with a safe placeholder in the masked cells.
    msk = mask === nothing ? (any(ismissing, Y) ? observed_mask(Y) : nothing) : mask
    Yc = Integer.(_sanitize_missing(Y, 0))

    # warm start: empirical log-scale intercepts + SVD (PPCA-like) loadings.
    # With an offset (η = β + offset + Λz), subtract it from the link-scale data so
    # β₀/Λ₀ estimate the offset-free part. Masked cells are overwritten with their
    # row's observed mean so neither the intercept nor the SVD sees the placeholder.
    Zemp = [linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    if msk !== nothing
        @inbounds for t in 1:p
            obs = view(msk, t, :)
            cnt = count(obs)
            rowmean = cnt > 0 ? sum(Zemp[t, i] for i in 1:n if msk[t, i]) / cnt : 0.0
            for i in 1:n
                msk[t, i] || (Zemp[t, i] = rowmean)
            end
        end
    end
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Zc = Zemp .- β0
    Λ0 = if Λ_init === nothing
        F = svd(Zc)
        kk = min(K, length(F.S))
        L = zeros(p, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end

    # alpha_lv warm start: least-squares regression of the initial PPCA scores on X_lv.
    alpha0 = if X_lv_fit === nothing
        nothing
    elseif alpha_lv_init === nothing
        F = svd(Zc)
        kk = min(K, length(F.S))
        scores0 = zeros(Float64, n, K)
        @inbounds for j in 1:kk
            scores0[:, j] = sqrt(n) .* F.V[:, j]
        end
        X_lv_fit \ scores0
    else
        Matrix{Float64}(alpha_lv_init)
    end

    θ0 = vcat(β0, pack_lambda(Λ0))
    N1 = ones(Int, size(Yc))                     # unit trials, hoisted out of the per-eval closure
    # R3 (workspace reuse, core070): one Float64 LaplaceModeWorkspace shared across
    # every site of every `negll` evaluation, instead of `_laplace_mode` allocating
    # its nine buffers fresh per site (543MB churn measured at p=50 — see
    # docs/dev-log/core070/poisson-perf-diagnosis.md). Concrete-only: `negll` is
    # always evaluated at a plain Float64 θ (the analytic gradient below never
    # differentiates through this closure), so this workspace's element type never
    # needs to be a dual.
    ws_negll = LaplaceModeWorkspace(Float64, p, K)
    # R8 (shared mode solve, S6 item 1): `zs`, when given, is threaded straight
    # into `marginal_loglik_laplace`'s per-site `z_precomputed`, so an `fg!` call
    # that already solved the same modes for the gradient path does not make
    # `negll` solve them again — see `fg!` below.
    function negll(θ; zs = nothing)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        v = try
            -marginal_loglik_laplace(Poisson(), Yc, N1, Λ, β, link; mask = msk, offset = offset,
                                     hessian = hessian, zs = zs,
                                     maxiter = newton_maxiter, tol = newton_tol, ws = ws_negll)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    # Exact gradient (issue #65): the implicit-step ForwardDiff gradient. Valid
    # for the plain Poisson marginal and the masked marginal — the mask is passed
    # through (masked-cell score/weight are zeroed, matching the masked objective;
    # FD-verified in test/test_missing_response.jl). The offset path still uses the
    # finite-difference gradient (the analytic gradient does not carry an offset).
    # A finite-difference fallback also covers any θ where the analytic gradient is
    # non-finite (e.g. a pathological line-search probe).
    res = if X_lv_fit !== nothing
        # Predictor-informed latent-score route: joint (β, alpha_lv, Λ) by finite
        # differences — the offset depends jointly on Λ and alpha_lv.
        θ0_lv = vcat(β0, vec(alpha0), pack_lambda(Λ0))
        negll_lv = θ -> begin
            v = try
                poisson_lv_nll_packed(θ, Yc, p, K, link;
                                      X_lv = X_lv_fit, q_lv = q_lv,
                                      mask = msk, offset = offset,
                                      maxiter = newton_maxiter, tol = newton_tol)
            catch
                return 1e12
            end
            return isfinite(v) ? v : 1e12
        end
        Optim.optimize(negll_lv, θ0_lv, ls, opts; autodiff = :finite)
    elseif gradient === :analytic && offset === nothing &&
           (hessian === _default_hessian(Poisson(), link) ||
            _glm_weight_matches_observed(Poisson(), link))
        # R4 (only_fg!, core070): a single combined closure so an accepted iterate
        # that needs BOTH the value and the gradient pays for value+gradient in one
        # `Optim.optimize` bookkeeping pass rather than two separate closures the
        # optimizer must call at the same θ. See
        # docs/dev-log/core070/poisson-perf-repair-notes.md for the measured effect.
        #
        # R7 (one GradientConfig per fit, S6 item 2): built ONCE here (not inside
        # `poisson_laplace_grad`, which would otherwise let `ForwardDiff.gradient`
        # allocate a fresh config — seed/partial buffers sized to nθ — on EVERY
        # Optim iteration) and reused across the whole fit. Valid because
        # `PoissonMargClosure`'s TYPE (not the field VALUES) is what the config's
        # tag is keyed on, and that type only depends on `Yc`'s type, `p`/`rr`/`K`,
        # and whether `msk` is `nothing` — all fixed for this fit. Chunk 32
        # measured best of {12, 24, 32} at p=20/50 (bench, see commit body);
        # `poisson_laplace_grad` falls back to an uncached config on any mismatch,
        # so a wrong/stale `_grad_gcfg` here is never unsafe, only non-optimal.
        _grad_chunk = min(32, p + rr)
        _grad_gcfg = try
            _proto_ẑs = [zeros(K) for _ in 1:n]
            _proto_marg = PoissonMargClosure(Yc, p, rr, K, _proto_ẑs, msk, Ref{Any}(nothing))
            ForwardDiff.GradientConfig(_proto_marg, θ0, ForwardDiff.Chunk{_grad_chunk}())
        catch
            nothing
        end
        function fg!(F, G, θ)
            # R8 (shared mode solve, S6 item 1): when this call needs BOTH the
            # value and the gradient at θ, solve the per-site Newton mode ONCE
            # here and hand it to both `negll` and `poisson_laplace_grad` — before
            # this change each solved its own copy of the identical fixed point
            # (2 solves/site/iteration whenever Optim asked for F+G together).
            shared_ẑs = nothing
            if G !== nothing
                β = θ[1:p]; Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
                # `poisson_laplace_grad`'s hoist always solves the mode under
                # `LogLink()` (pre-existing; not introduced here — see its
                # docstring); sharing that mode into `negll` is only valid when
                # this fit's own `link` is ALSO `LogLink()`, so a non-default
                # link keeps solving its own (correctly link-aware) mode in
                # `negll`, unaffected by this change.
                if F !== nothing && link isa LogLink
                    Λv = unpack_lambda(pack_lambda(Λ), p, K)  # round-tripped, matches poisson_laplace_grad's convention
                    # The shared mode must stop where `negll`'s own solve would have
                    # stopped: the fit's `newton_maxiter` / `newton_tol`, not
                    # `_laplace_mode`'s defaults, or a caller asking for a tighter
                    # mode would silently get the default one in the value path.
                    shared_ẑs = try
                        _poisson_hoist_zhats(Yc, Λv, float.(β); mask = msk,
                                             maxiter = newton_maxiter, tol = newton_tol)
                    catch
                        nothing
                    end
                end
                gg = try
                    poisson_laplace_grad(Yc, Λ, β; mask = msk, gcfg = _grad_gcfg, ẑs = shared_ẑs)
                catch
                    nothing
                end
                if gg === nothing || !all(isfinite, gg)
                    hh = 1e-6
                    @inbounds for i in eachindex(θ)
                        θp = copy(θ); θp[i] += hh; θm = copy(θ); θm[i] -= hh
                        G[i] = (negll(θp) - negll(θm)) / (2hh)
                    end
                else
                    G .= .-gg                   # ∇(negll) = −∇(marginal)
                end
            end
            if F !== nothing
                return negll(θ; zs = shared_ẑs)
            end
            return nothing
        end
        Optim.optimize(Optim.only_fg!(fg!), θ0, ls, opts)
    else
        Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    end
    θ̂ = Optim.minimizer(res)
    if X_lv_fit !== nothing
        cursor = 0
        β̂ = collect(θ̂[(cursor + 1):(cursor + p)])
        cursor += p
        alpha_hat = reshape(collect(θ̂[(cursor + 1):(cursor + q_lv * K)]), q_lv, K)
        cursor += q_lv * K
        Λ̂ = unpack_lambda(@view(θ̂[(cursor + 1):(cursor + rr)]), p, K)
        return PoissonFit(β̂, Λ̂, link, _fit_verdict(res)...,
                          alpha_hat, collect(Float64, θ̂), hessian)
    else
        β̂ = θ̂[1:p]
        Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
        return PoissonFit(β̂, Λ̂, link, _fit_verdict(res)..., nothing, Float64[], hessian)
    end
end
