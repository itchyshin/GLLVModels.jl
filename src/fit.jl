# Optim.jl-driven L-BFGS minimisation of the Gaussian GLLVM marginal
# negative log-likelihood. Matches the R engine's initial values and
# convergence tolerances; the head-to-head benchmark depends on this.
#
# Speed pass (MixedModels.jl-style):
#   - σ²_eps is profiled out analytically. The optimisation parameter
#     vector drops `log_σ_eps`. All other variance components are
#     reparameterised in σ²_eps units (Λ = σ_eps · L, etc.) so that the
#     profile NLL only depends on the rescaled parameters.
#   - At the optimum, σ̂²_eps = Q̃ / (n·p) where Q̃ is the quadratic form
#     computed with the rescaled (σ_eps = 1) covariance.
#   - ForwardDiff stays the AD backend for now. At the parameter counts
#     used by the benchmark grid (< 70 params), ForwardDiff's chunked
#     dual evaluation is competitive with or faster than ReverseDiff
#     for our hot path, which is dominated by p × n matrix ops with
#     cholesky on a small (K × K) matrix. The reverse-mode advantage
#     only kicks in at much larger param counts than this engine sees.

"""
    GllvmModel(p, K; K_W=0, has_diag=false, K_phy=0, has_phy_unique=false)

Immutable spec describing a Gaussian GLLVM. `p` traits, `K` (= K_B)
unit-tier latent factors, plus optional W tier (`K_W`), per-trait
diagonal random effects (`has_diag`), and phylogenetic block
(`K_phy` axes of `Λ_phy` and/or per-trait `σ_phy` when
`has_phy_unique`). The single-tier J1 case is the default
`K_W = 0`, `has_diag = false`, `K_phy = 0`, `has_phy_unique = false`.
"""
struct GllvmModel
    p::Int
    K::Int          # K_B (unit-tier rank); name kept for backward compatibility
    K_W::Int
    has_diag::Bool
    K_phy::Int
    has_phy_unique::Bool
end

GllvmModel(p::Integer, K::Integer) = GllvmModel(Int(p), Int(K), 0, false, 0, false)
GllvmModel(p::Integer, K::Integer, K_W::Integer, has_diag::Bool) =
    GllvmModel(Int(p), Int(K), Int(K_W), has_diag, 0, false)
GllvmModel(p::Integer, K::Integer, K_W::Integer, has_diag::Bool,
           K_phy::Integer, has_phy_unique::Bool) =
    GllvmModel(Int(p), Int(K), Int(K_W), has_diag, Int(K_phy), has_phy_unique)

abstract type AbstractIntegrationInfo end

"""
    GllvmFit

Result of `fit_gaussian_gllvm`. Holds the fitted parameters, the
converged log-likelihood, convergence info, and the raw Optim result.
"""
struct GllvmFit
    model::GllvmModel
    pars::NamedTuple
    logLik::Float64
    n_iter::Int
    converged::Bool
    optim_result
    cputime::Float64
    integration::Union{Nothing,AbstractIntegrationInfo}
end
GllvmFit(model,pars,logLik,n_iter,converged,optim_result,cputime)=
    GllvmFit(model,pars,logLik,n_iter,converged,optim_result,cputime,nothing)

"""
    fit_gaussian_gllvm(y; K, K_W=0, has_diag=false, K_phy=0,
                       has_phy_unique=false, Σ_phy=nothing, X=nothing,
                       X_lv=nothing,
                       σ_eps_init=1.0, λ_init=nothing, λ_W_init=nothing,
                       λ_phy_init=nothing,
                       σ²_B_init=0.1, σ²_W_init=0.1, σ_phy_init=0.1,
                       β_init=nothing, β_fixed=nothing, alpha_lv_init=nothing,
                       x_tol=1e-8, f_tol=0.0,
                       g_tol=1e-6, iterations=500) -> GllvmFit

L-BFGS minimisation of the closed-form Gaussian marginal NLL via
ForwardDiff gradients. Returns a `GllvmFit` with parameter estimates,
convergence diagnostics, and wall-clock fit time.

Stopping is left to the gradient test (`g_tol`) and the step size (`x_tol`);
the relative objective-change test is off by default (`f_tol = 0.0`, Optim's
own default). A relative `f_tol = 1e-10` stopped iterative fits (phylo-unique,
covariates) with gradients near 3e-3 and Wald SEs up to 0.25% off. Closed-form
single-variance fits take no L-BFGS steps and are unaffected.

Under the hood the optimisation runs on the profile NLL (σ²_eps and
optionally β profiled out analytically, MixedModels.jl-style). The
public API and parameter recovery semantics are unchanged.

J1 behaviour (`K_W = 0`, `has_diag = false`, `X = nothing`,
`K_phy = 0`, `has_phy_unique = false`, `Σ_phy = nothing`) is preserved
unchanged.

Optional extensions:
- J2-A-WD: `K_W::Integer = 0` (W-tier rank), `has_diag::Bool = false`
  (per-trait diagonal RE σ²_B, σ²_W). The W-tier scores are shared by all
  traits of a unit, so `Λ_W` adds the full `Λ_W Λ_Wᵀ` block to the per-unit
  covariance, as in gllvmTMB. With one unit per column of `y`, only
  `Λ Λᵀ + Λ_W Λ_Wᵀ` is identified, not the split between the two tiers
  (likewise only `σ²_B + σ²_W + σ²_eps`). A `K_W > 0` fit reaches the same
  log-likelihood and `sigma_y_site` as a single-tier fit with `K + K_W` axes;
  how `Λ` and `Λ_W` share that covariance depends on the starting values, so
  tier-scoped summaries (`communality`, `proportions`, `extract_Sigma` at
  `level = :unit` / `:unit_obs`, `getLV`) and Wald SEs of single `Λ` / `Λ_W`
  entries should not be interpreted. To separate between-unit from
  within-unit variation, use `fit_twolevel_gaussian` on repeated
  observations per unit.
- J3 phylogenetic: `K_phy::Integer = 0` (Λ_phy rank),
  `has_phy_unique::Bool = false` (per-trait σ_phy), and
  `Σ_phy::AbstractMatrix` (p × p species covariance, required when
  `K_phy > 0` or `has_phy_unique`).

Optional fixed effects:
- `X::AbstractArray{<:Real, 3}` of shape `(p, n_sites, q)`.
  A zero-column design, or fixing every coefficient to zero, gives the same
  zero-mean model as `X=nothing`.
- `β_init::AbstractVector` of length q (defaults to `zeros(q)`).
- `β_fixed` optionally fixes selected coefficients to zero; pass a Bool vector
  of length q, an integer index vector, or a Dict index=>0.

Optional predictor-informed latent-score mean (C1, ordinary Gaussian unit tier):
- `X_lv::AbstractMatrix` of shape `(n_sites, q_lv)` adds
  `z_total[s, :] = X_lv[s, :] * alpha_lv + z_innovation[s, :]` before the
  unit-tier loadings `Λ` enter the linear predictor.
- `alpha_lv_init::AbstractMatrix` of shape `(q_lv, K)` optionally seeds those
  latent-score predictor coefficients. The fitted `pars.alpha_lv` is
  rotation-dependent; `extract_lv_effects(fit)` returns the rotation-stable
  trait-effect matrix `Λ * alpha_lv'`.
- This C1 path is Gaussian-only. It admits Model A phylogenetic trait-covariance
  blocks (`K_phy > 0` and/or `has_phy_unique`) when `Σ_phy` is supplied, and
  rejects W-tier and diagonal random-effect blocks. Fits that combine fixed
  effects `X` with `X_lv` are point-estimate only for now:
  `confint_lv_effects` currently rejects them until a separate interval
  derivation lands.

The fit's `pars` NamedTuple always contains
`(σ_eps, Λ, β, β_fixed, alpha_lv, Λ_W, σ²_B, σ²_W, Λ_phy, σ_phy, θ_packed)` where
`β_fixed` is a Bool vector, `β` is expanded to the full coefficient length
with fixed entries set to zero, `alpha_lv` is a `q_lv × K` matrix or `nothing`,
and `Λ_W`, `σ²_B`, `σ²_W`, `Λ_phy`, `σ_phy` are `nothing` when the
corresponding flag is off.
"""
function _fit_gaussian_gllvm_exact(y::AbstractMatrix;
                            K::Integer,
                            K_W::Integer = 0,
                            has_diag::Bool = false,
                            K_phy::Integer = 0,
                            has_phy_unique::Bool = false,
                            Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                            X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                            X_lv::Union{Nothing, AbstractMatrix} = nothing,
                            σ_eps_init = 1.0,
                            λ_init = nothing,
                            λ_W_init = nothing,
                            λ_phy_init = nothing,
                            σ²_B_init = 0.1,
                            σ²_W_init = 0.1,
                            σ_phy_init = 0.1,
                            β_init = nothing,
                            β_fixed = nothing,
                            alpha_lv_init = nothing,
                            x_tol = 1e-8,
                            f_tol = 0.0,
                            g_tol = 1e-6,
                            iterations = 500)
    p, n = size(y)
    @assert K ≥ 1
    @assert K_W ≥ 0
    @assert K_phy ≥ 0
    n ≥ p || throw(ArgumentError(
        "fit_gaussian_gllvm needs n_sites ≥ p (got n_sites = $n, p = $p). This " *
        "closed-form Gaussian fitter is used by family = Normal() and by " *
        "family = Lognormal(), which fits a Gaussian GLLVM to log(Y). The " *
        "Laplace-fitted families (Poisson, Binomial, NegativeBinomial and the " *
        "rest) accept n_sites < p."))

    if (K_phy > 0 || has_phy_unique) && Σ_phy === nothing
        throw(ArgumentError(
            "Σ_phy is required when K_phy > 0 or has_phy_unique = true"))
    end
    if Σ_phy !== nothing
        size(Σ_phy, 1) == p && size(Σ_phy, 2) == p ||
            throw(ArgumentError(
                "Σ_phy must be p × p; got $(size(Σ_phy)) for p = $p"))
    end

    # Validate X dims if present, then drop fixed-zero columns for the optimiser.
    q_full = 0
    q = 0
    β_fixed_mask = Bool[]
    X_fit = X
    if X !== nothing
        size(X, 1) == p ||
            throw(ArgumentError("X first dim ($(size(X,1))) must equal p ($p)"))
        size(X, 2) == n ||
            throw(ArgumentError("X second dim ($(size(X,2))) must equal n_sites ($n)"))
        q_full = size(X, 3)
        β_fixed_mask = _fixed_zero_mask(β_fixed, q_full, "β_fixed")
        X_fit, _ = _slice_fixed_X(X, β_fixed_mask)
        q = size(X_fit, 3)
        β_init = _fixed_init_free(β_init, β_fixed_mask, "β_init")
    else
        if !(β_fixed === nothing || (β_fixed isa AbstractVector && isempty(β_fixed)))
            throw(ArgumentError("β_fixed requires X"))
        end
    end

    q_lv = 0
    X_lv_fit = X_lv
    if X_lv !== nothing
        K_W == 0 || throw(ArgumentError(
            "X_lv predictor-informed latent scores do not yet support K_W > 0"))
        !has_diag || throw(ArgumentError(
            "X_lv predictor-informed latent scores do not yet support has_diag = true"))
        # Model A: a phylo trait-covariance block (K_phy > 0 and/or has_phy_unique)
        # composes with the X_lv score mean on orthogonal axes — admitted for the
        # Gaussian path. It requires the species covariance Σ_phy.
        (!((K_phy > 0) || has_phy_unique) || Σ_phy !== nothing) || throw(ArgumentError(
            "X_lv with a phylo block (K_phy > 0 or has_phy_unique) requires Σ_phy"))
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

    has_phy_block = (K_phy > 0) || has_phy_unique

    # Build spec for profile NLL
    spec = (q = q, p = Int(p), K_B = Int(K), K_W = Int(K_W),
            has_diag = has_diag, K_phy = Int(K_phy),
            has_phy_unique = has_phy_unique)

    rr_B = rr_theta_len(p, K)
    rr_W = K_W > 0 ? rr_theta_len(p, K_W) : 0
    rr_phy = K_phy > 0 ? rr_theta_len(p, K_phy) : 0

    # ----- Warm-start via PPCA closed-form ML for Λ_B and σ_eps.
    # When the caller did not provide λ_init / σ_eps_init explicitly,
    # use the Tipping & Bishop (1999) closed-form solution as the
    # starting point. This is a multiplicative speedup on top of σ²_eps
    # profile-out: PPCA gives a near-optimal Λ_B for the dominant rank-K
    # piece, so LBFGS converges in a handful of iterations (often 1–5
    # for J1, ~10–20 for J2/J3) instead of 20–50 from the generic init.
    #
    # The PPCA closed form assumes y has zero mean. For the X case we
    # first do an OLS regression to get an initial β̂, then run PPCA on
    # the residuals. The OLS step costs one (q × q) solve and is
    # numerically trivial.
    σ_e₀ = float(σ_eps_init)

    # Default warm-start: only fires when the caller hasn't supplied
    # explicit initial values for Λ_B or σ_eps. This preserves the
    # explicit-init API (tests / experts) while giving the typical
    # caller the PPCA speed-up automatically.
    use_ppca_init = isnothing(λ_init) && (σ_eps_init == 1.0) && K < p
    if use_ppca_init
        # Build "residuals" y_resid for the PPCA: subtract X β̂_OLS if X
        # is provided, else use y directly.
        if X_fit !== nothing && q > 0 && isnothing(β_init)
            # OLS: stack the columns. r_s = y_s - X_s β; minimise sum_s ||r_s||² over β.
            # Build M (q × q) and v (q vec): M = Σ_s X_s' X_s, v = Σ_s X_s' y_s
            M_ols = zeros(Float64, q, q)
            v_ols = zeros(Float64, q)
            for s in 1:n
                Xs = @view X_fit[:, s, :]
                M_ols .+= Xs' * Xs
                v_ols .+= Xs' * @view(y[:, s])
            end
            β_ols = M_ols \ v_ols
            y_resid = similar(y, Float64)
            @inbounds for s in 1:n, t in 1:p
                μ = 0.0
                for k in 1:q
                    μ += X_fit[t, s, k] * β_ols[k]
                end
                y_resid[t, s] = y[t, s] - μ
            end
            Λ_ppca, σ_ppca = ppca_init(y_resid, K)
            σ_e₀ = σ_ppca
            λ_init = Λ_ppca
            if isnothing(β_init)
                β_init = β_ols
            end
        elseif q == 0
            # X === nothing, or X has zero free columns after masking (an
            # all-zero-column X, or every β entry fixed): there is no OLS
            # β̂ to subtract, so PPCA on y directly is the same closed-form
            # warm start as the X === nothing path. Gated on q == 0 rather
            # than X === nothing so a degenerate-but-non-nothing X_fit (a
            # p × n × 0 array from an empty or fully-fixed design) takes
            # the identical warm start as omitting X, instead of silently
            # falling through with no PPCA init at all (default init +
            # σ_eps_init = 1.0), which was the root cause of a ~1e-7
            # divergence in σ_eps / Λ / predict between the two routes
            # despite matching logLik (2026-08-31, S4a).
            Λ_ppca, σ_ppca = ppca_init(y, K)
            σ_e₀ = σ_ppca
            λ_init = Λ_ppca
        end
    end

    # β initial values: only included when β is NOT profiled out (phy path
    # or X-less case). For the non-phy + X case we profile β out via GLS
    # only when the per-evaluation cost is worth it; for q ≤ p we instead
    # keep β in the param vector (cheaper per gradient call at our sizes).
    # The decision: profile β when q ≤ small_threshold. Currently we keep
    # β in the param vector (profile_beta = false) because empirical
    # timing shows it dominates when q is small.
    profile_beta = false  # MixedModels-style β profile-out (off by default)

    do_profile_beta = profile_beta && !has_phy_block && q > 0
    β_in_params = !do_profile_beta && q > 0

    β₀ = if q > 0
        if isnothing(β_init)
            zeros(Float64, q)
        else
            length(β_init) == q ||
                throw(ArgumentError("β_init length ($(length(β_init))) must equal q ($q)"))
            collect(Float64, β_init)
        end
    else
        Float64[]
    end

    # L_B init: if λ_init is on the raw scale, divide by σ_eps_init.
    θ_B₀ = if isnothing(λ_init)
        init_theta_rr(p, K)
    else
        pack_lambda(λ_init ./ σ_e₀)
    end

    θ_W₀ = if K_W > 0
        if isnothing(λ_W_init)
            init_theta_rr(p, K_W)
        else
            pack_lambda(λ_W_init ./ σ_e₀)
        end
    else
        Float64[]
    end

    # C1 predictor-informed latent-score path. We use an explicit Gaussian
    # likelihood instead of the profiled-σ_eps objective because the mean term
    # Λ * alpha_lv' * X_lv depends jointly on Λ and alpha_lv. A separate profile
    # derivation can move this branch onto the fast path later.
    if X_lv_fit !== nothing
        alpha₀ = if alpha_lv_init === nothing
            zeros(Float64, q_lv, K)
        else
            Matrix{Float64}(alpha_lv_init)
        end
        # Model A phylo block init (same packed order as gaussian_lv_nll_packed:
        # σ_phy then θ_rr_phy, appended after Λ_B).
        σ_phy₀ = has_phy_unique ? fill(Float64(σ_phy_init), p) : Float64[]
        θ_phy₀ = K_phy > 0 ? init_theta_rr(p, K_phy) : Float64[]
        params₀ = vcat(β₀, vec(alpha₀), [log(σ_e₀)], θ_B₀, σ_phy₀, θ_phy₀)
        # The explicit (non-profiled) objective carries a free log σ_eps (required
        # because the mean Λ·alpha_lv'·X_lv couples Λ and alpha_lv). With a phylo
        # block the J3 marginal Cholesky can fail when the line search drives
        # σ_eps → 0; return a large finite value there so LBFGS backtracks instead
        # of crashing. AD-safe: a zero-derivative Dual at the rejected point.
        nll = function (params)
            return try
                gaussian_lv_nll_packed(params, y, Int(p), Int(K);
                                       X = X_fit, q = q,
                                       X_lv = X_lv_fit, q_lv = q_lv,
                                       K_phy = Int(K_phy),
                                       has_phy_unique = has_phy_unique,
                                       Σ_phy = Σ_phy)
            catch err
                err isa PosDefException || rethrow(err)
                # _NLL_SENTINEL, not an ad-hoc 1e10: the ad-hoc value sat BELOW
                # the 1e11 screen threshold, so a fit stranded on this penalty
                # plateau would have reported converged=true (the sentinel-escape
                # class, 2026-08-26). Aligned 2026-08-28.
                convert(eltype(params), _NLL_SENTINEL)
            end
        end

        opts = Optim.Options(
            x_abstol = x_tol,
            f_reltol = f_tol,
            g_tol    = g_tol,
            iterations = iterations,
            show_trace = false,
        )

        t0 = time()
        res = Optim.optimize(nll, params₀, Optim.LBFGS(), opts; autodiff = :forward)
        t1 = time()
        params_hat = Optim.minimizer(res)

        cursor = 0
        β_hat_free = if q > 0
            b = collect(params_hat[(cursor + 1):(cursor + q)])
            cursor += q
            b
        else
            Float64[]
        end
        alpha_hat = reshape(collect(params_hat[(cursor + 1):(cursor + q_lv * K)]),
                            q_lv, K)
        cursor += q_lv * K
        σ_eps_hat = exp(params_hat[cursor + 1])
        cursor += 1
        Λ_hat = unpack_lambda(@view(params_hat[(cursor + 1):(cursor + rr_B)]), p, K)
        cursor += rr_B
        # Model A phylo block (same packed order as gaussian_lv_nll_packed).
        σ_phy_hat = if has_phy_unique
            v = collect(params_hat[(cursor + 1):(cursor + p)]); cursor += p; v
        else
            nothing
        end
        Λ_phy_hat = if K_phy > 0
            rr_phy = rr_theta_len(p, K_phy)
            lp = unpack_lambda(@view(params_hat[(cursor + 1):(cursor + rr_phy)]), p, K_phy)
            cursor += rr_phy
            lp
        else
            nothing
        end
        β_full = X !== nothing ? _expand_fixed_zero(β_hat_free, β_fixed_mask) : β_hat_free

        # Keep the explicit packed vector for reproducibility. Interval engines
        # reject alpha_lv fits until the matching observed-information layout is
        # admitted.
        θ_packed_lv = collect(Float64, params_hat)

        return GllvmFit(
            GllvmModel(Int(p), Int(K), 0, false, Int(K_phy), has_phy_unique),
            (σ_eps = σ_eps_hat,
             Λ = Λ_hat,
             β = β_full,
             β_fixed = collect(Bool, β_fixed_mask),
             alpha_lv = alpha_hat,
             Λ_W = nothing,
             σ²_B = nothing,
             σ²_W = nothing,
             Λ_phy = Λ_phy_hat,
             σ_phy = σ_phy_hat,
             Σ_phy = Σ_phy,
             θ_packed = θ_packed_lv),
            # screened verdict (2026-08-28): a run stranded on the PosDef
            # sentinel plateau must not report converged=true.
            _fit_verdict(res)[1],
            Optim.iterations(res),
            _fit_verdict(res)[2],
            res,
            t1 - t0,
        )
    end

    # τ_B, τ_W on log-SD scale: log_τ = 0.5 * log(σ²_init / σ²_eps_init)
    log_τ_B₀ = has_diag ? fill(0.5 * log(σ²_B_init / σ_e₀^2), p) : Float64[]
    log_τ_W₀ = has_diag ? fill(0.5 * log(σ²_W_init / σ_e₀^2), p) : Float64[]

    # ρ_phy on identity (signed) scale: ρ_phy = σ_phy / σ_eps. Signed because
    # σ_phy now uses an identity link — see src/likelihood.jl, src/profile.jl.
    ρ_phy₀ = has_phy_unique ? fill(σ_phy_init / σ_e₀, p) : Float64[]

    θ_phy₀ = if K_phy > 0
        if isnothing(λ_phy_init)
            init_theta_rr(p, K_phy)
        else
            pack_lambda(λ_phy_init ./ σ_e₀)
        end
    else
        Float64[]
    end

    # Assemble profile params0 in the canonical order:
    # [β (if not profiled), log_τ_B, log_τ_W, θ_B, θ_W, ρ_phy, θ_phy]
    params₀ = if β_in_params
        vcat(β₀, log_τ_B₀, log_τ_W₀, θ_B₀, θ_W₀, ρ_phy₀, θ_phy₀)
    else
        vcat(log_τ_B₀, log_τ_W₀, θ_B₀, θ_W₀, ρ_phy₀, θ_phy₀)
    end

    nll = params -> gaussian_profile_nll(params, y;
                                          spec = spec, X = X_fit,
                                          Σ_phy = Σ_phy,
                                          profile_beta = do_profile_beta)

    opts = Optim.Options(
        x_abstol = x_tol,
        f_reltol = f_tol,
        g_tol    = g_tol,
        iterations = iterations,
        show_trace = false,
    )

    t0 = time()
    res = Optim.optimize(nll, params₀, Optim.LBFGS(), opts; autodiff = :forward)
    t1 = time()

    params_hat = Optim.minimizer(res)

    # Signed-σ_phy: escape sign-pattern basins via greedy single-flip restarts.
    # The identity-link σ_phy lets the optimisation reach any signed value, but
    # LBFGS is a *smooth* optimiser — to move from one sign pattern to another
    # it would have to pass through a saddle where some |σ_phy[t]| ≈ 0, and
    # in practice it gets trapped in the basin of its initial sign pattern.
    # We restore the discrete moves: at each converged point, evaluate the
    # NLL after flipping the sign of each individual σ_phy[t]; if any flip
    # strictly improves, restart LBFGS from the best single flip. Iterate
    # until no single flip helps. Bounded by O(p) outer iterations (each
    # iteration commits one sign change); in practice 0–2 restarts suffice.
    # Without this loop, LBFGS can stall at sign-pattern-local optima that
    # are *not* the global MLE — exactly the seed-17 / signed-MLE pathology
    # the identity link was introduced to fix.
    if has_phy_unique
        q_in_params = β_in_params ? q : 0
        diag_count = has_diag ? 2 * p : 0
        offset_pre_ρ = q_in_params + diag_count + rr_B + rr_W
        ρ_range = (offset_pre_ρ + 1):(offset_pre_ρ + p)
        max_flip_iters = p + 2  # safety cap
        for _ in 1:max_flip_iters
            nll_current = nll(params_hat)
            best_drop = 0.0
            best_idx  = 0
            for t in 1:p
                trial = copy(params_hat)
                trial[ρ_range[t]] = -trial[ρ_range[t]]
                drop = nll_current - nll(trial)
                if drop > best_drop + 1e-8
                    best_drop = drop
                    best_idx  = t
                end
            end
            best_idx == 0 && break
            # Restart LBFGS from the best single-flipped point.
            params_seed = copy(params_hat)
            params_seed[ρ_range[best_idx]] = -params_seed[ρ_range[best_idx]]
            res_flip = Optim.optimize(nll, params_seed, Optim.LBFGS(), opts;
                                       autodiff = :forward)
            if Optim.minimum(res_flip) < Optim.minimum(res) - 1e-8
                res = res_flip
                params_hat = Optim.minimizer(res)
            else
                break
            end
        end
        t1 = time()  # account for sign-exploration time in cputime
    end

    # Recover user-facing parameters
    rec = profile_recover(params_hat, y;
                          spec = spec, X = X_fit, Σ_phy = Σ_phy,
                          profile_beta = do_profile_beta)
    β_full = if X !== nothing
        _expand_fixed_zero(rec.β, β_fixed_mask)
    else
        rec.β
    end

    # Post-hoc global sign anchor for σ_phy (identity-link, signed).
    # The marginal likelihood is invariant under the joint flip
    # (σ_phy → -σ_phy, φ → -φ); this is the lone non-identifiable
    # symmetry (cross terms B[t,t'] = σ_phy[t]·σ_phy[t']·Σ_phy[t,t'] are
    # bilinear in σ_phy so a *global* sign flip leaves B unchanged).
    # Convention: flip so the largest-magnitude entry has non-negative
    # sign. Estimates are then deterministic up to the flip.
    if has_phy_unique && rec.σ_phy !== nothing
        i_max = argmax(abs.(rec.σ_phy))
        if rec.σ_phy[i_max] < 0
            σ_phy_anchored = -rec.σ_phy
            rec = (logLik = rec.logLik, σ_eps = rec.σ_eps, β = rec.β,
                   Λ_B = rec.Λ_B, Λ_W = rec.Λ_W,
                   σ²_B = rec.σ²_B, σ²_W = rec.σ²_W,
                   Λ_phy = rec.Λ_phy, σ_phy = σ_phy_anchored)
        end
    end

    # Build θ_packed in the *original* (legacy) layout for backward
    # compatibility with consumers that read .pars.θ_packed:
    # [β; log_σ_eps; (log_σ_B; log_σ_W if has_diag);
    #  θ_rr_B; θ_rr_W; (σ_phy if has_phy_unique, identity link); θ_rr_phy]
    legacy_pieces = Any[]
    if q > 0
        push!(legacy_pieces, rec.β)
    end
    push!(legacy_pieces, [log(rec.σ_eps)])
    if has_diag
        push!(legacy_pieces, log.(sqrt.(rec.σ²_B)))
        push!(legacy_pieces, log.(sqrt.(rec.σ²_W)))
    end
    push!(legacy_pieces, pack_lambda(rec.Λ_B))
    if K_W > 0
        push!(legacy_pieces, pack_lambda(rec.Λ_W))
    end
    if has_phy_unique
        # Identity link: store signed σ_phy values directly (no log).
        push!(legacy_pieces, collect(rec.σ_phy))
    end
    if K_phy > 0
        push!(legacy_pieces, pack_lambda(rec.Λ_phy))
    end
    θ_packed_legacy = reduce(vcat, legacy_pieces)

    return GllvmFit(
        GllvmModel(Int(p), Int(K), Int(K_W), has_diag,
                   Int(K_phy), has_phy_unique),
        (σ_eps = rec.σ_eps,
         Λ = rec.Λ_B,
         β = β_full,
         β_fixed = collect(Bool, β_fixed_mask),
         alpha_lv = nothing,
         Λ_W = rec.Λ_W,
         σ²_B = rec.σ²_B,
         σ²_W = rec.σ²_W,
         Λ_phy = rec.Λ_phy,
         σ_phy = rec.σ_phy,
         θ_packed = θ_packed_legacy),
        rec.logLik,
        Optim.iterations(res),
        # screened (2026-08-28), defensively: this legacy path's objective
        # (gaussian_profile_nll) has no catch/sentinel return today, so the
        # screen only guards against a future one; logLik stays the
        # recomputed closed-form value.
        (_fit_verdict(res)[2]),
        res,
        t1 - t0,
    )
end
