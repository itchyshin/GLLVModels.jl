# Latent-dimension selection — a practical workflow tool.
#
# Fits the same GLLVM at a sweep of latent dimensions K = 1:Kmax via the
# unified `fit_gllvm` dispatcher, records the information criteria (using the
# same `_nparams`/`_loglik` path that `aic`/`bic` use), and picks the K that
# minimises the chosen criterion among the fits that pass the guard below.
#
# Guard. A K model nests the (K−1) model (a zero loading column), so a correct
# maximum can never have a lower log-likelihood. A fit that throws, reports
# non-convergence, or falls below the best converged, non-runaway fit at any
# smaller K (accepted or not; a rejected fit is still a point inside every larger
# model) by more than max(tol, 1e-6·|logLik|) is not a valid candidate. Before rejecting a non-monotone or unconverged fit, the sweep
# refits that K once from the previous solution plus a small new column (a warm
# start), where the family fitter accepts `β_init`/`Λ_init`. Every attempted K is
# recorded in `attempts` with its status, so nothing is dropped silently.
# Measured on origin/main 2847b5dbf (NB2, n = 300, p = 20, true K = 3): K = 3 and
# K = 4 converged below K = 2, which made every criterion choose K = 2.

"""
    LVSelection

Result of a latent-dimension sweep (see [`select_lv`](@ref)). The vectors are
aligned by position and describe only the ACCEPTED fits: entry `i` is `K[i]`.

Fields:
- `K::Vector{Int}` — the latent dimensions whose fits were accepted.
- `nparams::Vector{Int}` — free-parameter count per fit (the same `_nparams`
  path `aic`/`bic` use; loadings counted modulo the `K(K−1)/2` rotational df).
- `loglik::Vector{Float64}` — maximised marginal log-likelihood per fit (under a
  loading ridge, the unpenalised log-likelihood at the penalised optimum).
- `aic::Vector{Float64}` — Akaike information criterion per fit.
- `bic::Vector{Float64}` — Bayesian information criterion per fit, penalty
  `log(p·n)` (observed cells, R's convention).
- `bic_sites::Vector{Float64}` — BIC with penalty `log(n)`, `n` the number of
  sites (columns of `Y` with at least one observed cell).
- `best_k::Int` — the accepted `K` minimising the chosen criterion.
- `best::Any` — the chosen fitted model (the one at `best_k`).
- `attempts::Vector` — one named tuple `(K, status, loglik, message)` per
  attempted `K`. `status` is `:ok`, `:warm_start` (accepted after a refit from
  the previous solution), `:nonmonotone` (log-likelihood below the last accepted
  `K`; under a loading ridge the penalised objective ℓ − ½Σλ²/τ² is compared,
  while `loglik` stays the unpenalised value), `:unconverged`, `:runaway` (inflated or separated loadings; `message`
  gives the statistic), or `:failed` (the fit threw; `message` says why).
"""
struct LVSelection
    K::Vector{Int}
    nparams::Vector{Int}
    loglik::Vector{Float64}
    aic::Vector{Float64}
    bic::Vector{Float64}
    bic_sites::Vector{Float64}
    best_k::Int
    best::Any
    attempts::Vector{NamedTuple{(:K, :status, :loglik, :message),Tuple{Int,Symbol,Float64,String}}}
end

# Warm start needs top-level `β`/`Λ` fields: Gaussian `GllvmFit` keeps them in
# `pars`, so Gaussian sweeps get the guard but no retry (they are well-behaved).
# Warm-start keywords for a K-dimensional refit from the last accepted fit (any
# smaller K), or `nothing` when the fit does not expose `β`/`Λ`. The added columns
# keep the top K×K block lower-triangular (zeros above the diagonal, 0.1 on and
# below it). Assumes `prev.Λ` is the fitter's constrained (unrotated) matrix.
function _lv_warm_start(prev, K::Integer)
    (hasproperty(prev, :β) && hasproperty(prev, :Λ)) || return nothing
    Λp = prev.Λ
    p, Kp = size(Λp, 1), size(Λp, 2)
    (Kp < K <= p) || return nothing
    Λ0 = zeros(p, K)
    Λ0[:, 1:Kp] .= Λp
    for j in (Kp + 1):K          # lower-triangular padding: zeros above row j
        Λ0[j:end, j] .= 0.1
    end
    return (β_init = copy(prev.β), Λ_init = Λ0)
end

_lv_converged(fit) = hasproperty(fit, :converged) ? Bool(fit.converged) : true

# Value the non-monotone check compares. A K model nests every smaller one, so its
# maximised objective cannot be lower. Under a loading ridge that objective is the
# PENALISED one, ℓ − ½Σλ²/τ²; the unpenalised ℓ at the ridge optimum can fall as K
# grows. Criteria still use the unpenalised ℓ (`_loglik`).
function _lv_bar_value(fit)
    τ = hasproperty(fit, :loading_ridge) ? Float64(fit.loading_ridge) : Inf
    isfinite(τ) || return _loglik(fit)
    return _loglik(fit) - 0.5 * sum(abs2, fit.Λ) / τ^2
end

# Runaway detector. Latent variables are standardised (u ~ N(0, I)), so a trait's
# loading row norm is its latent SD on the link scale: ~4 is as strong as real
# gradients get, 10 is saturated. Two modes need two statistics (vault note "Two
# runaway modes in GLLVM loadings", gllvmTMB lane 2026-07-30):
#   Mode B, common inflation (any family): a SCALE check, max row norm > max_latent_sd.
#   Mode A, one binary trait separating: a RATIO check, the largest per-trait max
#     |loading| over the median, ≥ ratio_max. Binomial only; the ratio is blind to
#     Mode B by construction (Λ → cΛ leaves it unchanged).
# Gaussian (Normal family) and identity-link fits are skipped: their loadings are in
# data units (e.g. `pervar = true` fits carry Λ but no link field). Both thresholds
# are provisional (measured in gllvmTMB: ratio 25 gave 96.3% detection, 0/551 false
# positives on binomial) and are to be recalibrated on the auto-d recovery grid.
# Returns "" when healthy, else the reason.
function _lv_runaway(fit, family; max_latent_sd::Real, ratio_max::Real)
    hasproperty(fit, :Λ) || return ""
    family isa Normal && return ""
    hasproperty(fit, :link) && fit.link isa IdentityLink && return ""
    Λ = fit.Λ
    (ndims(Λ) == 2 && size(Λ, 2) >= 1 && size(Λ, 1) >= 1) || return ""
    rows = [sqrt(sum(abs2, @view Λ[t, :])) for t in axes(Λ, 1)]
    smax = maximum(rows)
    if smax > max_latent_sd
        t = argmax(rows)
        return "latent SD $(round(smax; sigdigits = 3)) on the link scale for trait $t (> $max_latent_sd)"
    end
    if family isa Binomial
        m = [maximum(abs, @view Λ[t, :]) for t in axes(Λ, 1)]
        r = maximum(m) / max(median(m), eps())
        r >= ratio_max && return "loading ratio $(round(r; sigdigits = 3)) for trait $(argmax(m)) (≥ $ratio_max; separation)"
    end
    return ""
end

"""
    select_lv(Y; family = Normal(), Kmax = 3, criterion = :bic_sites,
              warm_start = true, tol = 1e-3, max_latent_sd = 10.0,
              ratio_max = 25.0, binary_ridge = 2.0, kwargs...) -> LVSelection

Latent-dimension selection: fit `fit_gllvm(Y; family, K = k, kwargs...)` for
`k in 1:Kmax` and pick the `K` minimising `criterion` among the fits that pass a
guard: `:bic_sites` (default; penalty `log(n)`, sites), `:bic` (penalty
`log(p·n)`, observed cells) or `:aic`. The default follows a recovery simulation
(17 687 datasets with known K): `:bic_sites` recovered the true K most often for
Gaussian and Poisson responses; `:bic` picked too few dimensions at small n.
Negative-binomial recovery has not yet been measured on the corrected
negative-binomial fitting code, so no rate is claimed for it.

Binary (single-trial `Binomial`) data get a loading ridge during the sweep:
every fit in the sweep — including a
warm-start refit — is called with `loading_ridge = binary_ridge` (default
`2.0`), unless the caller already passed their own `loading_ridge` (which then
wins) or `binary_ridge = Inf` (which disables it, restoring today's unpenalised
behaviour). This is because most unpenalised Bernoulli fits beyond `K = 1` run
away (see the runaway guard below): in an R recovery experiment with 20 species
and `n = 120`, the ridge recovered the true `K = 2` in 8/10 datasets versus 4/10
without it. The information criteria are still computed on the UNPENALISED
Laplace log-likelihood at the ridge optimum (`GLLVModels._loglik(fit)`); the
check that the log-likelihood does not fall as `K` grows uses the penalised value
`ℓ − ½Σλ²/τ²`, which is what nesting guarantees under a ridge.
The criteria count the loadings at their nominal number of parameters. Under
the ridge the effective number is smaller, and the gap grows with `K`, so ridge
BIC leans towards smaller `K` (a conservative bias). The ridge applies only on
the Laplace `fit_binomial_gllvm` route: with `aghq`, `row_eff`, `grouping`,
`phylo`, `disp_group` or `pervar` the sweep runs unpenalised and each attempt's
`message` says so. `confint` refuses a ridge fit, because it is a penalised
estimate; refit at the chosen `K` with `loading_ridge = Inf` for intervals.
Multi-trial `Binomial` (an `N` keyword whose entries are not all one) and every
other family are unaffected and never receive `loading_ridge`.

The guard rejects a fit that throws, is a runaway (below), or has a
log-likelihood more than `max(tol, 1e-6·|ℓ|)` below the best converged,
non-runaway fit at any smaller `K` (a `K` model nests every smaller one, so its
maximum cannot be lower; runaway fits are excluded from this bar because
separation inflates their log-likelihood). A fit whose optimiser did not report
convergence is kept, with a message in `attempts`, unless it is also runaway or
non-monotone; `require_converged = true` rejects it instead (on the auto-d recovery
grid the strict rule lost recovery for Poisson data, and every
broken unconverged fit was already caught by the other checks). It also rejects a runaway fit:
because the latent variables are standardised, a trait's loading row norm is its
latent SD on the link scale, and a value above `max_latent_sd` (default 10, a
saturated effect) marks common inflation of the loadings; for `Binomial`, one
trait's largest loading at `ratio_max` (default 25) times the median trait's
marks separation. Identity-link (Gaussian) fits skip the runaway check. Both
thresholds are provisional. With `warm_start = true`, a rejected
(non-monotone, unconverged or runaway) fit is first refitted once from the last
accepted solution padded with small lower-triangular loading columns, for
families whose fitter accepts
`β_init`/`Λ_init`; the refit is kept only if it passes every check. Rejected fits stay in
`attempts` with their reason and are never chosen. At least one `K` must be
accepted or an error is thrown. Healthy sweeps never trigger a refit, so their
results are unchanged. An interrupt is never swallowed.

The information criteria are read straight off the fits via [`aic`](@ref) and
[`bic`](@ref) (BIC uses `nobs(fit, Y)`, R's p·n observed-cell count), so the
parameter counting matches the single-fit accessors exactly.

The chosen `K` is itself an estimate: intervals and tests computed on
`sel.best` are conditional on it and do not include uncertainty about `K`.

`family` is a Distributions.jl marker (the [`fit_gllvm`](@ref) convention);
`kwargs...` pass through to the underlying fitter.

```julia
sel = select_lv(Y; family = Poisson(), Kmax = 3)   # pick K by BIC with log(n sites)
sel.best_k          # selected latent dimension
sel.best            # the fitted model at that K
sel.attempts        # every K tried, with status
```
"""
function select_lv(Y::AbstractMatrix; family = Normal(), Kmax::Integer = 3,
                   criterion::Symbol = :bic_sites, warm_start::Bool = true,
                   tol::Real = 1e-3, max_latent_sd::Real = 10.0,
                   ratio_max::Real = 25.0, require_converged::Bool = false,
                   binary_ridge::Real = 2.0,
                   _fitter = fit_gllvm, kwargs...)
    criterion in (:aic, :bic, :bic_sites) ||
        throw(ArgumentError("criterion must be :aic, :bic or :bic_sites; got :$criterion"))
    mask = get(kwargs, :mask, nothing)
    observed(i, j) = (mask === nothing || mask[i, j]) &&
                     !(Y[i, j] isa Missing) && !(Y[i, j] isa AbstractFloat && isnan(Y[i, j]))
    nsites = count(j -> any(i -> observed(i, j), axes(Y, 1)), axes(Y, 2))
    Kmax >= 1 || throw(ArgumentError("Kmax must be ≥ 1; got $Kmax"))

    Ks       = Int[]
    nps      = Int[]
    lls      = Float64[]
    aics     = Float64[]
    bics     = Float64[]
    bicns    = Float64[]
    fits     = Any[]
    attempts = NamedTuple{(:K, :status, :loglik, :message),Tuple{Int,Symbol,Float64,String}}[]

    # D-293: single-trial Binomial sweeps carry a loading ridge unless the caller
    # already picked their own `loading_ridge`, or `binary_ridge = Inf` (off).
    # Multi-trial Binomial (an `N` whose entries are not all one) is unaffected.
    # Only the Laplace `fit_binomial_gllvm` route takes `loading_ridge`; AGHQ refuses
    # it and the row-effect, grouping, phylo and dispersion routes have no such
    # keyword. On those routes the sweep runs unpenalised and each attempt says so.
    Nk = get(kwargs, :N, nothing)
    single_trial = Nk === nothing || all(isone, Nk)
    ridge_wanted = family isa Binomial && single_trial &&
                   !haskey(kwargs, :loading_ridge) && isfinite(binary_ridge)
    other_route = [string(k) for k in (:row_eff, :grouping, :phylo, :disp_group, :pervar)
                   if !(get(kwargs, k, nothing) in (nothing, :none, false))]
    _aghq_request(get(kwargs, :aghq, false)) === :off || push!(other_route, "aghq")
    ridge_active = ridge_wanted && isempty(other_route)
    ridge_kwarg = ridge_active ? (loading_ridge = binary_ridge,) : NamedTuple()
    ridge_note = ridge_active ? "loading_ridge=$binary_ridge" :
                 ridge_wanted ? "binary ridge not applied (not supported with " *
                                join(other_route, ", ") * ")" : ""
    withridge(msg) = isempty(ridge_note) ? msg :
                     (isempty(msg) ? ridge_note : msg * "; " * ridge_note)

    tryfit(k; init...) = try
        (_fitter(Y; family = family, K = k, kwargs..., ridge_kwarg..., init...), "")
    catch e
        e isa InterruptException && rethrow()
        # An ArgumentError at K = 1 is a misconfiguration, not a hard K: surface it.
        (k == 1 && e isa ArgumentError && isempty(init)) && rethrow()
        (nothing, sprint(showerror, e))
    end

    # Bar: best logLik of any converged, non-runaway fit at a smaller K. Accepted fits
    # rise monotonically and every other converged fit is either below the bar
    # (non-monotone) or excluded (runaway), so this equals the last accepted fit.
    llbar = -Inf
    for k in 1:Kmax
        fit, msg = tryfit(k)
        prev = isempty(fits) ? nothing : fits[end]
        llprev = llbar
        tolk(ll) = max(tol, 1e-6 * abs(ll))
        runaway(f) = _lv_runaway(f, family; max_latent_sd = max_latent_sd, ratio_max = ratio_max)
        acceptable(f) = f !== nothing && (_lv_converged(f) || !require_converged) && isempty(runaway(f)) &&
                        _lv_bar_value(f) >= llprev - tolk(llprev)
        status = :ok
        if !acceptable(fit) && fit !== nothing && warm_start && prev !== nothing
            init = _lv_warm_start(prev, k)
            if init !== nothing
                refit, _ = tryfit(k; init...)
                acceptable(refit) && ((fit, status) = (refit, :warm_start))
            end
        end
        if fit === nothing
            push!(attempts, (K = k, status = :failed, loglik = NaN, message = withridge(msg)))
            continue
        elseif require_converged && !_lv_converged(fit)
            push!(attempts, (K = k, status = :unconverged, loglik = _loglik(fit), message = withridge("")))
            continue
        elseif !isempty(runaway(fit))
            push!(attempts, (K = k, status = :runaway, loglik = _loglik(fit), message = withridge(runaway(fit))))
            continue
        elseif _lv_bar_value(fit) < llprev - tolk(llprev)
            push!(attempts, (K = k, status = :nonmonotone, loglik = _loglik(fit),
                             message = withridge(ridge_active ?
                                 "penalised objective ℓ − ½Σλ²/τ² = $(_lv_bar_value(fit)) below the bar $(llprev) set by a smaller K" :
                                 "logLik below a converged fit at a smaller K ($(llprev))")))
            continue
        end
        llbar = max(llbar, _lv_bar_value(fit))
        push!(attempts, (K = k, status = status, loglik = _loglik(fit),
                         message = withridge(_lv_converged(fit) ? "" : "optimiser did not report convergence; kept (not runaway, logLik non-decreasing)")))
        push!(Ks, k)
        push!(nps, _nparams(fit))
        push!(lls, _loglik(fit))
        push!(aics, aic(fit))
        push!(bics, bic(fit, Y; mask = mask))
        push!(bicns, bic(fit, nsites))
        push!(fits, fit)
    end

    isempty(Ks) && error("select_lv: no K in 1:$Kmax was accepted; attempts: " *
                         join(("K=$(a.K) $(a.status) $(a.message)" for a in attempts), "; "))

    crit = criterion === :aic ? aics : criterion === :bic ? bics : bicns
    ibest = argmin(crit)

    return LVSelection(Ks, nps, lls, aics, bics, bicns, Ks[ibest], fits[ibest], attempts)
end

# Tidy table display, best row marked with '*'.
function Base.show(io::IO, ::MIME"text/plain", sel::LVSelection)
    println(io, "GLLVModels latent-dimension selection (best K = ", sel.best_k, ")")
    println(io, "      K   nparams        logLik           AIC           BIC")
    for i in eachindex(sel.K)
        mark = sel.K[i] == sel.best_k ? "*" : " "
        println(io, mark, " ",
                lpad(string(sel.K[i]), 5), "   ",
                lpad(string(sel.nparams[i]), 7), "   ",
                lpad(string(round(sel.loglik[i]; sigdigits = 7)), 11), "   ",
                lpad(string(round(sel.aic[i]; sigdigits = 7)), 11), "   ",
                lpad(string(round(sel.bic[i]; sigdigits = 7)), 11))
    end
    for a in sel.attempts
        a.status in (:ok, :warm_start) && continue
        println(io, "  K = ", a.K, " not used: ", a.status, isempty(a.message) ? "" : " — " * a.message)
    end
end

Base.show(io::IO, sel::LVSelection) =
    print(io, "LVSelection(K=", sel.K, ", best_k=", sel.best_k, ")")
