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

import Printf   # `show` formats the table columns

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
- `aicc::Vector{Float64}` — corrected AIC, `aic + 2k(k+1)/(n − k − 1)` with
  `k = nparams[i]` and `n` the same observed-cell count the BIC penalty uses
  (`nobs(fit, Y; mask)`, R's `p·n`); `NaN` (R's `NA`) when `n − k − 1 ≤ 0`. Reported
  only: it is not a selection criterion here.
- `converged::Vector{Bool}` — whether the optimiser reported convergence for each
  accepted fit (R's `conv`). A fit that did not converge is still accepted and
  stays in the table, but cannot win `best_k` (#759).
- `pd_hessian::Vector{Union{Missing,Bool}}` — R's `pdHess`: `true` when the Wald
  observed information at the optimum is usable (finite, factorisable, positive
  variances; the flag [`confint`](@ref) reports as `pd_hessian`), `false` when it
  is not, `missing` (R's `NA`, "not determined") when it was not computed or the
  fit type has no Wald route. Computed only with `select_lv(...; pd_hessian = true)`.
  A rank with `false` is still accepted and can still be chosen.
- `best_k::Int` — the accepted `K` minimising the chosen criterion.
- `best::Any` — the chosen fitted model (the one at `best_k`).
- `attempts::Vector` — one named tuple `(K, status, loglik, message)` per
  attempted `K`. `status` is `:ok`, `:warm_start` (accepted after a refit from
  the previous solution), `:nonmonotone` (log-likelihood below the last accepted
  `K`; under a loading ridge the penalised objective ℓ − ½Σλ²/τ² is compared,
  while `loglik` stays the unpenalised value), `:unconverged`, `:runaway` (inflated or separated loadings; `message`
  gives the statistic), or `:failed` (the fit threw; `message` says why).
- `criterion::Symbol` — the criterion the sweep minimised (`:bic_sites`, `:bic` or `:aic`).

Printing a result shows the table gllvmTMB's `print.gllvmTMB_select_lv` shows, with
the same labels in the same order: a marker (`*` on the chosen row), `d` (= `K`),
`npar` (= `nparams`), `logLik`, `AIC`, `BIC`, `AICc`, `conv`, `pdHess`. Numbers are
rounded to three decimals and flags print as `TRUE`/`FALSE`/`NA`, as R does. `BIC` is
the `log(p·n)` BIC (`bic`), R's BIC, whatever criterion chose the row.

Where Julia differs from R: R also leaves a rank whose Hessian is confirmed not
positive definite out of the selection (the row stays in the table); `select_lv`
still keeps such a rank selectable and only reports `pdHess`. A rank that did
not converge, or whose criterion is non-finite, stays in the table but cannot
win `best_k`.
"""
struct LVSelection
    K::Vector{Int}
    nparams::Vector{Int}
    loglik::Vector{Float64}
    aic::Vector{Float64}
    bic::Vector{Float64}
    bic_sites::Vector{Float64}
    aicc::Vector{Float64}
    converged::Vector{Bool}
    pd_hessian::Vector{Union{Missing,Bool}}
    best_k::Int
    best::Any
    attempts::Vector{NamedTuple{(:K, :status, :loglik, :message),Tuple{Int,Symbol,Float64,String}}}
    criterion::Symbol
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

# n behind AICc: the same observed-cell count the BIC penalty uses, `nobs(fit, Y; mask)`
# (R's p·n). Stand-in fits that are not `AnyGllvmFit` get the same count from (Y, mask).
_lv_nobs(fit::AnyGllvmFit, Y, mask) = StatsAPI.nobs(fit, Y; mask = mask)
_lv_nobs(fit, Y, mask) = mask !== nothing ? count(mask) :
                         Missing <: eltype(Y) ? count(!ismissing, Y) : length(Y)

# AICc = AIC + 2k(k+1)/(n − k − 1), R's `.select_lv_aicc`; NaN (R's NA) when n − k − 1 ≤ 0.
_lv_aicc(aic, k, n) = n - k - 1 > 0 ? aic + 2 * k * (k + 1) / (n - k - 1) : NaN

# pdHess: the Wald flag `confint(fit, Y; method = :wald).pd_hessian` (observed information at
# the optimum finite, factorisable, positive variances). `missing` (R's NA, "not determined")
# when the fit type has no Wald route or the Hessian cannot be evaluated; never `false` for
# that. One finite-difference/AD Hessian per call.
# `confint` rebuilds the marginal from (fit, Y, X, mask, N, Σ_phy) only. A fit made with an
# `offset` would get the Hessian of a different, offset-free objective (a spurious FALSE on a
# regular fit), so pdHess is `missing` there until `confint` honours the offset.
function _lv_pd_hessian(fit, Y, kw)
    get(kw, :offset, nothing) === nothing || return missing
    try
        common = (X = get(kw, :X, nothing), mask = get(kw, :mask, nothing))
        ci = fit isa GllvmFit ?
             confint(fit, Y; common..., Σ_phy = get(kw, :Σ_phy, nothing)) :
             confint(fit, Y; method = :wald, common..., N = get(kw, :N, nothing))
        return Bool(ci.pd_hessian)
    catch e
        e isa InterruptException && rethrow()
        @debug "select_lv: pdHess not determined" exception = (e, catch_backtrace())
        return missing
    end
end

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
              ratio_max = 25.0, binary_ridge = 2.0, pd_hessian = false,
              kwargs...) -> LVSelection

Latent-dimension selection: fit `fit_gllvm(Y; family, K = k, kwargs...)` for
`k in 1:Kmax` and pick the `K` minimising `criterion` among the fits that pass a
guard: `:bic_sites` (default; penalty `log(n)`, sites), `:bic` (penalty
`log(p·n)`, observed cells) or `:aic`. The default follows a recovery simulation
(17 687 datasets with known K): `:bic_sites` recovered the true K most often for
Gaussian and Poisson responses; `:bic` picked too few dimensions at small n.
For negative-binomial responses, re-measured on the corrected NB2 fitting code
(4,794 datasets over 24 cells), `:bic_sites` recovered the true K in 0.934 of
datasets on average (`:bic` 0.840); its misses, at small n with p = 10 and
K = 3, pick too few dimensions.

Binary (single-trial `Binomial`) data get a loading ridge during the sweep:
every fit in the sweep — including a
warm-start refit — is called with `loading_ridge = binary_ridge` (default
`2.0`), unless the caller already passed their own `loading_ridge` (which then
wins) or `binary_ridge = Inf` (which disables it, restoring today's unpenalised
behaviour). This is because most unpenalised Bernoulli fits beyond `K = 1` run
away (see the runaway guard below): in an R recovery experiment with 20 species
and `n = 120`, the ridge recovered the true `K = 2` in 8/10 datasets versus 4/10
without it. With weak loadings (0.8·N(0,1), the recovery grid's own binary cells,
1,200 datasets) it did not raise recovery (0.41 versus 0.44 without it), but it never
chose too many dimensions; there the ridge buys safety rather than accuracy, and
dimensions beyond `K = 1` were found mainly at n = 300. The information criteria are still computed on the UNPENALISED
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

Each accepted `K` also gets the extra columns gllvmTMB's `select_lv` prints: `aicc`
(`aic + 2k(k+1)/(n − k − 1)`, `n` the same `p·n` as `bic`, `NaN` when `n − k − 1 ≤ 0`),
`converged` (the optimiser's flag) and `pd_hessian`. `pd_hessian` is `missing`
(printed `NA`, "not determined") unless you pass `pd_hessian = true`, which evaluates
the Wald observed information once per accepted `K` through
`confint(fit, Y; method = :wald)` and records whether it is usable. A fit type without a
Wald route gets `missing`, never `false`.

The Hessian is not free, and its cost grows roughly with the square of the parameter
count, which is why it is off by default. Measured on one Mac, sweeping `K = 1:3`:
Gaussian data add 0.01 s at `p = 6`, `n = 150`, 0.9 s at `p = 20`, `n = 150` and 17 s at
`p = 30`, `n = 200` (the Gaussian fits themselves take under 0.15 s); Poisson data add
about 9 to 10 times the fitting time (+3.5 s at `p = 10`, `n = 100`; +36 s at `p = 20`,
`n = 150`).

`aicc` and `pd_hessian` do not change which `K` is chosen, and `criterion` has no `:aicc`.
A rank that did not converge, or whose criterion is non-finite, stays in the table
but cannot win `best_k`. Unlike R, a rank whose Hessian is not positive definite
remains eligible; read `pd_hessian` before trusting the choice.

The chosen `K` is itself an estimate: intervals and tests computed on
`sel.best` are conditional on it and do not include uncertainty about `K`.

`family` is a Distributions.jl marker (the [`fit_gllvm`](@ref) convention);
`kwargs...` pass through to the underlying fitter.

```julia
sel = select_lv(Y; family = Poisson(), Kmax = 3)   # pick K by BIC with log(n sites)
sel.best_k          # selected latent dimension
sel.best            # the fitted model at that K
sel.attempts        # every K tried, with status
sel                 # table: d npar logLik AIC BIC AICc conv pdHess, chosen row marked *
select_lv(Y; family = Poisson(), Kmax = 3, pd_hessian = true)   # also fill pdHess (slower)
```
"""
function select_lv(Y::AbstractMatrix; family = Normal(), Kmax::Integer = 3,
                   criterion::Symbol = :bic_sites, warm_start::Bool = true,
                   tol::Real = 1e-3, max_latent_sd::Real = 10.0,
                   ratio_max::Real = 25.0, require_converged::Bool = false,
                   binary_ridge::Real = 2.0, pd_hessian::Bool = false,
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
    aiccs    = Float64[]
    convs    = Bool[]
    pds      = Union{Missing,Bool}[]
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
        push!(aiccs, _lv_aicc(aics[end], nps[end], _lv_nobs(fit, Y, mask)))
        push!(convs, _lv_converged(fit))
        push!(pds, pd_hessian ? _lv_pd_hessian(fit, Y, kwargs) : missing)
        push!(fits, fit)
    end

    isempty(Ks) && error("select_lv: no K in 1:$Kmax was accepted; attempts: " *
                         join(("K=$(a.K) $(a.status) $(a.message)" for a in attempts), "; "))

    crit = criterion === :aic ? aics : criterion === :bic ? bics : bicns
    eligible = findall(i -> convs[i] && isfinite(crit[i]), eachindex(crit))
    isempty(eligible) && error("select_lv: no accepted K in 1:$Kmax has a finite $criterion " *
                               "among converged fits; attempts: " *
                               join(("K=$(a.K) $(a.status) $(a.message)" for a in attempts), "; "))
    ibest = eligible[argmin(crit[eligible])]

    return LVSelection(Ks, nps, lls, aics, bics, bicns, aiccs, convs, pds, Ks[ibest], fits[ibest],
                       attempts, criterion)
end

# R prints a numeric column with `print.data.frame`: `round(x, 3)`, then one common number of
# decimals per column, the most any element needs to show up to 7 significant digits
# (`format(digits = 7)`), so a column of large values loses decimals and trailing zeros are
# kept. NaN prints as NA. Matches `print.gllvmTMB_select_lv` (R/select-lv.R at gllvmTMB P1).
function _lv_format_column(xs::AbstractVector{<:Real})
    vals = [isfinite(x) ? round(x; digits = 3) + 0.0 : x for x in xs]
    rgt = 0
    for x in vals
        (isfinite(x) && !iszero(x)) || continue
        mantissa, expo = split(Printf.format(Printf.Format("%.6e"), abs(x)), 'e')
        nsig = max(length(rstrip(replace(mantissa, "." => ""), '0')), 1)
        rgt = max(rgt, nsig - 1 - parse(Int, expo))
    end
    fmt = Printf.Format("%." * string(rgt) * "f")
    return [isfinite(x) ? Printf.format(fmt, x) : "NA" for x in vals]
end

_lv_format_flag(x) = ismissing(x) ? "NA" : x ? "TRUE" : "FALSE"

# Same table as gllvmTMB's `print.gllvmTMB_select_lv`: marker, d, npar, logLik, AIC, BIC,
# AICc, conv, pdHess; the selected row marked with '*'. Columns right-aligned, one space apart
# (R's `print(df, row.names = FALSE)` layout).
function Base.show(io::IO, ::MIME"text/plain", sel::LVSelection)
    println(io, "GLLVModels latent-dimension selection (criterion = ", sel.criterion,
            ", best K = ", sel.best_k, ")")
    cols = [
        ("",       [k == sel.best_k ? "*" : " " for k in sel.K]),
        ("d",      string.(sel.K)),
        ("npar",   string.(sel.nparams)),
        ("logLik", _lv_format_column(sel.loglik)),
        ("AIC",    _lv_format_column(sel.aic)),
        ("BIC",    _lv_format_column(sel.bic)),
        ("AICc",   _lv_format_column(sel.aicc)),
        ("conv",   _lv_format_flag.(sel.converged)),
        ("pdHess", _lv_format_flag.(sel.pd_hessian)),
    ]
    widths = [max(length(h), maximum(length, cells; init = 0)) for (h, cells) in cols]
    println(io, join((" " * lpad(h, w) for ((h, _), w) in zip(cols, widths))))
    for i in eachindex(sel.K)
        println(io, join((" " * lpad(cells[i], w) for ((_, cells), w) in zip(cols, widths))))
    end
    for a in sel.attempts
        a.status in (:ok, :warm_start) && continue
        println(io, "  K = ", a.K, " not used: ", a.status, isempty(a.message) ? "" : " — " * a.message)
    end
    any(ismissing, sel.pd_hessian) &&
        println(io, "  pdHess = NA: not determined. Pass select_lv(...; pd_hessian = true) to compute it (one Hessian per K); it stays NA when the fit has no Wald route, uses a loading ridge, or carries an offset.")
end

Base.show(io::IO, sel::LVSelection) =
    print(io, "LVSelection(K=", sel.K, ", best_k=", sel.best_k, ")")
