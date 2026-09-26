# Latent-dimension selection — a practical workflow tool.
#
# Fits the same GLLVM at a sweep of latent dimensions K = 1:Kmax via the
# unified `fit_gllvm` dispatcher, records the information criteria (using the
# same `_nparams`/`_loglik` path that `aic`/`bic` use), and picks the K that
# minimises the chosen criterion among the fits that pass the guard below.
#
# Guard. A K model nests the (K−1) model (a zero loading column), so a correct
# maximum can never have a lower log-likelihood. A fit that throws, reports
# non-convergence, or falls below the last accepted K by more than `tol` is not a
# valid candidate. Before rejecting a non-monotone or unconverged fit, the sweep
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
- `loglik::Vector{Float64}` — maximised marginal log-likelihood per fit.
- `aic::Vector{Float64}` — Akaike information criterion per fit.
- `bic::Vector{Float64}` — Bayesian information criterion per fit.
- `best_k::Int` — the accepted `K` minimising the chosen criterion.
- `best::Any` — the chosen fitted model (the one at `best_k`).
- `attempts::Vector` — one named tuple `(K, status, loglik, message)` per
  attempted `K`. `status` is `:ok`, `:warm_start` (accepted after a refit from
  the previous solution), `:nonmonotone` (log-likelihood below the last accepted
  `K`), `:unconverged`, or `:failed` (the fit threw; `message` says why).
"""
struct LVSelection
    K::Vector{Int}
    nparams::Vector{Int}
    loglik::Vector{Float64}
    aic::Vector{Float64}
    bic::Vector{Float64}
    best_k::Int
    best::Any
    attempts::Vector{NamedTuple{(:K, :status, :loglik, :message),Tuple{Int,Symbol,Float64,String}}}
end

# Warm-start keywords for a K-dimensional refit from an accepted (K−1) fit, or
# `nothing` when the fit does not expose `β`/`Λ`. The new column keeps the top
# K×K block lower-triangular (zeros above the diagonal, positive diagonal).
function _lv_warm_start(prev, K::Integer)
    (hasproperty(prev, :β) && hasproperty(prev, :Λ)) || return nothing
    Λp = prev.Λ
    p = size(Λp, 1)
    (size(Λp, 2) == K - 1 && K <= p) || return nothing
    col = zeros(p)
    col[K:end] .= 0.1
    return (β_init = copy(prev.β), Λ_init = hcat(Λp, col))
end

_lv_converged(fit) = hasproperty(fit, :converged) ? Bool(fit.converged) : true

"""
    select_lv(Y; family = Normal(), Kmax = 3, criterion = :bic,
              warm_start = true, tol = 1e-3, kwargs...) -> LVSelection

Latent-dimension selection: fit `fit_gllvm(Y; family, K = k, kwargs...)` for
`k in 1:Kmax` and pick the `K` minimising `criterion` (`:aic` or `:bic`) among
the fits that pass a guard.

The guard rejects a fit that throws, reports non-convergence, or has a
log-likelihood more than `tol` below the last accepted `K` (a `K` model nests the
`K − 1` model, so its maximum cannot be lower). With `warm_start = true`, a
non-monotone or unconverged fit is first refitted once from the previous
solution plus a small new loading column, for families whose fitter accepts
`β_init`/`Λ_init`; the better converged fit is kept. Rejected fits stay in
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
sel = select_lv(Y; family = Poisson(), Kmax = 3)   # pick K by BIC
sel.best_k          # selected latent dimension
sel.best            # the fitted model at that K
sel.attempts        # every K tried, with status
```
"""
function select_lv(Y::AbstractMatrix; family = Normal(), Kmax::Integer = 3,
                   criterion::Symbol = :bic, warm_start::Bool = true,
                   tol::Real = 1e-3, _fitter = fit_gllvm, kwargs...)
    criterion in (:aic, :bic) ||
        throw(ArgumentError("criterion must be :aic or :bic; got :$criterion"))
    Kmax >= 1 || throw(ArgumentError("Kmax must be ≥ 1; got $Kmax"))

    Ks       = Int[]
    nps      = Int[]
    lls      = Float64[]
    aics     = Float64[]
    bics     = Float64[]
    fits     = Any[]
    attempts = NamedTuple{(:K, :status, :loglik, :message),Tuple{Int,Symbol,Float64,String}}[]

    tryfit(k; init...) = try
        (_fitter(Y; family = family, K = k, kwargs..., init...), "")
    catch e
        e isa InterruptException && rethrow()
        (nothing, sprint(showerror, e))
    end

    for k in 1:Kmax
        fit, msg = tryfit(k)
        prev = isempty(fits) ? nothing : fits[end]
        llprev = isempty(lls) ? -Inf : lls[end]
        ok(f) = f !== nothing && _lv_converged(f) && _loglik(f) >= llprev - tol
        status = :ok
        if !ok(fit) && fit !== nothing && warm_start && prev !== nothing
            init = _lv_warm_start(prev, k)
            if init !== nothing
                refit, _ = tryfit(k; init...)
                if refit !== nothing && _lv_converged(refit) &&
                   (!_lv_converged(fit) || _loglik(refit) > _loglik(fit))
                    fit, status = refit, :warm_start
                end
            end
        end
        if fit === nothing
            push!(attempts, (K = k, status = :failed, loglik = NaN, message = msg))
            continue
        elseif !_lv_converged(fit)
            push!(attempts, (K = k, status = :unconverged, loglik = _loglik(fit), message = ""))
            continue
        elseif _loglik(fit) < llprev - tol
            push!(attempts, (K = k, status = :nonmonotone, loglik = _loglik(fit),
                             message = "logLik below accepted K = $(Ks[end]) ($(llprev))"))
            continue
        end
        push!(attempts, (K = k, status = status, loglik = _loglik(fit), message = ""))
        push!(Ks, k)
        push!(nps, _nparams(fit))
        push!(lls, _loglik(fit))
        push!(aics, aic(fit))
        push!(bics, bic(fit, Y))
        push!(fits, fit)
    end

    isempty(Ks) && error("select_lv: no K in 1:$Kmax was accepted; attempts: " *
                         join(("K=$(a.K) $(a.status) $(a.message)" for a in attempts), "; "))

    crit = criterion === :aic ? aics : bics
    ibest = argmin(crit)

    return LVSelection(Ks, nps, lls, aics, bics, Ks[ibest], fits[ibest], attempts)
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
