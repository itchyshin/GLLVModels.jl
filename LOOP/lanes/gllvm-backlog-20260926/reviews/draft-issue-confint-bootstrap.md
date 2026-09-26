DRAFT ISSUE, NOT FILED. For maintainer review before filing.

---

Title: `confint_family.jl`'s bootstrap and profile-refit helpers accept a non-converged replicate as valid

## Summary

`_family_bootstrap` and `_family_profile_refit` in `src/confint_family.jl` both decide whether a
refit "worked" by checking `isfinite(...)` on the returned objective or parameter vector, and
never read the convergence flag that `Optim.optimize` (or the family's own fitter) already
computed. A refit that stopped early, took a zero-length step, or landed on the fitters' `1e12`
failure sentinel is finite, so both helpers count it as a success.

This sits downstream of every per-family point-fit fix (#479, #480, #486/#494, and whatever
lands from the tracking issue for the remaining families): even a family whose point estimate
now correctly reports `converged=false` can still produce a confidence interval built from
silently-failed bootstrap replicates or profile refits, because neither helper checks the flag.

## Evidence, current `origin/main` @ `b90641c97` (2026-09-26)

### `_family_profile_refit`, `src/confint_family.jl:2798-2818`

```julia
function _family_profile_refit(ad::_FamilyCI, i::Integer, c::Real, θ_red_warm::AbstractVector;
                               g_tol::Real = 1e-4,
                               iterations::Integer = 200)
    ...
    res = try
        Optim.optimize(nll_red, collect(Float64, θ_red_warm),
                       Optim.LBFGS(), Optim.Options(g_tol = g_tol, iterations = iterations);
                       autodiff = :finite)
    catch
        return (NaN, false, collect(Float64, θ_red_warm))
    end
    nmin = Optim.minimum(res)
    isfinite(nmin) || return (NaN, false, collect(Float64, θ_red_warm))
    return (-nmin, true, Optim.minimizer(res))
end
```

`ok = true` is returned whenever `nmin` is finite. `Optim.converged(res)` is never called. If the
inner `ad.nll` returns the `1e12` sentinel used elsewhere in this package to signal "the mode
search failed," `1e12` is finite, so this function reports `ok = true` and a profile deviance
built from a failed inner fit.

### `_family_bootstrap`, `src/confint_family.jl:2893-2935`

```julia
function _family_bootstrap(ad::_FamilyCI, sel::Vector{Int}, level::Real,
                           n_boot::Integer, seed::Integer, parallel::Bool; retain_replicates::Bool=false)
    ...
    work = function (b)
        rng = MersenneTwister(seed + b)
        θb = try
            ad.refit(ad.simulate(rng))
        catch
            nothing
        end
        if θb !== nothing && length(θb) == m && all(isfinite, θb)
            @inbounds reps[b, :] .= θb
            ok[b] = true
        end
        return nothing
    end
    ...
    result=(term = term, estimate = est, lower = lo, upper = hi,
            n_converged = count(ok), method = :bootstrap)
    ...
```

`ok[b] = true` requires only that `ad.refit` returned a finite parameter vector of the right
length. `ad.refit`'s own `.converged` field (whatever the underlying fitter computed, including
any of the gradient-aware verdict fixes from #480/#483/#502) is never inspected. The result's
`n_converged` field is therefore a count of "finite," not "converged," and is liable to
overstate how many bootstrap replicates actually succeeded.

## Why this matters beyond the point-fit bugs

A user who fits a model, sees `converged = true` (now correctly, after the per-family fixes
land), and then calls `confint(...; method = :profile)` or `method = :bootstrap` for the
uncertainty on that same fit can still get an interval built partly or wholly from refits that
silently failed. This is a distinct risk from the point-fit bugs because confidence intervals
are usually treated as the already-checked uncertainty on a value the user has already accepted;
a silently wrong CI is harder to notice than a silently wrong point estimate, since there is no
obvious sanity check (like a wildly implausible logLik) to catch it by eye.

## Minimal reproduction (not yet run; sketch only, feasible but not attempted here given the time
budget for this triage pass)

1. Take any family+dataset where the per-site mode search still diverges at some bootstrap
   resample (any of the still-live families in the sibling tracking issue would do; NB1 grouped
   is the best-confirmed candidate since it is default-route reachable).
2. Call `confint(fit, Y; method = :bootstrap, n_boot = 200, seed = <fixed>)`.
3. Instrument `_family_bootstrap` (or copy it locally) to also record `ad.refit`'s own
   `.converged` flag per replicate, alongside the existing `isfinite` check.
4. Compare `count(ok)` (current behaviour) against a count that additionally requires
   `.converged == true`. A gap between the two confirms the defect on a live dataset; the
   `n_converged` field in the public result would need to change to reflect the stricter count,
   and any replicate that fails the stricter check should probably be excluded from the
   quantile computation the same way it is currently included.

The same sketch applies to `_family_profile_refit`, substituting `Optim.converged(res)` for the
recorded flag at step 3, and checking whether `nmin` equals or exceeds the `1e12` sentinel as an
additional guard, since a legitimate non-sentinel finite-but-unconverged result is possible too.

## Suggested fix shape

- `_family_profile_refit`: return `ok = false` unless `Optim.converged(res)` is true, in addition
  to the existing `isfinite(nmin)` check; also treat `nmin >= 1e12` (or whatever the current
  sentinel value is) as a failure regardless of the `Optim.converged` flag, since the sentinel
  can itself be a converged-looking finite value from the inner objective.
- `_family_bootstrap`: thread `ad.refit`'s convergence flag through to `ok[b]`, not just
  `isfinite`. This requires `ad.refit` to expose that flag in its return value if it does not
  already (check the `_FamilyCI` struct and callers before assuming this is a one-line change).
- Both changes will likely lower `n_converged` / the number of usable profile points on any
  dataset that currently silently accepts failed refits; that is the intended, honest behaviour,
  analogous to the `converged` flips documented in #479/#480/#486/#494's after-task reports.

## Not in scope for this issue

- The per-family point-fit mode-search defects themselves (Class A): tracked by #479 (fixed),
  #480 (fixed), #486/#494 (fixed), PR #500 (open), and the sibling tracking issue for remaining
  families.
- The `_fit_verdict` Class B gradient-criterion gap for point fits: tracked by PR #502 (draft,
  Fixes #485). That PR does not touch `confint_family.jl`.
