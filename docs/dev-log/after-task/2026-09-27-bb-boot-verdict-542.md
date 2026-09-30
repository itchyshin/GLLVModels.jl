# After-task: BetaBinomial bootstrap refits report their own verdict (#542, 2026-09-27)

Lane: Claude, `LANE_ID=claude:GLLVM.jl:6081`, branch `claude/bb-boot-verdict-542` from
`origin/main` @ `97e11be04`, worktree `.worktrees/bb-boot-verdict-542`.

## 1. Goal

#542 (part of #504): `confint(...; method = :bootstrap)` for the three beta-binomial fit types
counted a replicate whose refit reported `converged = false` as a good draw, because the refit
closures returned a bare vector. Apply the #516 Poisson contract. Maintainer choice (2026-09-27):
first option 1 (exclude such replicates, report `n_converged`), then, after the boundary-rate
measurement below, option 3: also report `Inf` as the upper bound of a `φ` whose boundary share
of usable replicates exceeds the upper tail.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `BetaBinomialGroupedFit`,
  `BetaBinomialGroupedCovFit` and `BetaBinomialFit` return
  `(θ = <same vector>, converged = fb.converged, loglik = fb.loglik)`. `_bootstrap_refit_ok` and
  `_family_bootstrap` already honour this contract (#508). The closures also return
  `upper_boundary` (`_bb_phi_upper_boundary`): a Bool vector flagging each `log φ` entry at or past
  `_BB_PHI_STABLE`.
- `src/confint_family.jl`, shared: new `_bootstrap_upper_boundary(raw, m)` reads that optional
  field. In `_family_bootstrap` a flagged replicate is excluded from every quantile, and for each
  parameter, when the flagged share of usable replicates (converged plus boundary) exceeds
  `(1 - level)/2`, the upper bound is `Inf`. Result fields are unchanged; adapters that never set
  the field (every other family) get identical results.
- `test/test_confint_bootstrap_verdict_betabinomial.jl` (new, registered after the Poisson file).
- `test/fixtures/beta_binomial_boot_boundary_542.toml` (new): two literal Binomial datasets
  (seeds 3 and 6, drawn on Julia 1.10.12) with sha256 guards.
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Option 3 (maintainer's call, after option 1 was first implemented).** Option 2 (keep boundary
  replicates in the `φ` quantiles) was not taken.
- **Per parameter, not per replicate.** The `Inf` rule counts, for each `φ`, the replicates where
  that `φ` hit the boundary, so one species at the boundary does not make every species' bound
  infinite.
- **Denominator = usable replicates.** A failed refit (sentinel) says nothing about `φ`, so it is
  left out of the share; a synthetic test pins this.
- **Opt-in field in the shared function**, rather than beta-binomial-specific logic in
  `_family_bootstrap`: any family with a numerical upper boundary can use it later, and nothing
  changes for families that do not set it.
- **Lower bound from the interior draws.** Removing the boundary mass lowers every quantile, so the
  reported lower bound is, if anything, conservative (too low), not too high.
- **Test the φ-boundary case on the ungrouped route.** On main only the ungrouped fitter reports
  `converged = false` at the boundary; the grouped routes do so once #541 merges. The grouped
  closures are covered through the failure-sentinel draw and the parity test, and the boundary
  rate for grouped refits was measured on a local merge with #541 (section 5).
- **New literal fixture, not seeds.** Seeded draws differ across Julia versions (#522 found this);
  both datasets reach the boundary on 1.10.12 and 1.13.0.

## 4. Files Touched

- `src/confint_family.jl` (modified, three closures)
- `test/test_confint_bootstrap_verdict_betabinomial.jl` (new)
- `test/fixtures/beta_binomial_boot_boundary_542.toml` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-bb-boot-verdict-542.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file includes, full suite not run.

- RED, new file on `origin/main` 97e11be04 source, Julia 1.10.12: 17 pass, 17 fail, 23 error of 57.
  Errors: no `.converged`/`.loglik`/`.θ` on a bare `Array`. Failures include the end-to-end count
  (`0 < 6 < 6`: main counts the 3 failed replicates as converged) and `_bootstrap_refit_ok`
  accepting both boundary replicates.
- GREEN on the branch: 57/57 on Julia 1.10.12 and 57/57 on Julia 1.13.0 (option 1 commit).
- Option 3 commit, same file (now 76 assertions): RED on origin/main 28 pass, 19 fail, 29 error;
  GREEN 76/76 on Julia 1.10.12 and 1.13.0. On main the synthetic 2-of-40 case gives an upper bound
  of 8.91 where option 3 requires `Inf`.
- Shared-function neighbours after option 3, Julia 1.10.12, each file in its own process:
  `test_confint_bootstrap_verdict_poisson.jl` 24/24, `test_confint_bootstrap.jl` 9/9,
  `test_bridge_ci.jl` 65/65, `test_derived_ci_surfaces.jl` 92/92, `test_aghq_public_binomial.jl`
  108/108, `test_aghq_public_gaussian.jl` 69/69, `test_confint_family.jl` 341/341 (708 total).
- Live option-3 result, `confint(fit, Y; method = :bootstrap, n_boot = 50, seed = 1, N = N)` on a
  per-species grouped fit of healthy_seed_9001 (Julia 1.10.12, this branch without #541):
  `n_converged` 38 of 50; `φ[1]` 17.18 [8.825, Inf], `φ[6]` 127.3 [14.31, Inf]; `φ[2]`, `φ[3]`,
  `φ[4]` upper bounds 1.9e5, 1.2e5, 2.6e5 (interior draws just under 1e6); `φ[5]` 18.05 [10.85, 61.22].
- Neighbour: the beta-binomial Wald + bootstrap testset of `test/test_confint_family.jl`
  (lines 453-494, extracted and run alone) passes 15/15 with `n_converged` 8 of 8 on main and
  branch, Julia 1.10.12 and 1.13.0.
- Bootstrap boundary rate (#542's "first thing to check"), Julia 1.10.12, 50 replicates per
  dataset at `seed = 1`, on a local merge of this branch with #541:
  - ungrouped (`BetaBinomialFit`), healthy_seed_9001 to 9005: 250 of 250 replicates converged,
    none at the boundary.
  - per-species grouped (`BetaBinomialGroupedFit`, `group = 1:p`, the route `fit_gllvm` and the
    bridge use), healthy_seed_9001 / 9004 / 9005: 12, 9 and 14 of 50 replicates reached
    `φ >= 1e6` (35 of 150, 23%); every other replicate converged, none threw. Under option 1 these
    are all dropped once #541 merges. (With the option-3 `upper_boundary` flag they are dropped and
    counted even before #541, because the flag reads `φ` directly.)

## 6. Tests of the Tests

- The boundary testset asserts both halves: the old bare-vector contract accepts the replicate
  (`_bootstrap_refit_ok(raw.θ, m)` is true) and the new one rejects it. It cannot pass if the
  fixture stops reaching the boundary, because it first asserts `φ >= 1e6 && !converged`.
- The first version of the endpoint-equality testset used `n_boot = 6`. `_family_bootstrap`
  returns `NaN` bounds below 10 usable replicates, so it compared `NaN` with `NaN` and passed
  vacuously. Fixed to `n_boot = 10` with an explicit `all(isfinite, ...)` assertion before the
  comparison.

## 7a. Issue Ledger

- #542: addressed by this branch. The PR does not use a closing keyword, because the grouped
  boundary case also needs #541; close #542 by hand once both are merged.
- #504 stays open: 40 other fit types still use the bare-vector adapter.

## 8. Consistency Audit

- No other ref on the remote or locally already migrates the beta-binomial closures (scanned 829 refs).
- Open PRs #518 and #363 touch `src/confint_family.jl` but not the beta-binomial closures.
- The `BetaBinomialFit` refit does not pass `mask` (the grouped ones do). Pre-existing, unchanged
  here; noted in the PR.
- `test_confint_family.jl:486` only checks bounds when finite, so with `n_boot = 8` it never checks
  a bootstrap bound at all. Pre-existing, unchanged; noted in the PR.

## 9. What Did Not Go Smoothly

- My fixture generator was passed its own path as `ARGS[1]` and wrote the TOML over itself; the
  content was correct and was moved into place, then hash-checked.
- Two scratch worktrees needed `Pkg.instantiate()` before they would load.
- A shell loop over refs failed silently under zsh (`$r:src` is a history modifier) until braced.

## 10. Known Residuals

- The 1e6 line is a tripwire. Replicates that stop with `φ` near 1e5 are interior draws, so a
  species can get a finite but huge upper bound (2.6e5 for `φ[4]` above). That is wide, not
  misleadingly tight, but it is not `Inf` either.
- The lower bound comes from the interior draws only (conservative, see 3a).
- Before #541 merges, the grouped fitters still report `converged = true` at the boundary for the
  point fit itself; the bootstrap is covered either way through `upper_boundary`.
- Full suite not run; CI will run it.

## 11. Team Learning

Before trusting an equality between two bootstrap results, check the bounds are finite:
`_family_bootstrap` returns `NaN` below 10 usable replicates, and `isequal(NaN, NaN)` is true.

## 12. Cross-Product Coverage

Covered: all three beta-binomial fit types, Julia 1.10.12 and 1.13.0, aarch64 macOS, sentinel and
φ-boundary failures. Not covered: Linux and Windows (CI only), `parallel = true` bootstraps, masked
data, profile intervals (already gated by #508).
