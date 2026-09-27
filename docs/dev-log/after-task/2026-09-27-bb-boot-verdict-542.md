# After-task: BetaBinomial bootstrap refits report their own verdict (#542, 2026-09-27)

Lane: Claude, `LANE_ID=claude:GLLVM.jl:6081`, branch `claude/bb-boot-verdict-542` from
`origin/main` @ `97e11be04`, worktree `.worktrees/bb-boot-verdict-542`.

## 1. Goal

#542 (part of #504): `confint(...; method = :bootstrap)` for the three beta-binomial fit types
counted a replicate whose refit reported `converged = false` as a good draw, because the refit
closures returned a bare vector. Apply the #516 Poisson contract. Maintainer choice (2026-09-27):
option 1, exclude such replicates and report `n_converged`.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `BetaBinomialGroupedFit`,
  `BetaBinomialGroupedCovFit` and `BetaBinomialFit` return
  `(θ = <same vector>, converged = fb.converged, loglik = fb.loglik)`. `_bootstrap_refit_ok` and
  `_family_bootstrap` are unchanged; they already honour this contract (#508).
- `test/test_confint_bootstrap_verdict_betabinomial.jl` (new, registered after the Poisson file).
- `test/fixtures/beta_binomial_boot_boundary_542.toml` (new): two literal Binomial datasets
  (seeds 3 and 6, drawn on Julia 1.10.12) with sha256 guards.
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Option 1 (maintainer's call).** Options 2 (keep boundary replicates for `φ` rows) and 3
  (report `Inf` upper bounds) were listed on #542 and not taken.
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
- GREEN on the branch: 57/57 on Julia 1.10.12 and 57/57 on Julia 1.13.0.
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
    are all dropped once #541 merges.

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

- Option 1 can make an upper bound for `φ` look tighter than the data support; the only signal is
  `n_converged` below `n_boot`. The measured rate makes this concrete: on per-species grouped
  fits about a quarter of replicates have some species at the boundary. The count is over the
  largest `φ` per replicate, not per species; for any species that reaches the boundary in more
  than 2.5% of replicates, the honest upper bound is unbounded and option 1 reports a finite one.
  Options 2 and 3 on #542 remain open for the maintainer.
- Grouped boundary replicates are only rejected once #541 merges; until then the grouped
  fitters report `converged = true` at the boundary.
- Full suite not run; CI will run it.

## 11. Team Learning

Before trusting an equality between two bootstrap results, check the bounds are finite:
`_family_bootstrap` returns `NaN` below 10 usable replicates, and `isequal(NaN, NaN)` is true.

## 12. Cross-Product Coverage

Covered: all three beta-binomial fit types, Julia 1.10.12 and 1.13.0, aarch64 macOS, sentinel and
φ-boundary failures. Not covered: Linux and Windows (CI only), `parallel = true` bootstraps, masked
data, profile intervals (already gated by #508).
