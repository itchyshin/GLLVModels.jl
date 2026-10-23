# After-task: Ordinal bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:ordinal-boot-verdict-504:6081`, branch `claude/ordinal-boot-verdict-504`
from `origin/main` @ `880cad4c7`, worktree `.worktrees/ordinal-boot-verdict-504`. One slice of the #504
family migration, following the Poisson (#516) and Gamma patterns.

## 1. Goal

Move the three Ordinal bootstrap refit closures onto the #508/#516 contract, so a non-converged refit is
excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `OrdinalFit`, `OrdinalPerTraitFit`
  and `OrdinalPerTraitCovFit` return `(θ, converged = fb.converged, loglik = fb.loglik)`. θ is the same
  vector as before; the `fb.C == C || return nothing` drop is untouched.
- `test/test_confint_bootstrap_verdict_ordinal.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Stub failed refit instead of a data-driven bad draw.** A probe found no draw that makes an ordinal
  fitter fail softly (`converged = false`, `loglik = -Inf`, no throw) when bounds are checked. Category
  probabilities are clamped at 1e-12, so any draw coded `1:C` gives a finite likelihood. A level of 0
  or -1 throws `BoundsError` under `--check-bounds=yes` (the `Pkg.test` setting) on all three routes.
  A level above C changes the category count, which the closure already drops. The rejection and
  end-to-end testsets therefore use a stub refit returning `(θ = finite, converged = false,
  loglik = -Inf)`, labelled as a stub in the file header and at its definition.
- **No `upper_boundary` flag.** Loadings, intercepts and cutpoints have no flat-likelihood limit of the
  kind the #542 field describes.
- **Seeded simulation, no fixture**, as #516: every assertion is a relation.
- **Endpoint equality on the shared-cutpoint route**, the fastest one (about 0.3 s per refit).

## 4. Files Touched

- `src/confint_family.jl` (modified, three closures)
- `test/test_confint_bootstrap_verdict_ordinal.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-ordinal-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file, full suite not run.

- Probe (Julia 1.10.12, p = 4, n = 150, K = 1, C = 4, logit): healthy data converges on all three
  routes (0.23 s, 0.57 s and 0.59 s per fit). Four degenerate draws (empty middle category, only the
  extreme categories, one trait nearly constant, one trait with each level once) all converge. Levels
  0 and -1: `BoundsError` on every route with bounds checking; with bounds checks elided, the per-trait
  routes return `converged = false, loglik = -Inf` (see section 10). Level 7: fits converge with C = 7.
- RED, new file on `origin/main` 880cad4c7, Julia 1.10.12: 22 pass, 5 fail, 12 error of 39. The failures
  are the three adapter-parity testsets and the endpoint testset; the stub testsets pass on main
  because they exercise `_bootstrap_refit_ok` and `_family_bootstrap`, not the closures.
- GREEN: 39/39 on Julia 1.10.12 (file 21.4 s) and 39/39 on Julia 1.13.0 (file 27.2 s).
- Neighbours on 1.10.12, each file alone: `test_bridge_missing_mask.jl` 92/92, `test_bridge_x.jl`
  200/200, `test_confint_family.jl` 341/341, `test_diagnostics.jl` 65/65, `test_extractors.jl` 92/92,
  `test_lv_ci.jl` 196/196. No failures and no broken markers.

## 6. Tests of the Tests

- The adapter-parity testset fails on main (the closure returns a bare vector) and compares the migrated
  closure field by field with an independent direct refit.
- The stub testset asserts the old bare-vector contract accepts the stub θ and the new one rejects the
  NamedTuple. It does not discriminate main from the branch, which section 5 records.
- The endpoint-equality testset uses `n_boot = 10` and asserts finite bounds before comparing.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- The other #504 slices (Gamma, Tweedie and the rest) each add a CHANGELOG bullet at the same place and
  an include after the Poisson one in `test/runtests.jl`, so textual merge conflicts are expected when
  several land; each resolves by keeping both lines.

## 9. What Did Not Go Smoothly

- The first probe run used `timeout`, which macOS does not ship; rerun without it.
- The Gamma recipe (a single bad cell) does not transfer: an ordinal zero cell is an index error, not a
  likelihood of -Inf, so the plan changed to the stub route.

## 10. Known Residuals

- `_pack_initial_ordinal_pertrait` indexes `counts[Int(Y[t, i])]` inside `@inbounds`. With bounds checks
  elided (the default outside `Pkg.test`), a level of 0 or below is an out-of-bounds write there, and
  the per-trait fitters went on to return a sentinel fit rather than an error (inference from the probe;
  the write itself was not observed). The shared-cutpoint fitter threw `BoundsError` in both modes. The
  fitters do not validate that levels are at least 1. Not changed here.
- A real soft failure of an ordinal refit is not exercised end to end; only the stub is.
- Full suite not run; CI will run it.

## 11. Team Learning

A bad-draw recipe is family-specific: check first whether an invalid value reaches the likelihood (a
sentinel) or an index (an exception or, under `@inbounds`, undefined behaviour).

## 12. Cross-Product Coverage

Covered: all three Ordinal fit types with the logit link, Julia 1.10.12 and 1.13.0, aarch64 macOS. Not
covered: probit link, `X_lv` shared-cutpoint fits, masked data, Linux and Windows (CI).
