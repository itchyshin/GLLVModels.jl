# After-task: zero-inflated bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:zi-boot-verdict-504:6081`, branch `claude/zi-boot-verdict-504` from
`origin/main` @ `1214e948e`, worktree `.worktrees/zi-boot-verdict-504`. One slice of the #504
per-family migration, following the Gamma and Ordinal slices.

## 1. Goal

Move the five zero-inflated bootstrap refit closures onto the #508/#516 contract, so a refit that ends
on the fitter's failure sentinel is excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `ZIPFit`, `ZIPCovFit`, `ZINBFit`,
  `ZINBCovFit` and `ZIBFit` return `(θ = ..., converged = fb.converged, loglik = fb.loglik)`. The θ
  expression in each is byte-identical to before.
- `test/test_confint_bootstrap_verdict_zi.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **A real failing draw on every route, no stub.** For ZIP and ZINB one cell holds 0.5: the count
  log-density calls `Int(y)`, the `InexactError` is caught inside each fitter's objective, and every
  evaluation returns the 1e12 sentinel. For ZIB one cell is `N + 1`, whose binomial log-density is
  `-Inf`. All five routes then report `converged = false`, `loglik = -Inf`, finite θ, with no throw,
  with and without `--check-bounds=yes`.
- **Rejected: a negative count as the bad draw.** It does not fail. `_tp_pieces` tests `y > 0`, so a
  negative count is scored as a zero and every route converges.
- **No `upper_boundary` flag,** per the slice scope. Whether the ZINB size `r` needs one is not decided
  here.
- **Endpoint equality on the ZIP route,** the fastest one (0.9 s per healthy fit), with `n_boot = 10`.

## 4. Files Touched

- `src/confint_family.jl` (modified, five closures)
- `test/test_confint_bootstrap_verdict_zi.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-zi-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, machine load about 65, per-file, full
suite not run.

- Probe (Julia 1.10.12, p = 4, n = 120, K = 1, the test's data). Healthy fits, all
  `converged = true`: ZIP 0.9 s (29 iterations), ZIP + X 1.1 s, ZINB 2.9 s, ZINB + X 3.3 s, ZIB 0.8 s.
  Half-count cell: `converged = false`, `loglik = -Inf`, finite θ on all five, 0.6 to 1.4 s. Count
  above `N` on ZIB: the same, 0.07 s. Negative count: converges on all five. Same verdicts under
  `--check-bounds=yes`.
- RED, new file on `origin/main` 1214e948e, Julia 1.10.12: 11 pass, 22 fail, 25 error of 58.
- GREEN: 58/58 on Julia 1.10.12 (105.9 s wall), 58/58 on 1.10.12 with `--check-bounds=yes` (113.6 s),
  58/58 on Julia 1.13.0 (98.4 s).
- Neighbours on 1.10.12, each file alone: `test_bridge_capabilities.jl` 242/242,
  `test_bridge_x.jl` 192/192, `test_confint_family.jl` 341/341.

## 6. Tests of the Tests

- RED shows the file fails on main in every testset. The 11 passes there are the five healthy-fit
  checks, the five direct-fit failure checks and one assertion of the endpoint testset; every
  assertion on the adapter's output fails or errors.
- The failing-draw testset asserts the direct fit is a genuine failure, that the old bare-vector
  contract accepts its θ, and that the new contract rejects it.
- The end-to-end testset mixes healthy and failing draws and asserts `0 < n_converged < 6` on all five
  routes.
- The endpoint-equality testset asserts all 10 replicates converge and that the bounds are finite
  before comparing, so it cannot pass vacuously.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- No other branch touches the five ZI closures (checked with `git diff origin/main...<branch>` over the
  #504 branches).
- The sibling slices all add a CHANGELOG bullet at the same place and an include next to the Poisson
  one in `test/runtests.jl`, so textual merge conflicts are expected there; both are additive.
- The zero-inflated closures call the fitters with `K` (plus `X`, `γ_fixed` or `N` where relevant) and
  defaults for every other keyword. Unchanged here.

## 9. What Did Not Go Smoothly

- Nothing blocking. The negative-count idea from the brief did not produce a failing draw, which the
  probe caught before any test was written.

## 10. Known Residuals

- Negative counts are silently treated as zeros by the ZIP, ZINB and ZIB likelihoods, and a fit on such
  data reports `converged = true`. This is input validation, outside the bootstrap; not changed.
- The ZI refits pass no `hessian` or `offset` keyword, so a fit made with non-default values is refit
  with the defaults (pre-existing; not probed).
- Full suite not run; CI will run it.

## 11. Team Learning

Probe the bad draw before writing the test: the obvious corruption (a negative count) was absorbed by a
`y > 0` branch, while a less obvious one (a half count) failed cleanly through a caught `InexactError`.

## 12. Cross-Product Coverage

Covered: all five zero-inflated fit types, Julia 1.10.12 and 1.13.0, aarch64 macOS, with and without
bounds checks on 1.10.12. Not covered: Linux and Windows (CI), K > 1, `γ_fixed` masks, ZIB covariate
fits (`ZIBCovFit` has no `_family_ci` method), offsets.
