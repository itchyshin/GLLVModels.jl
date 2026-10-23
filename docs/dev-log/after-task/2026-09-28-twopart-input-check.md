# After-task: two-part fitters reject observed out-of-support values (2026-09-28)

Branch `claude/twopart-input-check` from `origin/main` @ `863ee0f78`; fix commit
`dd934d182`. Local only: not pushed, no PR.

## 1. Goal

Each public two-part fitter refuses an observed value outside its family's support
with an `ArgumentError` naming the value, its index and the support, before fitting.
Before this change every `_tp_pieces` branched on `y > 0`, so NaN and negative values
took the zero branch and were scored as observed zeros, and the fit reported
`converged = true`. Beta-hurdle also clamped values `>= 1` into (0,1). Valid data must
fit exactly as before.

## 2. Implemented

- `src/families/twopart.jl`: three predicates (`_tp_count_ok`, `_tp_positive_ok`,
  `_tp_unit_ok`) and one helper, `_check_twopart_support(fname, Y, ok, support)`.
  The helper scans `Y` and throws on the first failing cell.
- One call per fitter, placed after the keyword validation and before the warm start:
  `fit_zip_gllvm`, `fit_zip_gllvm_cov`, `fit_zinb_gllvm`, `fit_zinb_gllvm_cov`,
  `fit_zib_gllvm`, `fit_zib_gllvm_cov`, `fit_hurdle_poisson_gllvm`,
  `fit_hurdle_nb_gllvm`, `fit_delta_lognormal_gllvm`, `fit_delta_gamma_gllvm`
  (twopart.jl), `fit_beta_hurdle_gllvm` (beta_hurdle.jl) and
  `fit_delta_gamma_gllvm_va` (variational_dgamma.jl).
- `fit_gllvm`, `gllvm()`, the bridge and the bootstrap refits call these fitters, so
  they inherit the check.

Supports, read from each family's docstring and density:

| Family | Documented support | Refused up front |
|---|---|---|
| ZIP, ZINB, hurdle-Poisson, hurdle-NB | non-negative integer counts | NaN, negatives |
| ZIB | integers 0..N (scalar `N`) | NaN, negatives |
| delta-lognormal, delta-gamma (Laplace and VA) | 0 or positive real | NaN, negatives, `Inf` |
| beta-hurdle | 0, or a value in (0,1) (docstring) | NaN, negatives, values `>= 1` |

## 3. Decisions and Rejected Alternatives

- **No mask and no `missing` tests.** The brief asked that masked and `missing` cells
  stay allowed. None of these fitters has a `mask` keyword, and all take
  `Y::AbstractMatrix{<:Real}`, so a matrix holding `missing` is a `MethodError` at
  dispatch on `origin/main` and still is. `src/bridge.jl` lists missing-response masks
  for ZIB as a follow-up. There was nothing to preserve, so no test was written for it.
- **ZIB above `N` is not refused.** The brief asked for a bound check on ZIB, but also
  named `N + 1` as a bootstrap "bad draw" in PR #582 that must not be newly rejected.
  #582's test calls `fit_zib_gllvm` directly on such a draw and expects
  `converged = false` without a throw. A count above `N` already gives a `-Inf`
  log-density and ends on the failure verdict, so it is not silently mis-scored.
  I kept the soft failure; the check refuses NaN and negatives only.
- **0.5 and 1e300 are not refused**, as the brief required (#582, #583). The count
  predicate is `y >= 0`, not an integer check.
- **Beta-hurdle refuses exactly 1.** The docstring says values in `(0,1)`; the clamp
  comment ("guard against exact 0/1 data") suggested tolerance, but the brief said to
  follow the docstring.
- **`Inf` is refused for the delta families** (not a finite positive real). For the
  count families `Inf` already fails via `InexactError` inside the objective, so it is
  left alone, like 0.5.
- Rejected: one check inside `_tp_pieces`. The objectives wrap their bodies in
  `try/catch` and turn a throw into a penalty, so a throw there would be laundered
  into a non-converged fit rather than reaching the user.

## 4. Files Touched

`src/families/twopart.jl`, `src/families/beta_hurdle.jl`,
`src/families/variational_dgamma.jl`, `test/test_twopart_input_check.jl` (new),
`test/runtests.jl` (one include after `test_twopart_hessian_kwarg.jl`), `CHANGELOG.md`
(Unreleased / Fixed), `docs/dev-log/check-log.md`, this file.

## 5. Checks Run

All per file, `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, via the scratch runner.

- RED on `863ee0f78` (src clean): 27 passed, 92 failed. The 27 are the 24 valid-data
  assertions and the 3 soft-failure assertions, which hold before and after.
- GREEN: 119/119 on Julia 1.10.12, 119/119 on 1.10.12 `--check-bounds=yes`,
  119/119 on 1.13.0.
- Neighbours on 1.10, each file in its own process: 23 of 24 green; counts are in the
  check-log entry. `test_confint_family.jl` 341/341, `test_zero_inflated.jl` 29/29,
  `test_bridge_x.jl` 8 + 192. `test_second_order_delta_followup.jl` 43 pass + 1 broken
  (a `@test_skip` gated on `GLLVM_PARITY_TESTS`, reported as broken).
- `test_variational_dgamma.jl` errors when run alone: `dot` not defined at line 43,
  because the file does not load LinearAlgebra. The file is untouched and the error is
  independent of `src/`. With `using LinearAlgebra` preloaded it passes 17/17.
- Full suite not run.

## 6. Tests of the Tests

- RED shows every rejection assertion (30 value-by-fitter cases, three assertions each,
  plus 2 dispatcher cases) failing on `origin/main`: the bug is real at all twelve
  entry points.
- Each rejection asserts the fitter's name, the value and the index `[2, 3]` appear in
  the message, so a throw from somewhere else would not pass.
- The soft-failure testset pins the #504 contract from the other side: 0.5 (ZIP),
  `N + 1` (ZIB) and 1e300 (delta-lognormal) must return `converged = false` without a
  throw. A future over-eager check would fail it.

## 7. Issue Ledger

No GitHub issue opened. Interacts with draft PRs #582 (ZI bootstrap verdict), #583
(hurdle/delta) and the beta-hurdle/ordered-beta slice on
`origin/claude/bhob-boot-verdict-504`.

## 8. Consistency Audit

Read the three PR test files on their branches:

- #582 `test_confint_bootstrap_verdict_zi.jl`: bad draws are 0.5 (ZIP, ZINB, both
  `_cov`) and `N + 1` (ZIB). Neither is refused.
- #583 `test_confint_bootstrap_verdict_hurdle_delta.jl`: bad draw 1e300 for all four
  routes. Finite and positive, so not refused.
- `bhob` `test_confint_bootstrap_verdict_bhob.jl`: the beta-hurdle route uses a stub
  failed refit (`_bhob504_stub_failed`) and a sentinel matrix never passed to a fitter;
  its real bad draw (-1.0) is on the ordered-beta route only. Its adapter-parity test
  passes a simulated beta-hurdle draw, whose positives are `Beta` draws in (0,1).
  Unaffected.

Both #582 and #583 comment that a negative count (and NaN, in #583) "is NOT a failing
draw" because it is scored as a zero. After this change that sentence is stale: those
values now throw. Their tests do not use such draws, so nothing breaks, but the
comments should be updated when those PRs are next touched.

## 9. What Did Not Go Smoothly

- The brief's mask and `missing` requirements assumed an API these fitters do not
  have; found on reading the signatures.
- The brief asked both to reject ZIB counts above `N` and not to reject `N + 1`; I
  resolved it in favour of the open PR (section 3).
- RED ran in the branch worktree at `HEAD = origin/main` with `src/` clean (the
  runner prints `dirty=false`) rather than in a separate scratch worktree. Same code,
  one fewer checkout on the Dropbox path.
- My first attempt at the parallel neighbour runner failed (`xargs` command too long,
  `mapfile` absent from macOS bash 3.2); replaced by a two-slot shell loop.

## 10. Known Residuals

- A beta-hurdle bootstrap draw that rounds to exactly 1.0 in Float64 now makes that
  refit throw; `confint_family.jl` catches it and drops the replicate, where before it
  was clamped and fitted. Rare, but a behaviour change in bootstrap.
- ZIB counts above `N`, non-integer counts and `Inf` counts still reach the objective
  and end as `converged = false` rather than as an `ArgumentError`.
- The public `*_marginal_loglik_laplace` functions have no check; they are
  lower-level entry points, out of scope.
- `ordered_beta.jl` and `src/families/ordinal.jl` were not touched.
- Stale "negative is scored as a zero" comments in #582 and #583 (section 8).

## 11. Team Learning

A density that branches on `y > 0` sends NaN to the `else` branch silently; any
family written that way needs its support checked at the entry point, not in the
density, because the objective's `try/catch` turns a throw into a penalty.

## 12. Cross-Product Coverage

Twelve entry points by {NaN, negative} (all), plus `Inf` (three delta entry points)
and `-Inf`, 1.0, 1.5 (beta-hurdle); valid data for all twelve; the `fit_gllvm`
dispatcher for ZIP and beta-hurdle; soft-failure draws for ZIP, ZIB and
delta-lognormal. Not covered: `gllvm()` formula route and the bridge (they call the
same fitters), and Julia 1.13 neighbours.
