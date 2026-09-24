# After-task: D3 `loading_profile` Stage 1 slice (2026-09-24)

Lane: `claude/d3-stage1-slice-20260924`, worktree
`/Users/z3437171/local-scratch/gllvm-d3-stage1-slice-20260924`, branch created
at rebased DRAFT #411 head `e613a56a4`. **Work is local only — not pushed.**
PR #411 must merge first; this lane pushes afterwards per the coordinator's
instruction.

## Scope

Maintainer paste **`G0 Stage 1`** received 2026-09-24. Implemented the
runbook's ("First actions after paste",
`docs/dev-log/plans/2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md`)
bounded slice:

1. Fitter pin path: `fit_gaussian_gllvm(y; K, lambda_constraint = M)`.
2. Public export: `loading_profile(fit; level, entries, n_grid, grid_extent,
   conf_level, y)`, exact scout signature
   (`docs/dev-log/after-task/2026-09-14-loading-profile-d3-surface-scout.md`).
3. Tests: unit coverage plus one pin-and-refit grid cell on the frozen Stage 0
   fixtures.
4. Ledger: **not touched** — the runbook requires paired fixture evidence
   before rebinding `namespace/export/loading_profile`, and this slice has no
   cross-package (R) numeric comparison, only internal consistency checks. See
   "Not claimed" below.
5. Docs cascade: reference page, `CHANGELOG.md`, this report, check-log entry.

## Change

- `src/families/aghq_gaussian_fit.jl`: `fit_gaussian_gllvm` gains
  `lambda_constraint = nothing`. When supplied, fits the ordinary unconstrained
  model first (unchanged code path, zero modification), then calls
  `_fit_confirmatory_lambda_constraint` to re-optimise with the pinned entries
  held fixed. Refuses `aghq`, `mask`/`offset`, structured blocks
  (`K_W > 0`/`has_diag`/`K_phy > 0`/`has_phy_unique`), and `X`-carrying fits —
  all out of Stage 1 scope. The original unconstrained code path is untouched
  (the new branch is an early return before any existing logic runs).
- `src/loading_profile_confirmatory_internal.jl`: added
  `_confirmatory_lambda_constraint_theta_fixes` (fit-time counterpart of
  DRAFT #411's `_confirmatory_lambda_pin_theta_fixes`, fixing *all* of a pin
  matrix's numeric entries rather than one profile-grid override) and
  `_fit_confirmatory_lambda_constraint` (builds the confirmatory `GllvmFit`,
  reusing `_profile_refit_with_multi_fixed` and `_derived_unpack` from the
  existing profile/derived-CI machinery; stores `pars.lambda_constraint`).
- `src/confint_derived.jl`: new `loading_profile(fit::GllvmFit; ...)` method
  (multiple dispatch alongside the existing generic
  `loading_profile(args...; kwargs...)` deprecation shim — see "Shim
  collision" below). Refuses non-confirmatory fits and fits with no free
  entries; builds a grid per free entry from a Wald-SE heuristic
  (`_profile_wald_se`, reused unchanged) and refits at each grid point via
  `_confirmatory_profile_refit_lambda_pin(...; require_paste = false)`.
- `test/test_loading_profile_confirmatory.jl`: new top-level `@testset`
  (9 nested testsets, 26 assertions) — fit-time pin exactness on
  `MASK-B-PINS`/`MASK-B-UPPER`/`MASK-B-ALLFIXED`, the all-NaN no-op fast path,
  four refusal paths, one grid-cell profile, the `entries` filter, and a
  regression guard that the old 3-positional-argument `loading_profile(fit, t,
  k)` calling convention still dispatches to `loading_profile_exploratory`
  unchanged.
- `docs/src/derived-confidence-intervals.md`: rewrote the "Difference from R"
  paragraph (the old text asserted no confirmatory mode exists — now false)
  and added `loading_profile` to the `@docs` block.
- `CHANGELOG.md`: `Unreleased` → `Changed` entry.

## Bug found and fixed in DRAFT #411's own substrate

`_confirmatory_lambda_pin_theta_fixes` divided the raw pin value by `σ_eps`
before fixing it in `θ_packed`
(`push!(fixes, (idx, Float64(v) / σ_eps))`), under the comment "Working values
use the J1 packed scale `L = Lambda / sigma_eps`". `src/likelihood.jl`'s
`gaussian_nll_packed` unpacks `θ_rr_B` straight into `Λ` via `unpack_lambda`
with **no** σ_eps rescaling — the packed Λ entries are raw-scale directly.
This bug was never caught in #411 because the only test that exercises the
actual refit numerics (`test_loading_profile_confirmatory.jl`'s "J1
pin-and-refit smoke") only runs when `ENV["GLLVM_STAGE1_PASTE"]` is set, which
it never was in CI — the test was silently `@test_skip`'d every run. Removing
the division (both in the existing single-entry function and the new
fit-time counterpart) makes pins land **exactly** on their target raw values:
verified interactively (`Λ[1,1] == -0.8`, `Λ[3,2] == 0.0` on `MASK-B-PINS`;
all four `MASK-B-ALLFIXED` entries exact) and in the new test assertions.

## Shim collision (resolved, not a STOP)

The scout's target design ("only the confirmatory mirror takes
`loading_profile`... shim removed in a later maintainer-approved API slice")
and the runbook's fence ("no removal of the deprecation shim in the same PR
as first export") appear to conflict on first read. They do not: Julia
multiple dispatch resolves it. The new method's signature is
`loading_profile(fit::GllvmFit; kwargs...)` — exactly one positional
argument, type-annotated. The existing shim's signature is
`loading_profile(args...; kwargs...)` — fully generic. A call with one
`GllvmFit` positional argument dispatches to the new, more specific method; a
call with the old `(fit, t, k)` three-positional-argument shape (which was
never valid under the new method, since it has no `t`/`k` parameters) still
falls through to the untouched generic shim. No previously-working call
pattern changes behavior; `loading_profile(fit)` alone previously always
errored (the shim required `t`, `k`) and now succeeds only when `fit` is
confirmatory. Verified with an explicit regression test.

## Not claimed

- **No R numeric comparison has run.** No RCall session and no frozen R
  readback JSON for this specific grid quantity exist in this repo yet (per
  the scout, that is future work). "R-aligned" in the test file name means
  "built on R-derived Stage 0 pin fixtures", not "numerically matched against
  R's `loading_profile()` output". The grid-cell test checks internal
  consistency only (non-negative deviance, ~zero deviance at the confirmatory
  MLE, monotonic-ish shape) — a Stage 1 receipt, not parity evidence.
- **Grid spacing is a documented heuristic**, not R's exact rule (which this
  session did not have R source available to consult precisely): half-width =
  `grid_extent * Wald_SE`, falling back to `grid_extent * (|Λ̂|/2 + 0.5)` when
  the Hessian-based SE is non-finite.
- **`namespace/export/loading_profile` ledger row: not touched.** Per the
  runbook, binding it requires paired fixture evidence this slice does not
  yet have. T5 row 8 stays as it was.
- Stage 1 scope only: ordinary J1 Gaussian (`K_W = 0`, `has_diag = false`,
  `K_phy = 0`, `has_phy_unique = false`), `X = nothing`. Structured blocks,
  `X`-carrying fits, `aghq`, `mask`/`offset` combined with `lambda_constraint`
  all refuse with an explicit `ArgumentError`, not a silent fallback.
- The `loading_profile` deprecation shim is untouched — not removed, not
  altered.

## Fences honoured

No `Project.toml` bump. `src/grouped_nongaussian_fit.jl` not touched. No
existing `rtol`/`atol` widened (the bug fix removed an erroneous division, not
a tolerance). gllvmTMB (R) not touched. No GitHub `@handle` used anywhere in
this report or the check-log entry.

## Flag for whoever lands this

`docs/src/derived-confidence-intervals.md`'s "Difference from R" paragraph is
also touched, unmerged, by `origin/codex/derived-ci-reader-cleanup` (a
reader-facing wording pass with different text over the same paragraph, no
Stage 1 content). The two will conflict on merge; whoever integrates them
needs to reconcile content (this slice documents new behavior that branch's
version does not know about), not just pick one side's prose.

## Checks run

- `julia --project=. -e 'using Pkg; Pkg.instantiate(); using GLLVModels'` — OK.
- `julia --project=. test/test_loading_profile_confirmatory.jl` — 31 pass
  (5 pre-existing + 26 new), 1 broken (`@test_skip`, expected: the paste-gated
  low-level smoke test in the pre-existing testset still requires
  `ENV["GLLVM_STAGE1_PASTE"]`, which the new export path does not use).
- `julia --project=. test/test_loading_profile_stage1_harness.jl` — 9 pass.
- `julia --project=. test/test_loading_profile_stage0.jl` — 24 pass (existing
  test that includes the substrate file DRAFT #411 modified).
- `julia --project=. test/test_fit.jl` — 12 pass.
- `julia --project=. test/test_confint_profile.jl` — 8 pass.
- `julia --project=. test/parity/test_gaussian_parity.jl` — 31 pass (Gaussian
  R↔Julia oracle parity, unaffected).
- `julia --project=. test/test_confint_derived.jl` — 45 pass.
- `test/runtests.jl` unchanged (no new test *file*, only new testsets in an
  existing one); `grep -o '_shard_include("[^"]*")' test/runtests.jl | sort |
  uniq -d` empty; `Meta.parseall` on `runtests.jl` succeeds.
- Docs build (`julia --project=docs docs/make.jl --local`): **not run** — the
  `docs/` project environment is separate from the main one and would need
  its own `Pkg.instantiate()` inside the time budget available; skipping per
  the task's "skip if over 15 min, say so" allowance rather than risk an
  unbounded first-instantiate.

## Rose

Claim matches code: this is a **Stage 1 receipt** — fit-time pinning and a
profiling export exist, are tested for internal consistency and exact pin
placement on the Stage 0 fixtures, and refuse cleanly outside their scope. It
is **not** full R grid parity (no R numeric comparison has run) and **not**
`T5` row 8 "covered" (ledger untouched, by design). The one substantive risk
this session found and closed was silent, not cosmetic: DRAFT #411's pin
scaling was wrong in a way its own test suite could not have caught, because
the only test that would have caught it never ran with the paste set.
