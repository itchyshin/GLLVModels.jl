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
- `src/loading_profile_confirmatory.jl` (**new file**, not
  `src/confint_derived.jl`): the exported `loading_profile(fit::GllvmFit; ...)`
  method (multiple dispatch alongside the existing generic
  `loading_profile(args...; kwargs...)` deprecation shim, which lives in
  `confint_derived.jl` and is untouched — see "Shim collision" below).
  Refuses non-confirmatory fits and fits with no free entries; builds a grid
  per free entry from a Wald-SE heuristic (`_profile_wald_se`, reused
  unchanged) and refits at each grid point via
  `_confirmatory_profile_refit_lambda_pin(...; require_paste = false)`.
  Included from `src/GLLVModels.jl` immediately after `confint_derived.jl`.
  **Kept out of `confint_derived.jl` deliberately** — see "Lane bleed" below.
- `docs/src/confidence-intervals.md`: new "Confirmatory fits and
  `loading_profile` (D3 Stage 1)" subsection under "Gaussian engine"
  (**not** `docs/src/derived-confidence-intervals.md` — see "Lane bleed").
- `docs/src/api.md`: added `loading_profile` to the existing "Inference &
  Confidence Intervals" `@docs` block (`fit_gaussian_gllvm`'s updated
  docstring, with the new `lambda_constraint` section, already renders there
  automatically via the existing `fit_gaussian_gllvm` entry — no separate
  edit needed for that half).
- `test/test_loading_profile_confirmatory.jl`: new top-level `@testset`
  (9 nested testsets, 26 assertions) — fit-time pin exactness on
  `MASK-B-PINS`/`MASK-B-UPPER`/`MASK-B-ALLFIXED`, the all-NaN no-op fast path,
  four refusal paths, one grid-cell profile, the `entries` filter, and a
  regression guard that the old 3-positional-argument `loading_profile(fit, t,
  k)` calling convention still dispatches to `loading_profile_exploratory`
  unchanged.
- `CHANGELOG.md`: `Unreleased` → `Changed` entry.

## Lane bleed fix (coordinator-flagged, 2026-09-24)

Open foreign Codex PR #437 (`codex/derived-ci-reader-cleanup`) touches exactly
`src/confint_derived.jl` and `docs/src/derived-confidence-intervals.md`. This
slice originally added the new `loading_profile` method and its docs to those
same two files. Moved out:

- `src/confint_derived.jl` restored **byte-identical** to `e613a56a4`
  (`git diff e613a56a4 -- src/confint_derived.jl` is empty). The new method
  moved verbatim into new file `src/loading_profile_confirmatory.jl`, included
  from `src/GLLVModels.jl` right after `confint_derived.jl`'s include line.
  Dispatch is unaffected by file/include order — Julia resolves multiple
  dispatch by method specificity, not definition order — verified by re-
  running the full test file after the move (unchanged: 31 pass / 1 broken).
- `docs/src/derived-confidence-intervals.md` restored **byte-identical** to
  `e613a56a4`. The docs cascade moved to `docs/src/confidence-intervals.md`
  (narrative, under its existing "Gaussian engine" section, which already
  covers `fit_gaussian_gllvm`/`confint`/`profile_ci`/`bootstrap_ci` — the
  natural home) and `docs/src/api.md` (added `loading_profile` to the
  existing "Inference & Confidence Intervals" `@docs` block).
- The shim-dispatch regression test (`test/test_loading_profile_confirmatory.jl`,
  "old 3-positional-arg shim still dispatches to loading_profile_exploratory")
  is kept, unchanged.

## Bug found and fixed in DRAFT #411's own substrate

**File:** `src/loading_profile_confirmatory_internal.jl`.
**Function:** `_confirmatory_lambda_pin_theta_fixes` (the pre-existing #411
function; the same bug was present verbatim in the new fit-time counterpart
`_confirmatory_lambda_constraint_theta_fixes` I wrote by copying its pattern,
and was fixed there too before either was tested).

- **Old line:** `push!(fixes, (idx, Float64(v) / σ_eps))`
- **New line:** `push!(fixes, (idx, Float64(v)))`

The old line divided the raw pin value by `σ_eps` before fixing it in
`θ_packed`, under a comment asserting "Working values use the J1 packed scale
`L = Lambda / sigma_eps`". `src/likelihood.jl`'s `gaussian_nll_packed`
(`Λ = unpack_lambda(θ_rr, p, K)`, used directly with no rescaling before
entering `gaussian_marginal_loglik`) unpacks the packed `Λ_B` block straight
into `Λ` with **no** σ_eps rescaling anywhere — the packed loading entries are
raw-scale directly. This bug was never caught in #411 because the only test
that exercises the actual refit numerics
(`test_loading_profile_confirmatory.jl`'s "J1 pin-and-refit smoke") only runs
when `ENV["GLLVM_STAGE1_PASTE"]` is set, which it never was in CI — the test
was silently `@test_skip`'d every run.

**Numeric evidence** (interactive probe, `MASK-B-PINS` fixture, `p=3, K=2`,
seed `MersenneTwister(49)`, `n=40`, pinning `Λ[1,1] = -0.8` and `Λ[3,2] = 0.0`,
fitting via `fit_gaussian_gllvm(Y; K=2, lambda_constraint=pins)`):

| | `Λ[1,1]` (target `-0.8`) | `Λ[3,2]` (target `0.0`) |
|---|---|---|
| **Before fix** | `-1.0863138493246327` (35.79% relative offset) | `0.0` (exact — masks the bug, since `0 / σ_eps == 0` for any `σ_eps`) |
| **After fix** | `-0.8` (exact) | `0.0` (exact) |

The `Λ[3,2] = 0.0` pin could not have exposed this bug on its own — only a
nonzero pin can, since dividing zero by anything is still zero. The implied
`σ_eps` at the reference (unconstrained) fit was `0.8 / 1.0863138493246327 ≈
0.7364`, consistent with the division hypothesis. Re-verified after the fix
in the committed test suite: `fit.pars.Λ[1, 1] == -0.8` and
`fit.pars.Λ[3, 2] == 0.0` (exact `==`, not `≈`), plus the `MASK-B-ALLFIXED`
case (all five packed coordinates pinned, including three nonzero values
`0.8`, `0.7`, `-0.15`) landing exactly.

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

- **No R numeric comparison has run — investigated precisely, found not
  achievable in this checkout.** The coordinator asked for one test cell
  comparing the new `lambda_constraint` path against the frozen R oracle
  named in `docs/dev-log/core070/loading-profile-confirmatory-substrate.json`
  (`masks-known-contract.json` case
  `CORE070-MASKS-KNOWN-MASK-B-PINS-PAIRED-CONTROL`, point `MASK-B-PINS-P1`,
  `r_nll = 65.5136777950417`, tolerance `abs_nll_delta = 1e-06`, per
  `docs/dev-log/core070/masks-known-evidence.json`). Two independent reasons
  this is not achievable here, documented in full (with the exact evidence
  trail) as a comment in `test/test_loading_profile_confirmatory.jl`
  immediately after the internal-consistency grid-cell test:
  1. The point's raw inputs (observations/design/parameters) are not present
     in this checkout — only SHA-256 provenance of an ephemeral campaign
     sandbox (`masks-known-points-01/attempt1/out/MASK-B-PINS-P1/*.tsv`)
     under `retained_artifacts`. Confirmed absent by search of this repo,
     `~/local-scratch` (including the true-parity lane and the frozen R
     library directory itself), and `~/shinichi-brain`.
     `docs/dev-log/core070/masks-known-leaf.md` gives enough to reconstruct
     the *parameter point* exactly (`beta = (.2,-.1,.3)`, `sigma_eps = .8`,
     full loading `(.8,.7,.1,.2,-.15)` before pinning, at P1) but not the
     *response matrix* the NLL was evaluated against, which is what is
     actually missing.
  2. Even with the point's inputs, that comparand is an NLL match at a
     **fixed, given** parameter vector (a likelihood-kernel check), not a
     converged refit/MLE match, for an R call with per-trait fixed
     intercepts (`value~0+trait+latent(...)`, i.e. `beta != 0`). This
     slice's `lambda_constraint` fit-time path is deliberately restricted to
     `X = nothing`, so it cannot reproduce that R call's model shape
     regardless of data availability.
  A live R call through the frozen oracle library
  (`/Users/z3437171/local-scratch/R-gllvmtmb-frozen-b4d5fee64`, gllvmTMB @
  `b4d5fee64`) was considered per the coordinator's allowance. It requires an
  *existing* `GLLVM_PARITY_TESTS=1`-gated live-R helper to reuse;
  `test/parity/parity_helpers.jl` has the gating pattern, but no
  `fit_gllvmtmb_parity_*` helper there calls `gllvmTMB(...,
  lambda_constraint = ...)` — every one is for an unconstrained fit. Building
  that call would be new R-calling infrastructure for this one cell, not
  reuse of an existing gated path, so it was not built.
  "R-aligned" in the internal-consistency test's original name overclaimed
  this — renamed to "one pin-and-refit grid cell, internal consistency
  (MASK-B-PINS)" to not imply a numeric R match that was never run. That test
  checks internal consistency only (non-negative deviance, ~zero deviance at
  the confirmatory MLE) — a Stage 1 receipt, not parity evidence.
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

## Flag for whoever lands this (resolved 2026-09-24)

Originally flagged: `docs/src/derived-confidence-intervals.md`'s "Difference
from R" paragraph was also touched, unmerged, by
`origin/codex/derived-ci-reader-cleanup` (PR #437). **Resolved by the lane
bleed fix above** — that file is now byte-identical to `e613a56a4` again, and
this slice's docs cascade moved to `docs/src/confidence-intervals.md` and
`docs/src/api.md`, neither of which PR #437 touches. No remaining flag on
this point.

## Checks run

All re-run after the lane bleed fix (file move); counts identical to before
the move, confirming the relocation did not change behavior.

- `git diff e613a56a4 -- src/confint_derived.jl` — empty (byte-identical).
- `git diff e613a56a4 -- docs/src/derived-confidence-intervals.md` — empty
  (byte-identical).
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
