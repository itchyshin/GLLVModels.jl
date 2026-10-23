# After-task: Delta dispersion Option A, D1 remeasure

**Date:** 2026-09-24
**Branch:** `claude/delta-a-d1-remeasure-20260924` from `origin/main` @ `94a7b56f9` (post #399 merge)
**Worktree:** `/Users/z3437171/local-scratch/gllvm-delta-a-d1-20260924`
**Scope:** Remeasure D1 for the two Delta second-order cells per
[`2026-09-16-delta-dispersion-a-d1-remeasure-runbook.md`](../plans/2026-09-16-delta-dispersion-a-d1-remeasure-runbook.md),
now that PR #399 shipped `disp_group = :species` as the public default
(decision: [`2026-09-15-delta-dispersion-alignment-pending.md`](../decisions/2026-09-15-delta-dispersion-alignment-pending.md),
ACCEPTED (A) 2026-09-24). Contract: [`second-order-parity-contract.md`](../core070/second-order-parity-contract.md) section 4.

```
D1_RESULT lognormal=PASS gamma=PASS
```

## R oracle used

- Frozen `gllvmTMB` at `/Users/z3437171/local-scratch/R-gllvmtmb-frozen-b4d5fee64`.
- Loaded version and path, printed at run time: `packageVersion("gllvmTMB")` = `0.7.0`,
  `find.package("gllvmTMB")` = `/Users/z3437171/local-scratch/R-gllvmtmb-frozen-b4d5fee64/gllvmTMB`.
- Confirmed this is a distinct build from the machine's default install at
  `/Users/z3437171/Library/R/arm64/4.6/library/gllvmTMB` (same DESCRIPTION version string,
  different compiled `gllvmTMB.so` SHA1, different `NAMESPACE`/`NEWS.md` size, different mtime),
  so the run genuinely exercised the frozen snapshot and not whatever happened to be installed
  by default.

**A surprising finding, worth recording plainly.** `GLLVM_PARITY_R_LIBS` is read by
`test/parity/parity_helpers.jl` (its `_parity_prepend_twin_lib!` mechanism), but the code path
this remeasure actually uses, `tools/core070_second_order/common.jl`'s `_require_gllvmtmb!`
(called by both smoke scripts and by `test_second_order_delta_followup.jl`'s R-paired testset),
does **not** read `GLLVM_PARITY_R_LIBS` at all. It just calls `library(gllvmTMB)` and trusts R's
own `.libPaths()`. Setting only `GLLVM_PARITY_R_LIBS` would have had no effect here and the run
would have silently loaded the machine's default `gllvmTMB` instead of the frozen oracle. The
fix was to also export the standard R environment variable `R_LIBS_USER`, combining the frozen
library first (for priority) with the default user library second (so `gllvmTMB`'s other R
dependencies, e.g. `assertthat`, still resolve):

```bash
export R_LIBS_USER="/Users/z3437171/local-scratch/R-gllvmtmb-frozen-b4d5fee64:/Users/z3437171/Library/R/arm64/4.6/library"
```

`GLLVM_PARITY_R_LIBS` was still exported alongside it (matching the instruction and in case any
other consumer reads it), but `R_LIBS_USER` is what actually made the frozen library load.

## Estimate (before running)

A timed probe (`smoke_delta_lognormal_eoo.jl`, after the worktree's one-time `Pkg.instantiate()`)
completed in 30.6s of R+Julia work, 36s wall including semaphore queueing. Both cells are small
(p=5, K=1, n=130), so the full remeasure (test file with the R-paired testset re-running both
cells, plus both smoke scripts) was estimated at 2-3 minutes wall, well under the 30-minute
D-139 gate. Full estimate note in
`/Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-20260924/.unlazy/true-parity-20260924/logs/d1-remeasure.log`.
Measured actual: 49s (test file) + 37s + 37s = under 2.5 minutes total.

## Commands run

All through the lane semaphore
(`/Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-20260924/.unlazy/true-parity-20260924/checks/julia_slot.sh`,
which pins `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`), from the worktree above,
with `GLLVM_PARITY_TESTS=1`, `GLLVM_PARITY_R_LIBS` and `R_LIBS_USER` set as above:

```bash
julia --project=. test/test_second_order_delta_followup.jl
julia --project=. tools/core070_second_order/smoke_delta_lognormal_eoo.jl \
  docs/dev-log/core070/delta-lognormal-2so-d1-remeasure-receipt.json
julia --project=. tools/core070_second_order/smoke_delta_gamma_eoo.jl \
  docs/dev-log/core070/delta-gamma-2so-d1-remeasure-receipt.json
```

Full output saved to
`/Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-20260924/.unlazy/true-parity-20260924/logs/d1-remeasure.log`.

`test/test_second_order_delta_followup.jl`: 49 pass / 49 total (the R-paired testset ran live
this time instead of skipping).

## D1 result: before (2026-09-15) vs after (2026-09-24)

Each-own-optimum, contract section 4 bounds: SE rel <= 1e-2 (scaled by `cond(H)_R/1e3` when
`cond(H)_R > 1e3`), vcov Frobenius rel <= 1e-2 (same scaling), CI endpoint <= 1e-4 abs or <= 5e-2
relative to interval half-width. Both R condition numbers below are under 1e3, so no scaling
applies (scale = 1.0).

| Cell | logLik delta (J-R) | SE max rel delta | vcov Fro rel delta | CI endpoint max delta | r_cond(H) | D1 |
|------|---:|---:|---:|---:|---:|---|
| `delta_lognormal` seed 61, before | -1.923 | 0.145 | 0.221 | 0.041 | (not recorded) | FAIL |
| `delta_lognormal` seed 61, after | 1.82e-8 | 4.01e-5 | 6.34e-5 | 1.81e-5 | 84.3 | **PASS** |
| `delta_gamma` seed 62, before | (same class as fid 13, no explicit number recorded) | 0.212 | 0.255 | 0.082 | (not recorded) | FAIL |
| `delta_gamma` seed 62, after | 1.79e-8 | 3.06e-5 | 3.76e-5 | 1.14e-5 | 22.3 | **PASS** |

The 2026-09-15 FAIL was a parameterisation gap: Julia used one shared scalar dispersion while R
estimated one dispersion per trait, so the two sides were different models with different
maximised likelihoods. With `disp_group = :species` now the default on both the named fitters
and `fit_gllvm`, both sides fit the same model, and the each-own-optimum deltas collapse to
essentially machine precision on logLik and to two to four orders of magnitude inside the SE,
vcov and CI bounds. Both fits converged with a positive-definite Hessian on both sides
(`pd_hessian_native` and `pd_hessian_r` both true for both cells).

No tolerance in `second-order-parity-contract.md` section 4 or in
`tools/core070_second_order/eoo_assess.jl` (`EOO_SE_REL`, `EOO_VCOV_FRO_REL`, `EOO_CI_REL_HALF`,
`COND_SCALE_THRESHOLD`) was changed to reach this result. The change was entirely on the
estimand side (matching `disp_group`), landed and merged in PR #399, before this remeasure ran.

## Rose fence

- EOO smoke `eoo_smoke_pass` is not by itself the D1 programme gate; this after-task judges D1
  directly against contract section 4's each-own-optimum table, using the receipt fields, not
  just the smoke script's boolean.
- Accepting Option A (the 2026-09-24 decision-doc paste) was never itself evidence of a D1 pass.
  D1 needed a live remeasurement against the R oracle, which is what this after-task records.
- This remeasure is each-own-optimum only, matching what the contract says ships
  ("the claim uses each-own-optimum"). It is not a matched-coordinates second-order claim, not a
  multi-seed recovery or coverage claim, and not a programme section 7 closure.
- No rtol, atol, or contract-section-4 bound was widened anywhere in this slice.

## Files touched

- `docs/dev-log/core070/delta-lognormal-2so-d1-remeasure-receipt.json` (new receipt)
- `docs/dev-log/core070/delta-gamma-2so-d1-remeasure-receipt.json` (new receipt)
- `docs/dev-log/after-task/2026-09-24-delta-dispersion-a-d1-remeasure.md` (this file)

## Next

Both cells passed, so the remaining post-paste closeout item from the decision doc (mark DRAFT
#399 ready and merge) is already done (#399 merged before this slice started). What is left per
the runbook is a decision-doc update recording the D1 pass, and any downstream capability-ledger
row promotion, which is a maintainer/board call, not made here.
