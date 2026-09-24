# Totoro #323 Track A receipt

**Date:** 2026-09-24
**Paste ack:** Shinichi gave `ack Totoro D-139 #323 Track A` in chat on 2026-09-24. He approved the full run ("go Track A") after the pre-run below was reported.
**Executor:** Claude (lane `true-parity-20260924`). The runbook named Codex; Shinichi chose "Claude runs it here".
**GLLVModels.jl tip:** `94a7b56f9` (main after #409, #410, #399)
**gllvmTMB frozen pin:** `b4d5fee64def88bc768dda1f1f77c29b295edd86`
**Contract:** `docs/dev-log/core070/frozen-r070-contract.toml`
**Raw evidence:** `docs/dev-log/core070/totoro-323-track-a-20260924/` (cell receipts, `runparity.log`, oracle `build.json`, the run script and its log)

TRACKA_PRERUN dry_run=OK r_source=b4d5fee64def88bc768dda1f1f77c29b295edd86 archive=0c2f4323eb9fb19acccf039b8d57b4dd6bda82e2aa8b4a7bb712f36a64b022bc julia=1.10.12 R=4.5.3 wall_min=0.5
TRACKA_RESULT NATIVE06_gradmax=not_reached NATIVE10_gradmax=not_recorded NATIVE12_gradmax=0.0005901797966929578 wall_min=56

## Scope boundary

- IN: the Track A Frozen R smoke on Totoro, following the launch pack Runner block (steps 1 to 5a), and the pre-run that preceded it.
- OUT: programme or goal completion, advisory CI gating, any gllvmTMB `src/` edit, and waiving #323.

## Environment

| Field | Value |
|-------|--------|
| Host | Totoro (D-50), one process, OPENBLAS/OMP/JULIA threads 1, MAKEFLAGS=-j1 |
| R version | 4.5.3 |
| Julia | 1.10.12 (juliaup) |
| GLLVM_ROOT | `~/gllvmodels-track-a-20260924/GLLVModels.jl` |
| TRACK | A |
| RECEIPT_STAMP | 20260924-105152 |
| Wall clock | 56 min (10:51 to 11:47 MDT): oracle build and verify 7 min, Julia parity environment 3 min, `runparity.jl` 45 min 35 s (its own Test Summary), setup the rest |
| D-139 band | 90 to 150 min estimated; the run finished in 56 min, below the band |

## Pre-run (D-139), 2026-09-24 09:42 MDT, at #410's head `78de8d0a1`

Environment check only: Julia instantiate OK; launcher `--dry-run` printed `TOTORO_323_TRACK_A_PREFLIGHT_DRY_RUN_OK`; frozen source fetched at the pin; `core070_build_oracle.py prepare` printed `CORE070_ORACLE_SOURCE_PASS` with archive sha256 `0c2f4323…`. No prior oracle build existed on Totoro, so a gradient number was not possible before the full run.

A first attempt was stopped by us. The juliaup channel `1.10` is not installed on Totoro (the fix was `+1.10.12`), and a full-history gllvmTMB clone was taking 15 minutes (replaced by a depth-1 fetch of the pin).

## Commands (exact)

The script is committed as `totoro-323-track-a-20260924/trackA.sh`. It runs the launch pack's steps in order:

1. install_deps from the 2026-08-31 Posit snapshot
2. `core070_build_oracle.py` build and verify
3. the `test/parity` Julia environment, including the RCall build
4. the parity environment exports
5. `julia --project=test/parity test/parity/runparity.jl`

Deviations, both deliberate:
- `R_LIBS_USER` pointed to a directory inside the run folder, so `install_deps` could not write to Totoro's shared user R library. Package versions are unchanged: same snapshot, `upgrade = "never"`.
- Step 2b `prepare` refused because the pre-run had already prepared the archive from the same pinned commit (identical source, sha256 `0c2f4323…`). The build used that archive.

## Outcome

| Check | Result | Notes |
|-------|-----------|-------|
| Oracle build | PASS | `BUILD_EXIT=0`, `VERIFY_EXIT=0`, 7 min |
| Full runparity (Track A) | 14 of 17 required cells succeeded, 3 failed | exit 1; the three failures are exactly the #323 holdouts |
| Holdout NATIVE-06-NB2 `r_gradient_max ≤ 1e-4` | NOT REACHED | The cell stopped before its R check. The health helper's guard (`test/parity/nb2_health.jl:8`) refused with `original NB2 data changed`: data simulated under `Random.seed!(45)` no longer hashes to the frozen value. The Julia fit on that data also reported `converged = false`. |
| Holdout NATIVE-12-TRUNCATED-NB2 | FAIL, R side | `r_gradient_max = 5.90e-4` (2026-09-05 baseline 6.47e-4); native gradient 6.5e-6 |
| Holdout NATIVE-10-STUDENT | 32 of 33 pass | Parity Cell 9 (per-trait σ and ν, the twin default) passes: Δ logLik = 2.0e-8. The one failure is the near-Gaussian estimated-ν diagnostic, where the Julia fit reports `converged = false`. This cell's test records no `r_gradient_max`. |

The other 14 cells succeeded. They include both default delta cells, after today's #399 default change, and Tweedie, whose cell took 40 of the 56 minutes and passed all 28 assertions.

**Julia-version dependence (checked against CI).** The advisory Frozen R CI job runs the same frozen pin on Julia 1.13.0. On every PR in this lane (#409, #411, #472) it gives 278 pass / 8 fail, with the opposite holdout pattern: NATIVE-06 passes the data guard and fails on the R side (`r_gradient_max` 2.43e-3); NATIVE-10's Parity Cell 9 FAILS (R `optimizer_code` 1, |Δ logLik| 2.86e-3 > 1e-3); NATIVE-12 PASSES 21/21. Holdout outcomes therefore depend on the Julia version or platform, and no holdout counts as a pass on today's evidence.

## Failure classification

- [ ] R build / remotes / TMB compile: none; the oracle built and verified.
- [ ] Frozen oracle drift vs pin: none; source and build are at the pin.
- [x] Julia parity gradient or convergence miss (no `@test` tolerance widened):
  - NATIVE-12 is R-side.
  - The Student near-Gaussian diagnostic is Julia-side convergence.
  - NATIVE-06: on Totoro the fit ran on data that failed the frozen-hash guard, so its non-convergence says nothing about the frozen fixture. In the advisory CI job (Julia 1.13.0) the guard passes, and the frozen fixture gives R `r_gradient_max` 2.43e-3.
- [x] Fixture guard (NATIVE-06): the seeded-data hash guard failed on Julia 1.10.12. AGENT-INFERRED, not verified: the seeded RNG stream differs from the Julia version the hash was recorded on, which is the D-275 class ("a pin on a seeded fixture is a pin on the Julia version too"). `nb2_health.jl` last changed on 2026-09-18. Consistent with this inference: the same guard passes on Julia 1.13.0 in CI.

## What this does NOT cover

- It does not waive or close #323, and does not change the advisory status of the Frozen R CI job.
- It does not refresh NATIVE-06's R gradient (not reached) or NATIVE-10's (not recorded by that test).
- It is not a programme-completion or parity claim beyond the numbers in this table.

## Follow-up

- [ ] NATIVE-06: decide how the NB2 fixture should be pinned across Julia versions, and re-run that cell.
- [ ] NATIVE-10: decide whether the Student cell should record `r_gradient_max`, as the smoke baseline implies.
- [ ] NATIVE-12: the R-side gradient miss is unchanged from 2026-09-05; the existing disposition stands (`docs/dev-log/core070/advisory-smoke-fail-disposition-2026-09-05.md`).
