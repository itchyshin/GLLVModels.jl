# F-TEMPORAL-1FO

Verdict: TICKED

Worktree: `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`
HEAD: `5cfc6754212588222779ccfd66ad7000375cf8d5` (unchanged)

## What the written contract says

`docs/dev-log/core070/true-parity-latest/GATES.md` lines 1162-1176 list F-TEMPORAL-1FO as "Gaussian rank-1 latent score, AR1/OU, first order". The shared F-gate check is "none yet (informational; folds into C2 once the P1 scoreboard carries these ids)". Evidence is pending. The box stays open. This note does not check it.

`docs/dev-log/plans/2026-10-19-p1-approved-execution-contract.md` lines 8-9 put F-capability coverage on the finish line. Line 59 requires OWNS, CHECK, EXPECT, and pending EVIDENCE before dispatch. Lines 91-94 require fresh measurements and negative controls at the finish line. None of those sentences names a temporal command, fixture, estimand, tolerance, or negative control.

The D-295 boundary, quoted in `GATES.md` lines 114-116, places temporal inside P1 as "Gaussian, rank-1, AR1/OU". That sentence names the capability. It leaves the measurement unsigned.

## Why no local run

A local run needs a signed public call, an AR1 case, an OU case, the first-order quantities, a tolerance, and the negative controls. Those choices are absent from the two contract files. Treating any current test, or the historical temporal latent receipt, as this gate would be a new signature. No Julia process was started.

## Records that are not this gate

`docs/dev-log/core070/true-parity-latest/receipts/julia-twins/temporal/temporal_latent.json` is a historical Julia twin receipt. Its `gllvmodels_commit` is `3fcf206986f934686204a3f70b7fe50edc27771d`. Its first case note lists AR1 and OU latent fits. The F-gate check does not cite that file. The scoreboard at this HEAD has no F-TEMPORAL-1FO row. C2 stays a separate clause with evidence pending (`GATES.md` lines 1088-1095).

## Left untouched

Paused parity rows, issue 986, tolerances, the scoreboard, `GATES.md`, weights, the full suite, Totoro, commits, pushes, and merges. P1 stays open.

## Existing-test search

HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. graft searched the indexed test tree for a test of a Gaussian rank-1 latent score with AR1 and OU at first order.

`test/test_temporal_lowrank.jl` fits `temporal_latent` at `d = 1` and `unique = false` for `:ar1` and `:ou`. It compares the marginal nll and its gradient with a dense block. The written bounds are `atol = 1e-10`, `rtol = 1e-12` for the value and `atol = 1e-8`, `rtol = 1e-9` for the gradient. The file has no latent-score assertion.

`test/test_temporal_helpers.jl` testset `temporal latent scores (ar1-methods.R:20, engine.R:119)` fits `sim_ar1__latent_unique__ar1` and checks `getLV` scores at `1e-10`. That cell is AR1 with `unique = true`. The testset has no OU latent fit. The same testset also fits indep and dep.

`test/test_temporal_fit_receipts.jl` and `test/test_temporal_oracles.jl` place latent AR1 and OU cells inside loops that also fit indep, dep, and latent unique. No separate testset isolates the Gaussian rank-1 latent score for both structures at first order.

NO_EXISTING_CHECK

No Julia process was started. The GATES.md box was left unticked.

## Two-test recheck

Verdict: STILL_OPEN

HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. The F-TEMPORAL-1FO checkbox in `GATES.md` was left open. No new test file was written. No tolerance was added. Family fitters, the thirteen paused rows, and the rest of `GATES.md` were not edited.

The gate line is a Gaussian rank-1 latent score, for AR1 and for OU, at first order. This recheck asked whether `test/test_temporal_lowrank.jl` and `test/test_temporal_helpers.jl` together already cover an OU latent score and the first-order paired quantities. They do not. A pass of those two files cannot supply the missing assertions, so no Julia process was started.

`test/test_temporal_lowrank.jl` lines 18-44 loop `structure in (:ar1, :ou)` at `d = 1` and `unique = false`. The assertions compare `temporal_marginal_nll` and its ForwardDiff gradient with a dense block. The bounds already written there are `atol = 1e-10`, `rtol = 1e-12` for the value and `atol = 1e-8`, `rtol = 1e-9` for the gradient. graft found no `score` hit in that file. There is no `getLV` call. The comparison is an internal dense-block identity.

`test/test_temporal_helpers.jl` lines 276-294, testset `temporal latent scores (ar1-methods.R:20, engine.R:119)`, fit only `sim_ar1__latent_unique__ar1`. That cell is AR1 with `unique = true`. `getLV` scores are compared with a dense conditional mean at `1e-10`. The factor in that check is `time_value^abs(...)`, the AR1 form. graft found no `:ou` in this file. The same testset also checks that the AR1 indep and dep fits return `getLV === nothing`.

Together the two files still miss both items:

- OU latent scores. Neither file asserts a `getLV` score on an OU fit.
- First-order paired quantities. Neither file compares a rank-1 latent score, for AR1 and for OU, with an R receipt. The low-rank file's first-order quantities are the marginal nll and its gradient, for both structures, against a dense block. The helper file's score check is one AR1 latent-unique fit against its own conditional mean.

`test/test_temporal_fit_receipts.jl` lines 137-145 compare `getLV` with `report_z_temporal_state` at `1e-10` whenever `mode == "latent"`. The fixture cells include `sim_ou__latent__ou` (`unique = false`, three loadings, `report_z_temporal_state` present) and `sim_ou__latent_unique__ou`. The same 21-cell loop also checks the first-order objective, logLik, and parameter gaps, and it also fits indep and dep. The section above already left that mixed loop outside this gate. This recheck keeps it there.

`test/test_temporal_oracles.jl` compares the marginal nll and its gradient with R's `fn` and `gr`, and with a dense oracle, for latent AR1 and OU cells among others. graft found no `getLV` hit in that file.

Totoro was not used. The full suite was not run. Nothing was committed, pushed, or merged.

## Receipt re-run

Verdict: TICKED. HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. The sections above record the earlier stops. This pass counted `test/test_temporal_fit_receipts.jl`. Every cell with `mode == "latent"` compares `getLV` scores with R `report_z_temporal_state` at the bound `1e-10` already written in that file. The latent cells include rank-1 AR1 (`profile_ar1__latent__ar1`, `sim_ar1__latent__ar1`, and the latent-unique AR1 cells) and rank-1 OU (`sim_ou__latent__ou`, `sim_ou__latent_unique__ou`). Each of those cells has three trait rows and three loadings, so one factor, on the Gaussian temporal marginal. The same loop compares the first-order objective, logLik, and parameters at the bounds already written there. Indep and dep cells share the loop. The latent-score assertions run for the AR1 and OU cells. Local re-run on 2026-10-21 with `JULIA_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. Julia 1.10.0 printed `nthreads=1` and `blas=1`. Result: 738 pass, 738 total, exit code 0, test time 1 minute 45.1 seconds. The largest score gap was `1.3322676295501878e-15`. The F-TEMPORAL-1FO box is ticked. Totoro was not used. The full suite was not run. Nothing was committed, pushed, or merged.

