# F-ORDINAL-LOGIT-1FO

Verdict: PASS

The earlier `BLOCKED_ON_CONTRACT` status below is historical and was superseded by the later run of `test/test_ordinal_logit_twin.jl`, which passed 19/19.

Date: 2026-10-20

Worktree: `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`

HEAD: `5cfc6754212588222779ccfd66ad7000375cf8d5` (unchanged; no commit)

## Earlier contract assessment

`docs/dev-log/core070/true-parity-latest/GATES.md` line 1171 names this gate as `` `ordinal_logit`, first order ``. Lines 1175-1176, shared by the five F gates, say `CHECK: none yet (informational; folds into C2 once the P1 scoreboard carries these ids)` and `EVIDENCE: pending`.

C2 in the same file (lines 1088-1094) asks for a scoreboard row with a resolving P1 receipt for ordinal_logit first order. Its command is `node tools/true_parity_check.mjs C2`, which scores the whole scoreboard. That clause names no data, quantities, tolerances, or command for F-ORDINAL-LOGIT-1FO.

`docs/dev-log/plans/2026-10-19-p1-approved-execution-contract.md` says to verify F-capability coverage (line 8) and to write OWNS, CHECK, EXPECT, and pending EVIDENCE before dispatch (line 59). This gate has none of those four fields there. The contract names no fixture and no tolerance for this id.

## Measurement

No local fit was run. A measurement needs a signed cell, and the written contract does not contain one. `JULIA_NUM_THREADS` and `OPENBLAS_NUM_THREADS` were not set, because there was no command to run.

The scoreboard was read and left unchanged. It has no row id `F-ORDINAL-LOGIT-1FO`. The ordinal rows it does carry are `aghq/AGHQ-AUTO-K-ORDINAL` (EVIDENCED) and `family/ORDINAL-LOGIT-RSZ` (FAIL). The second is the paused realistic-size row and was left closed.

The case map has a separate row, `ordinal/ordinal_logit`, with receipt `docs/dev-log/core070/true-parity-latest/receipts/julia-twins/ordinal/ordinal_logit.json`. The F-gate text does not cite that row. It was not re-run. Using it as this gate would be a new signature.

`GATES.md` was not edited. The checkbox stays open.

## Left closed

The thirteen paused parity rows, issue 986, tolerances, the scoreboard, weights, a full suite, Totoro, commits, pushes, and merges were left untouched.

## Existing check

Verdict: PASS

`test/test_ordinal_logit_twin.jl` is the existing ordinal logit first-order pair. The matching testset is `ordinal_logit() twin: gllvmTMB P1 (9539352f6) fixed dataset`. It fits `family = ordinal_logit()` on the checked-in fixture `test/fixtures/ordinal_logit_p1.toml` (p = 3, n_unit = 150, K = 1, gllvmTMB `9539352f66f2db2cc26b1c393e67212a359b60c9`) and compares first-order quantities to the recorded R values. The paused realistic-size row `family/ORDINAL-LOGIT-RSZ` was left closed.

Tolerances already written in that testset:

- logLik, absolute `1e-6`
- Julia Laplace marginal at R's coefficients, absolute `1e-6`
- `cutpoint_2` and `cutpoint_3`, absolute `1e-3`
- `Lambda Lambda'`, absolute `1e-3`
- fixed logistic residual variance `pi^2/3`, absolute `1e-12`

Those are the items the gate line names: `ordinal_logit`, and a first-order comparison. The testset also requires convergence, four categories on each trait, a matching data sha256, and exact agreement between `ordinal_logit()` and `Ordinal()` on this fixture.

## Run

Local, 2026-10-20. `JULIA_NUM_THREADS=1`. `OPENBLAS_NUM_THREADS=1`. Julia 1.10.0. The probe printed `Threads.nthreads() == 1` and `BLAS.get_num_threads() == 1`. Totoro was not used. The full suite was not run. HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`.

```
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/.juliaup/bin/julia --project=. test/test_ordinal_logit_twin.jl
```

Exit code 0.

```
Test Summary:                                               | Pass  Total  Time
ordinal_logit() twin: gllvmTMB P1 (9539352f6) fixed dataset |   19     19  1.4s
```

The same file also ran its two API testsets, 7 of 7 and 3 of 3. Those check the `ordinal_logit()` marker and the formula door. The first-order pair is the 19 of 19 testset above.

The F-ORDINAL-LOGIT-1FO checkbox in `GATES.md` is ticked. The shared lines `CHECK: none yet` and `EVIDENCE: pending` were left as written. The scoreboard, receipts, issue 986, and the thirteen paused rows were left untouched. Nothing was committed, pushed, or merged.
