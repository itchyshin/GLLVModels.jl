# ORDINAL-LOGIT-RSZ returned-fit capture

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. This file records one diagnostic capture. ORDINAL-LOGIT-RSZ is not admitted. C3 is not met.

## Repair

The shared campaign runner still accepts extra cells, a nonnegative reconstruction tolerance, and more than one Julia thread. A C4 capture writes through that same runner, so it was left unchanged. The ordinal-only runner is `tools/true_parity/campaign/ordinal_logit_rsz_capture.jl`. Its settings are fixed in the file:

- central finite differences
- h = 1e-5
- inner maxiter = 100
- inner tol = 1e-9
- one Julia thread, `OPENBLAS_NUM_THREADS=1`
- stop at 1800 seconds
- reconstruction absolute tolerance 0.0

The capture stops before confidence intervals, prediction, point replacement, and `DONE`.

## Command

Totoro load was 71 when the job started, under the 150-core cap. Fresh directory `/home/snakagaw/hsq_work/p1-ordinal-logit-rsz-20261020`. One thread.

```sh
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export ORDINAL_RSZ_OUT=/home/snakagaw/hsq_work/p1-ordinal-logit-rsz-20261020/out-ordinal-20261020
export GLLVM_JL_SHA=5cfc6754212588222779ccfd66ad7000375cf8d5
timeout --signal=TERM 1860 /home/snakagaw/.juliaup/bin/julia --project=. --threads=1 --startup-file=no \
  tools/true_parity/campaign/ordinal_logit_rsz_capture.jl
```

The outer timeout is a backstop. The runner's own clock stops the capture at 1800 seconds. Julia 1.12.6. Host `totoro`.

## Numbers

Pinned input `ordinal.csv` sha256 `df6a45d4f51cabe50a394aad203fae7368fd2abb88729d067e7683d2d193c0c1`. Size p = 20, n = 500, 99 returned coordinates. `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, BLAS threads = 1.

| Quantity | Value |
| --- | ---: |
| status | CAPTURED_DIAGNOSTIC_ONLY |
| converged | true |
| iterations | 53 |
| reported logLik | -11506.928561299756 |
| reconstructed logLik | -11506.928561299756 |
| reconstruction absolute error | 0.0 |
| reconstruction status | RECONSTRUCTED |
| max absolute gradient | 9.367795428261161e-6 |
| gradient coordinates finished | 99 |
| fit seconds | 69.781 |
| one-objective seconds | 0.624 |
| gradient seconds | 1.277 |
| capture seconds | 79.885 |

Package precompile in the log was about 18 seconds before that capture clock. The 1800-second stop did not fire. Raw record: `docs/dev-log/2026-10-20-p1-ordinal-logit-rsz-capture.toml`.

## Admission

ORDINAL-LOGIT-RSZ is not admitted. This capture does not bind the scoreboard row, does not meet C3, and does not replace the existing campaign receipt.
