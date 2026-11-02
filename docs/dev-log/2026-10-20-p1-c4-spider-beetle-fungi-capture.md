# P1 C4 spider, beetle, and fungi returned-fit capture

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`, branch `codex/p1-isdm-current-main-20261019`, commit `5cfc6754212588222779ccfd66ad7000375cf8d5`.

This file records three diagnostic captures. None of `data/RD-SPIDER-NB2`, `data/RD-BEETLE-NB2`, or `data/RD-FUNGI-BINOMIAL` is admitted. The scoreboard, GATES ledger, and decision packet were left unchanged. `data/RD-URBANISATION-BINOMIAL` was not run.

The signed contract is the 2026-10-17 four-fit method: reconstruct the objective at the returned parameter vector with `reconstruction_atol=0.0`, inner `maxiter=100` and `tol=1e-9`, analytic gradients, one Julia thread, `OPENBLAS_NUM_THREADS=1`, and a 30-minute stop. Spider and beetle use the grouped-NB2 analytic gradient and the fit's stored Laplace curvature mode (`observed`). Fungi uses the binomial covariate analytic gradient with `has_disp=false`. No new Hessian, confidence interval, prediction, polish, or beetle curvature split was run.

Host: Totoro, through the existing ControlMaster. Load before the three cells was 60.33, 62.10, and 71.65. Julia was Totoro's juliaup binary, version 1.12.6. The source tree was a `git archive` of `5cfc6754212588222779ccfd66ad7000375cf8d5` at `/home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/repo`. The capture script SHA-256 is `e94565bc03c3c95c8d457d69fa5eef0096ff9d8dc6865bb3bb2ee19dfe210780`. The three pinned input hashes matched before each fit. The cells ran one after another.

## data/RD-SPIDER-NB2

Ran. Exit 0. `2026-10-20T17:45:16Z` to `2026-10-20T17:46:04Z`. Status `CAPTURED_DIAGNOSTIC_ONLY`. `NBGroupedCovFit`, converged true, 139 iterations.

Command:

```text
timeout --signal=TERM 1800 env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_BLAS_THREADS=1 OMP_NUM_THREADS=1 GLLVM_JL_SHA=5cfc6754212588222779ccfd66ad7000375cf8d5 /home/snakagaw/.juliaup/bin/julia --project=/home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/repo --threads=1 --startup-file=no /home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/capture.jl spider /home/snakagaw/hsq_work/true-parity-campaign-20261002/data /home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/out/spider
```

Reported logLik `-686.16799517124798`. Reconstructed objective `686.16799517124798`. Reconstructed logLik `-686.16799517124798`. Reconstruction absolute error `0`. Max absolute gradient `6.7742738197318353e-06`. Fit wall `22.396661996841431` seconds. Input `e9341ee556df40e6900888e2dff56f81c13d2588235eb2f83ce562a4487f799b`.

Raw record: `/home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/out/spider/capture.toml`. Capture blocker: none. Row admission: not admitted.

## data/RD-BEETLE-NB2

Ran. Exit 0. `2026-10-20T17:46:04Z` to `2026-10-20T17:52:58Z`. Status `CAPTURED_DIAGNOSTIC_ONLY`. `NBGroupedCovFit`, converged true, 228 iterations. No diagnostic separating flat dispersion, loading degeneracy, and negative curvature was run.

Command:

```text
timeout --signal=TERM 1800 env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_BLAS_THREADS=1 OMP_NUM_THREADS=1 GLLVM_JL_SHA=5cfc6754212588222779ccfd66ad7000375cf8d5 /home/snakagaw/.juliaup/bin/julia --project=/home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/repo --threads=1 --startup-file=no /home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/capture.jl beetle /home/snakagaw/hsq_work/true-parity-campaign-20261002/data /home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/out/beetle
```

Reported logLik `-8933.781193989882`. Reconstructed objective `8933.781193989882`. Reconstructed logLik `-8933.781193989882`. Reconstruction absolute error `0`. Max absolute gradient `6.5472829954215804e-06`. Fit wall `391.95455884933472` seconds. Input `1d7603ddceac61ea7179ed9519cb87bc5b7557ec1dd30f9267bd6224cd952c9b`.

Raw record: `/home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/out/beetle/capture.toml`. Capture blocker: none. Row admission: not admitted.

## data/RD-FUNGI-BINOMIAL

Ran. Exit 0. `2026-10-20T17:52:58Z` to `2026-10-20T17:54:17Z`. Status `CAPTURED_DIAGNOSTIC_ONLY`. `GllvmCovFit`, converged true, 154 iterations. Analytic covariate gradient, tolerances unchanged.

Command:

```text
timeout --signal=TERM 1800 env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_BLAS_THREADS=1 OMP_NUM_THREADS=1 GLLVM_JL_SHA=5cfc6754212588222779ccfd66ad7000375cf8d5 /home/snakagaw/.juliaup/bin/julia --project=/home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/repo --threads=1 --startup-file=no /home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/capture.jl fungi /home/snakagaw/hsq_work/true-parity-campaign-20261002/data /home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/out/fungi
```

Reported logLik `-5792.1862523887676`. Reconstructed objective `5792.1862523887676`. Reconstructed logLik `-5792.1862523887676`. Reconstruction absolute error `0`. Max absolute gradient `7.5812079072257177e-06`. Fit wall `55.806670904159546` seconds. Input `498b84b93d2759785272cf33e7620f556033cb28c8a017596e146d6b5318b1bb`.

Raw record: `/home/snakagaw/hsq_work/p1-c4-capture-20261020-5cfc67542/out/fungi/capture.toml`. Capture blocker: none. Row admission: not admitted.

## Limit

A finished capture is a receipt. It does not bind a scoreboard row, change a tolerance, or authorise a merge. None of the three rows is admitted.
