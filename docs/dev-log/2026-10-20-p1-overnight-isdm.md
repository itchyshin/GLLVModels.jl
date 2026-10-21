# ISDM-HEADLINE-RSZ overnight check

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`. HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. No source file was edited. `GATES.md`, the scoreboard, and the receipt were left unchanged. No commit, push, merge, Totoro job, new fit, or full suite.

Verdict: **NO_PROVEN_BUG**.

`isdm/ISDM-HEADLINE-RSZ` is not admitted. The predict tolerance stays `1e-6` absolute, as written in `docs/dev-log/core070/true-parity-latest/receipts/isdm/campaign/ISDM-HEADLINE-RSZ.json`. The 2026-10-17 approval names ordinal, spider, beetle, and fungi. This row is outside that list. No fifth capture was started.

## The failing comparison

The receipt verdict is FAIL. Log-likelihood and `b_fix` are already inside their bars. Four predict quantities are not.

| Quantity | Absolute difference | Bound | Within bound |
| --- | ---: | ---: | --- |
| logLik | 4.090543370693922e-07 | 1e-6 | yes |
| beta | 1.1181552182104593e-05 | 0.0001 | yes |
| predict_link | 7.691881574919956e-05 | 1e-6 | no |
| predict_response | 0.0001490010101150574 | 1e-6 | no |
| predict_newdata_offset0_link | 7.691881574919956e-05 | 1e-6 | no |
| predict_newdata_offset0_response | 9.933400675521398e-05 | 1e-6 | no |

R records convergence 0, `pdHess` true, message `relative convergence (4)`, and maximum absolute gradient `0.0211623821882103`. Julia records `converged` true and 97 iterations. The Julia gradient and Hessian are absent from the receipt, and the receipt has no `convergence_parity` block. This check did not compute a new gradient.

The raw files are `docs/dev-log/core070/true-parity-latest/receipts/isdm/campaign/raw/isdm_R.json.gz` and `isdm_J.toml.gz`. Both have 20000 prediction rows. Alignment by `(cell_id, trait, isdm_source)` is one to one.

## What the stored predictions already satisfy

In-sample link minus offset-zeroed link is `log(1.5)` on every gbif row and `log(0.9)` on every survey row, on both engines. The largest departure from those two constants is under `1e-14`. The signed link gap and the signed offset-zeroed link gap differ by at most `6.661338147750939e-15`, so the offset cancels and the discrepancy sits in the fixed part plus the latent part.

Stored response equals the inverse link of the stored linear predictor on each engine: `exp(eta)` on gbif, `1 - exp(-exp(eta))` on survey. The largest absolute departure is `2.5579538487363607e-13` on R and `3.552713678800501e-15` on Julia. The response maximum, cell `c251` trait `sp14` source gbif, is that map applied to a link gap of `3.06417467266229e-06` at a mean of about `48.62687`. The same link gap, with the offset removed, is the offset-zeroed response maximum `9.933400675521398e-05`. All four failing numbers are one linear-predictor gap.

`Lambda Lambda'` is stored on both sides even though the receipt does not score it. Elementwise maximum absolute difference `3.3775344014636e-05`. Leading eigenvalues: R `5.075264363459551` and `1.4776345178821375`; Julia `5.075229902295653` and `1.4776220603064263`.

Removing `X (b_R - b_J)` from the link gap leaves a residual whose gbif and survey copies differ by at most `1.0380585280245214e-14`. The latent contribution is shared across sources. The fixed-effect piece has maximum absolute value `2.275087e-05`. The residual has maximum absolute value `7.893051e-05`. Its singular values are those of two rank-2 score differences; the fifth singular value is about `1e-13`. `14681` of `20000` link rows exceed `1e-6`. The median absolute link gap is `2.12719e-06`.

## Reconstruction at the stored points

Local, 2026-10-20. Julia 1.10.0. `JULIA_NUM_THREADS=1`. `OPENBLAS_NUM_THREADS=1`. The campaign Julia raw file was written by 1.12.6. The run built `isdm_table` on `tools/true_parity/campaign/data/isdm.csv.gz` with the campaign formula and `isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink()))`. It did not call `fit_isdm_gllvm`.

Each stored `Lambda Lambda'` was factored by its top two eigenpairs. The factor reproduces the stored matrix to `2.6645352591003757e-15` (R) and `3.3306690738754696e-15` (Julia). Coefficient names match `table.X_names`. For each of the 500 cells, `_isdm_cell_loglik` (`src/families/isdm_laplace.jl`) returned a converged mode, and `_isdm_eta` assembled the linear predictor the same way `fit_isdm_gllvm` stores `fit.eta` (`src/families/isdm_fit.jl`). In-sample `predict` copies that vector (`src/families/isdm_predict.jl`).

| Comparison | Maximum absolute difference |
| --- | ---: |
| Rebuilt linear predictor at R's `(b_fix, Lambda Lambda')` vs stored R `predict_link` | 1.12810205621372e-10 |
| Rebuilt log-likelihood at that R point vs stored R logLik `-23699.3261706562` | 5.384208634495735e-10 |
| Rebuilt linear predictor at Julia's `(b_fix, Lambda Lambda')` vs stored Julia `predict_link` | 6.194224300148932e-11 |
| Rebuilt log-likelihood at that Julia point vs stored Julia logLik `-23699.326170247146` | 2.546585164964199e-11 |
| Rebuilt R point vs rebuilt Julia point | 7.691881768623343e-05 |

Every cell mode converged on both points. The rebuilt gap sits on the same row as the receipt, cell `c434` trait `sp01` source gbif.

## Why no source edit follows

The current Laplace linear predictor, evaluated at each stored parameter point, reproduces that engine's stored `predict_link`. The `7.691881574919956e-05` receipt gap is the gap between those two parameter points. Beta already differs by `1.1181552182104593e-05`, inside the beta bar of `0.0001` and above the predict bar of `1e-6`. `Lambda Lambda'` differs by `3.3775344014636e-05`. The linear predictor multiplies those shifts by the covariates and the latent scores.

R's maximum absolute gradient at the returned point is `0.0211623821882103`. The two log-likelihoods differ by `4.090543370693922e-07`. A change to `predict` would have to move one engine's linear predictor by about `1e-4` while leaving these stored parameters' likelihood values where they are. The reconstruction says the coded predictor is already the stored predictor, to `1e-10`.

No iSDM source file was edited. The receipt stays FAIL.

NO_PROVEN_BUG
