# Overnight check of RD-SPIDER-NB2 and RD-BEETLE-NB2

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`. HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. No grouped-NB2 source file was edited. No tolerance, scoreboard verdict, GATES ledger, or decision packet was edited. Nothing was committed, pushed, or merged. Totoro was not used. The full suite was not run. The beetle pathology split was not run.

RD-SPIDER-NB2: NO_PROVEN_BUG

RD-BEETLE-NB2: BLOCKED_BY_R_PDHESS

## What was evaluated

Spider's data file was rebuilt with the campaign loader (`tools/true_parity/campaign/gen_data.R`) and local `gllvm` 2.0.13. The sha256 is `e9341ee556df40e6900888e2dff56f81c13d2588235eb2f83ce562a4487f799b`, the same value as `RD-SPIDER-NB2.json` and `raw/spider_R.json.gz`.

The shared grouped-NB2 Laplace objective, `nb_grouped_marginal_loglik_laplace` with observed curvature, inner `maxiter = 100`, and `tol = 1e-9`, was then evaluated at both stored parameter sets. Local Julia was 1.10.0, one thread, `OPENBLAS_NUM_THREADS=1`. Loadings were the eigenfactor of the stored `Lambda Lambda'` matrix. That factor reproduces the stored matrix to `2.132e-14` on the Julia side and `5.329e-14` on the R side. The covariate offset is the three shared slopes times the scaled `ConWate`, `BareSand`, and `CovMoss` columns.

## RD-SPIDER-NB2

At Julia's stored intercepts, slopes, and dispersions the objective is `-686.1679951712287`. The receipt value is `-686.1679951713709`. Absolute difference `1.422e-10`.

At R's stored intercepts, slopes, dispersions, and `Lambda Lambda'` the same objective is `-686.167996629499`. R's stored log-likelihood is `-686.167996628694`. Absolute difference `8.050e-10`.

The two evaluations differ by `1.458e-6`, matching the receipt gap of `1.457e-6`. Both engines are reporting this objective. Julia's point is the higher one. R stopped on relative convergence with max absolute gradient `0.00294377547171853`. The same-day capture already recorded Julia's max absolute gradient `6.774e-6` at its returned point, with reconstruction error 0.

On the 336 spider counts, the conditional NB2 log-density matches a 256-bit evaluation of the same mean-size formula to a maximum absolute error of `7.372e-14` at Julia's linear predictor and `9.326e-14` at R's. The same Float64 density matches that reference at the beetle-sized spot checks `r = 1.016e21` and `r = 9.672e74` (counts 0, 3, and 5; means `1e-8`, 1, and 2). Spider's stored linear predictors lie in `(-19.8765, 5.4065)`, inside the eta clamp of ±30.

The LLt gap of `0.0001617` on `sp03` and the linear-predictor gap of `6.740e-5` at stored index 304 are the parameter gap between these two points. Among the nine dispersions under `1e6`, the largest absolute R-versus-Julia difference is `0.0001892` (`sp04`). The three Poisson-limit dispersions are far apart on a flat direction: `sp01` `5.637e8` versus `2.209e10`, `sp05` `9.759e5` versus `4.866e12`, and `sp11` `1.511e9` versus `1.060e12`.

No incorrect line in the shared grouped-NB2 objective was found. The three receipt gaps stay outside the existing bars. The row is not admitted.

## RD-BEETLE-NB2

R `pdHess` is false on `RD-BEETLE-NB2.json` (`engines.R.pdHess`, and `pass_rule.legs.R_pdHess_true`). The campaign binds a row only when that leg is true, so the row stays unbound even if the log-likelihood, beta, LLt, and linear-predictor gaps later fall inside their tolerances. This check did not change the flag. The flat-dispersion / loading-degeneracy / negative-curvature split was not run.

A local rebuild of `beetle_wide.csv` with the same loader and `gllvm` 2.0.13 has sha256 `5ea37a22fab9ca2fbb599e18e8fa28ed7477675f73db8add3b6dc0df0419f545`. The receipt data sha256 is `1d7603ddceac61ea7179ed9519cb87bc5b7557ec1dd30f9267bd6224cd952c9b`. The beetle Laplace objective was not evaluated on that mismatched file. The row is not admitted.

## Code

No source change. Spider's objective already agrees with R's stored log-likelihood when it is given R's parameters, so there is no formula correction that would bring the three spider gaps inside the bars without moving Julia off its higher point of the same objective.
