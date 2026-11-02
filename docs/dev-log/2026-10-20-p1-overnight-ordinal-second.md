# ORDINAL-LOGIT-RSZ second convention check

Date: 2026-10-21. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Verdict: **NO_NEW_BUG**. No source file was edited. The tolerance 0.0001 is unchanged. ORDINAL-LOGIT-RSZ is not admitted.

## Scope

The earlier note `docs/dev-log/2026-10-20-p1-overnight-ordinal.md` already evaluated Julia's per-trait Laplace likelihood at R's intercepts, R's cutpoints, and a lower-triangular factor of R's `Lambda Lambda'`. This note does not repeat that evaluation and does not report it as a new measurement. No fit was run. Julia was not started.

The question here is whether the stored `(t01, t15)` pair is the wrong comparison: a trait permutation, a cutpoint sign or a free `tau_1`, the sign of `beta + Lambda z`, or a loading rotation that the receipt treats as the same covariance.

## Trait order

Pinned input sha256 `df6a45d4f51cabe50a394aad203fae7368fd2abb88729d067e7683d2d193c0c1` on both raw campaign files. Formula on the R file: `value ~ 0 + trait + latent(0 + trait | site, d = 2, unique = FALSE)`.

`trait_levels` on the R json and the Julia toml are the same list, `t01` through `t20`. `LLt_trait_names` on the R json is that same list. R `beta_names` run `traitt01` through `traitt20`. Julia builds the response with traits sorted (`wide_from_long` in `tools/true_parity/campaign/run_J.jl`) and stores `rows(fit.Λ * fit.Λ')` in that row order. The receipt writer compares `LLt` by position (`tools/true_parity/campaign/write_receipts.py`). Position `(1, 15)` is therefore the named pair `(t01, t15)` on both sides.

Both stored matrices are symmetric (maximum absolute asymmetry 0). The receipt values sit at that position: R `0.243906750881311`, Julia `0.24403664215890872`, absolute difference `0.00012989127759771568`. Two of the 400 entries exceed 0.0001, and they are this pair and its transpose.

Positional intercepts from the same files differ by at most `3.283892589023285e-06` (trait `t04`). No intercept changes sign. A swapped trait row would move an intercept by far more than that.

## Cutpoints and `tau_1`

Both engines fix `tau_1 = 0` and build the later cutpoints by cumulative `exp` of free log-spacings.

Julia `_unpack_cutpoints_pertrait` (`src/families/ordinal.jl`) sets column 1 to 0 and then `tau[t, c] = tau[t, c-1] + exp(psi)`. The stored Julia `tau` is 20 by 3, every first entry is 0, and every free entry is positive. Categories per trait are 4, so there are two free cutpoints per trait.

On the pinned gllvmTMB commit `9539352f66f2db2cc26b1c393e67212a359b60c9`, family id 20 sets `cuts(0) = 0` and the same cumulative exponential (`src/gllvmTMB.cpp` in that commit). R's `extract_cutpoints` rows are indices 2 and 3 only (40 rows). The receipt maps index `i` to Julia column `i - 1`, which lands on the free cutpoints and skips the fixed zero. All 40 R values are positive. The maximum absolute difference on those 40 pairs is `5.585089662085707e-06`.

Both cell probabilities use `F(tau_k - eta) - F(tau_{k-1} - eta)` with the logistic CDF. Julia `_ord_prob` subtracts in that order. The pinned R source states the same expression for family id 20. A reversed cutpoint sign would negate and reorder the positive thresholds. The stored thresholds stay positive and within `5.585e-06`.

## Sign of `beta + Lambda z`

Julia's per-trait site likelihood sets `eta = beta + Lambda * z` (`ordinal_loglik_site_pertrait`). The score `dP/d eta` is `f(tau_{c-1} - eta) - f(tau_c - eta)`, and the Newton step multiplies that score by `Lambda'`. The chain rule for `eta = beta + Lambda * z` produces that product.

On the pinned R commit the latent contribution is `u = sum_k Lambda_B(t, k) * z_B(k, s)` and `eta += u`. The same plus sign.

A column sign flip of `Lambda` leaves `Lambda Lambda'` unchanged, and a flip of one trait row would flip the sign of that trait's covariances. Zero of the 400 stored entries disagree in sign. The intercepts, which share the linear-predictor sign with the loadings, also agree in sign.

## Loading rotation

Both packers use the same lower-triangular gauge. Julia `unpack_lambda` (`src/packing.jl`) writes the diagonal, then the strict lower triangle column by column, and leaves the strict upper triangle at 0. `gll_unpack_rr_loadings` at the pinned R commit does the same. `fit_ordinal_gllvm_pertrait` returns that unpacked matrix. It does not rotate the loadings after the optimisation. R records `LLt_from_Lambda_B_max_abs_diff = 0`, so the stored R matrix is `Lambda_B Lambda_B'`.

The receipt rule calls `Lambda Lambda'` rotation and sign invariant. For this ordinal row the writer does not apply the temporal sign helper `orient`. An orthogonal rotation of the latent axes, or a sign flip of a whole latent axis, does not change the product either engine stores. The two stored products still differ by `0.00012989127759771568` at `(t01, t15)`. The largest absolute diagonal gap is `1.8920253896936856e-05`, so one side has not had the logistic residual variance `pi^2/3` added to the diagonal.

## Result

The named pair `(t01, t15)` is the same entry on both stored matrices. Cutpoints, `tau_1 = 0`, and the sign of `beta + Lambda z` match in the two sources and in the stored fit. The Gram matrices are already the rotation-invariant estimand, and they still differ by more than 0.0001 on this one pair. No ordinal source file was changed. The comparison `0.00012989127759771568` stays above 0.0001.
