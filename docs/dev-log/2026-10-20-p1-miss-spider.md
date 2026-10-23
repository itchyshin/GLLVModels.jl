# RD-SPIDER-NB2 three misses

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Diagnosis only. `data/RD-SPIDER-NB2` is not admitted. The tolerances are unchanged.

The receipt is `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-SPIDER-NB2.json`. Its `row_status` is the string from `reasons` at `tools/true_parity/campaign/write_receipts.py:343`: each failed case prints `abs difference` of the checker maximum against its tolerance. `maxabs` at `tools/true_parity/campaign/write_receipts.py:88` is `max(abs(x - y) for x, y in zip(a, b))`. `Case` at `tools/true_parity/campaign/write_receipts.py:109` stores that maximum. Recomputing it from the receipt's `r_value` and `julia_value` returns the three stored maxima.

| Miss | Label | R | Julia | Absolute difference | Tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| logLik | REAL_DIFFERENCE | -686.167996628694 | -686.1679951713709 | 1.4573231510439655e-06 | 1e-06 |
| LLt, entry (sp03, sp03) | REAL_DIFFERENCE | 15.3191830196009 | 15.319344747390675 | 0.00016172778977541213 | 0.0001 |
| linear predictor, stored index 304 of 336 | REAL_DIFFERENCE | -19.8764715520553 | -19.87653895149775 | 6.739944245026663e-05 | 1e-06 |

## logLik: REAL_DIFFERENCE

`cases_for` at `tools/true_parity/campaign/write_receipts.py:216` compares `R["logLik"]` with `J["logLik"]` as one scalar. R writes `logLik = -opt$objective` at `tools/true_parity/campaign/run_R.R:120`. Julia writes `fit.loglik` at `tools/true_parity/campaign/run_J.jl:207`.

`abs(-686.167996628694 - (-686.1679951713709)) = 1.4573231510439655e-06`. Julia's reported marginal log-likelihood is higher by that amount. `reasons` prints it as `1.45732e-06 > 1e-06`.

The 2026-10-20 capture in `docs/dev-log/2026-10-20-p1-c4-spider-beetle-fungi-capture.md` rebuilt the returned Julia objective. Reported logLik `-686.16799517124798`. Reconstructed logLik `-686.16799517124798`. Reconstruction absolute error `0`. Maximum absolute gradient `6.7742738197318353e-06`. The absolute gap between that captured log-likelihood and the receipt Julia value `-686.1679951713709` is `1.2289547157706693e-10`. The capture file has no R log-likelihood. It shows that the receipt's Julia number is Julia's own objective at the returned parameters. It does not match that objective to R.

R returned a finite objective: convergence `0`, message `relative convergence (4)`, `pdHess` true, `max_abs_gradient` `0.00294377547171853` (`run_R.R:121` and the receipt's R engine block). The same data sha256 is true on the receipt. The compared scalars are the two engines' marginal log-likelihoods, and they differ by more than `1e-06`.

## LLt: REAL_DIFFERENCE

`cases_for` at `tools/true_parity/campaign/write_receipts.py:237` compares `R["LLt"]` with `J["LLt"]` element by element. No sign flip and no trait reorder. `orient` at `tools/true_parity/campaign/write_receipts.py:142-146` is used only for temporal loadings.

R writes `extract_Sigma(fit, level = "unit", part = "shared", link_residual = "none")$Sigma` at `tools/true_parity/campaign/run_R.R:143-145`, and sets `LLt_from_Lambda_B_max_abs_diff` to `max(abs(Lambda_B %*% t(Lambda_B) - S))`. The raw file `raw/spider_R.json.gz` records that check as `0`, so the compared R matrix is `Lambda_B Lambda_B'`. Trait names are `sp01` through `sp12`.

Julia writes `rows(fit.Λ * fit.Λ')` at `tools/true_parity/campaign/run_J.jl:208`. `rows` at `tools/true_parity/campaign/run_J.jl:39` stores one matrix row per TOML row.

The maximum is the diagonal entry for `sp03`, row-major index 26 (0-based), and the receipt stores it as `0.000161728 > 0.0001`. Seven of 144 entries exceed `0.0001`. They are three symmetric pairs plus that diagonal: `(sp01, sp04)`, `(sp03, sp03)`, `(sp03, sp04)`, and `(sp04, sp08)`. The median absolute entry difference is `2.0185772e-05`. Both stored matrices are symmetric at these entries.

`Lambda Lambda'` is the rotation-and-sign invariant named in the tolerance rule at `tools/true_parity/campaign/write_receipts.py:155`. The trait names run `sp01` through `sp12` on the R matrix. The fixed-effect block, compared in the same species order, stays inside its own bar: maximum absolute beta difference `4.9511286195569326e-05` against tolerance `0.0001`, at the `sp03` intercept (`-1.09188500312509` versus `-1.0919345144112855`). A swapped species would move an intercept by an amount on the scale of the intercepts themselves. The remaining LLt gap is the value of the `sp03` latent variance, `15.3191830196009` versus `15.319344747390675`.

## Linear predictor: REAL_DIFFERENCE

`cases_for` at `tools/true_parity/campaign/write_receipts.py:249` compares `R["eta"]` with `J["eta"]` under the name `linear predictor on the training data (link scale)`. There is no key alignment step. The spider predict-key alignment later in that function is only for the iSDM predict quantities.

R writes `as.numeric(fit$report$eta)` at `tools/true_parity/campaign/run_R.R:137`. `eta_n` in the raw R file is `336`, which is `28 * 12`. The long data are trait-fast: `long_from_wide` at `tools/true_parity/campaign/run_R.R:69-70` repeats each site across the species and fills `value` with `as.vector(t(as.matrix(w[, sp])))`.

Julia, for `NBGroupedCovFit`, writes `vec(fit.β .+ O .+ fit.Λ * Z')` at `tools/true_parity/campaign/run_J.jl:218-221`, with `Z = getLV(fit, Y, X; rotate = false)`. `getLV` for this fit is `src/families/grouped_dispersion.jl:826-834`. With `rotate = false`, `_grouped_getLV` at `src/families/grouped_dispersion.jl:325` returns the site modes without `_svd_rotation`. The comment at `src/families/grouped_dispersion.jl:294-296` says those modes are the `z` at which the Laplace objective was evaluated. `vec` of the `p` by `n` matrix is trait-fast, the same layout as the R long data.

The maximum is stored index 304 (1-based). Under that trait-fast layout it is species `sp04` at site 26: R `-19.8764715520553`, Julia `-19.87653895149775`. `reasons` prints `6.73994e-05 > 1e-06`. The first twelve stored pairs already agree at about `1e-05` while those twelve linear predictors differ from one another by amounts of order 1, which is what matching species at one site look like. 297 of 336 entries exceed `1e-06`. The median absolute difference is `5.5978898e-06`. Every species has at least 20 of 28 sites over `1e-06`.

That pattern is a small shift of the fitted linear predictor. A rotated or sign-flipped score, or a species permutation, would move entries by amounts on the scale of the linear predictor (values from about `-20` to `5`) or of the latent variances (the `sp03` diagonal is about `15.3`). The largest linear-predictor gap, `6.739944245026663e-05`, is the same order as the fixed-effect gap that still passes `0.0001`.

## Recorded Hessian flags

These flags are not inputs to the three comparisons. The receipt records Julia `pd_hessian` false (`engines.julia.pd_hessian`, and `fixed_block!` at `tools/true_parity/campaign/run_J.jl:63` writes `ci.pd_hessian`). It records R `pdHess` true. `cond_H_statement` says Julia `cond(H)` was not recorded because the Hessian is not positive definite or not finite. The log-likelihood, `Lambda Lambda'`, and training linear predictor above are point values.

`data/RD-SPIDER-NB2` is not admitted.
