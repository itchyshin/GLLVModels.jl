# Four P1 gaps have no shared comparison defect

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Diagnosis only. No fits were rerun. No row is admitted. No tolerance is changed.

The first-pass labels stay as written: ordinal LLt, spider, and fungi are REAL_DIFFERENCE, and beetle's numeric gaps are REAL_DIFFERENCE with R `pdHess` false kept as R_SIDE_BLOCKER. This note asks whether one shared, already-approved comparison or normalization defect explains more than one of those gaps.

## How the four quantities are aligned

`maxabs` at `tools/true_parity/campaign/write_receipts.py:88` subtracts the stored series element by element. `cases_for` does not flip a sign on these rows.

Log-likelihood is one scalar. R stores `-opt$objective` at `tools/true_parity/campaign/run_R.R:120`. Julia stores `fit.loglik` (`tools/true_parity/campaign/run_J.jl:117` for ordinal, `tools/true_parity/campaign/run_J.jl:207` for spider, beetle, and fungi). The receipt scalars are negative on both sides.

Beta is scored for spider, beetle, and fungi, and not for ordinal. `cases_for` at `tools/true_parity/campaign/write_receipts.py:225-234` requires the R names to be the trait intercepts in `trait_levels` order, then the three covariate slopes. Julia stores `vcat(fit.β, fit.γ)` at `tools/true_parity/campaign/run_J.jl:215`.

`Lambda Lambda'` is the rotation and sign invariant named at `tools/true_parity/campaign/write_receipts.py:155`. R stores `extract_Sigma(fit, level = "unit", part = "shared", link_residual = "none")$Sigma` at `tools/true_parity/campaign/run_R.R:143-145`, and records `LLt_from_Lambda_B_max_abs_diff` against `Lambda_B Lambda_B'`. Julia stores `rows(fit.Λ * fit.Λ')` (`tools/true_parity/campaign/run_J.jl:118` for ordinal, `tools/true_parity/campaign/run_J.jl:208` for the three data cells). `cases_for` at `tools/true_parity/campaign/write_receipts.py:237` passes the two matrices through. `orient` at `tools/true_parity/campaign/write_receipts.py:142-146` is applied only to temporal loadings (`tools/true_parity/campaign/write_receipts.py:248`).

The training linear predictor is scored for the three data cells, and not for ordinal. The checker name is `linear predictor on the training data (link scale)` at `tools/true_parity/campaign/write_receipts.py:249`. R stores `as.numeric(fit$report$eta)` at `tools/true_parity/campaign/run_R.R:137`. Spider and beetle take the `NBGroupedCovFit` branch `vec(fit.β .+ O .+ fit.Λ * Z')` with `getLV(fit, Y, X; rotate = false)` at `tools/true_parity/campaign/run_J.jl:218-221`. Fungi takes `vec(predict(fit, Y, X; type = :link))` at `tools/true_parity/campaign/run_J.jl:223`. There is no key alignment step for this quantity.

Ordinal and spider store the full series in the receipt. Fungi and beetle store the full series only up to length 1000 (`tools/true_parity/campaign/write_receipts.py:75`); their `Lambda Lambda'` and linear-predictor blocks record the maximum and point at `receipts/data/campaign/raw/<cell>_*.gz`. The checks below read those raw files. They do not refit.

## Checks against a shared normalization

Sign. Every compared series has zero sign mismatches: ordinal `Lambda Lambda'` (400) and cutpoints (40); spider beta (15), `Lambda Lambda'` (144), and the linear predictor (336); fungi and beetle beta, `Lambda Lambda'`, and the linear predictor. The worst entry on each failing series has the same sign on both sides. A flipped sign on the ordinal pair would move that entry by about 0.49. A flipped sign on the spider `sp03` diagonal would move it by about 30. A flipped sign on the beetle `sp19` by `sp37` entry would move it by about 118.

Trait order and storage order. Trait levels match across engines, and the R `LLt_trait_names` match those levels. Spider's fixed effects stay inside 0.0001 (maximum `4.9511286195569326e-05`) while the intercepts themselves differ from one another by amounts of order 1. The linear predictor, zipped trait-fast (trait fastest inside the site), reproduces the receipt maxima. Reading one side as site-fast instead produces a maximum absolute difference of 13.9184 on fungi and 54.8455 on beetle.

Rotation. `Lambda Lambda'` is unchanged by a sign flip or a swap of latent axes. The linear-predictor gaps are `6.739944245026663e-05` (spider), `0.00032827901245191526` (fungi), and `0.0008694545642065066` (beetle). The stored linear predictors run from `-19.8764715520553` to `5.40651756499366` (spider), `-11.9921269604643` to `3.85805114502722` (fungi), and `-51.0649794994249` to `7.70330775596798` (beetle). A one-sided rotation would move entries on that scale.

Link scale. Those same ranges are link-scale numbers, and the other engine sits inside 0.001 of the same range on each row. A response-scale series for these fits would be a count or a probability.

Dropped constant. Subtracting one global median from the linear-predictor difference leaves the maxima essentially unchanged: fungi `0.000328279` to `0.000328521`, beetle `0.000869455` to `0.000866942`. Subtracting a separate constant per trait leaves `0.000203669` on fungi trait `sp54` and `0.0005887` on beetle trait `sp19`. The per-trait mean gap tracks that trait's intercept gap (fungi within `3.83319e-06`, beetle within `2.62973e-05`), and the leftover is the latent part of the same predictor. On spider, the per-trait mean gap is not the intercept gap (species 1: mean linear-predictor gap `+1.091679e-06`, intercept gap `-2.891208e-05`). Removing the per-trait mean still leaves `6.35051401297361e-05` on `sp04`.

The log-likelihood gaps are not one additive constant. Julia minus R is `+2.1035339159425348e-07` (ordinal, inside `1e-06`), `+1.4573231510439655e-06` (spider), `+6.151276465971023e-08` (fungi, inside `1e-06`), and `+9.080691597773694e-06` (beetle).

The approved link-residual choice is already the comparison. `tools/true_parity/campaign/run_R.R:140-142` sets `link_residual = "none"` because the default `link_residual = "auto"` adds a link-scale residual to the diagonal. Each raw file records `LLt_from_Lambda_B_max_abs_diff = 0`, so the compared R matrix is `Lambda_B Lambda_B'`. A leftover diagonal residual would move diagonals and leave off-diagonals alone. Ordinal's only two entries above `0.0001` are the symmetric off-diagonal pair `(t01, t15)`, receipt index 14, R `0.243906750881311` and Julia `0.24403664215890872`. Its largest diagonal gap is `1.8920253896936856e-05`. Beetle's largest gap is also off-diagonal: `(sp19, sp37)`, R `-59.276468357273` and Julia `-59.28110129156818`.

A single scale factor does not remove the covariance gaps. The least-squares scale of the Julia matrix on the R matrix is `0.9999990195` (ordinal), `1.0000023777` (spider), `1.0000004583` (fungi), and `1.0000029874` (beetle). The maximum residual stays the same size as the raw maximum.

## The four gaps sit on different estimates

Ordinal fails only `Lambda Lambda'`. Log-likelihood and cutpoints are inside tolerance. The matrix is symmetric. 398 of 400 entries are inside `0.0001`. The median absolute entry difference is `2.8306665238453377e-06`.

Spider's fixed effects are inside `0.0001`. The failing covariance entry is the `sp03` diagonal, R `15.3191830196009` and Julia `15.319344747390675`. Seven of 144 entries exceed `0.0001`. The linear-predictor maximum is stored index 304, R `-19.8764715520553` and Julia `-19.87653895149775`. Those species are not the ordinal pair.

Fungi's log-likelihood is inside `1e-06`. One intercept fails, `traitsp54`, R `-5.55145011088071` and Julia `-5.551574043405938`. The covariance maximum is the `sp54` diagonal, R `7.30290967368964` and Julia `7.303384111216356`. Twenty-one unique pairs exceed `0.0001`, and each touches `sp54`, `sp47`, or `sp45`. The linear-predictor maximum is the same species, flattened index 7900, R `-11.8103781114479` and Julia `-11.810706390460352`. The three slopes are inside `0.0001`.

Beetle's numeric gaps are larger: log-likelihood `9.080691597773694e-06`, six intercepts above `0.0001`, 1188 of 4624 covariance entries above `0.0001` (605 unique pairs), and linear predictor index 1650, R `-9.7874270571224` and Julia `-9.788296511686607` (`sp19`). The relative Frobenius difference of `Lambda Lambda'` is `2.312787562271615e-05`. R `pdHess` false is a separate blocker on the Hessian flag. It is not a comparison of these point values. This note does not assign a cause to that flag.

The passing quantities, the failing coordinates, and the gap sizes differ across the four rows. The checker already uses the same sign, the same trait order, the rotation invariant, the link scale, and the `link_residual = "none"` matrix.

No row is admitted. No tolerance is changed.

no shared defect; each gap is a separate estimate difference.
