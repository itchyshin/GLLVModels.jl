# P1 ISDM-HEADLINE-RSZ: read-only check of the predict path

Verdict: NO_PROVEN_BUG

Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`, HEAD `5cfc67542`. Read only. No source, test, receipt, GATES.md or existing note was edited. Julia was not started. No fit was run. No tolerance was changed. Nothing was committed.

## Question

The receipt `docs/dev-log/core070/true-parity-latest/receipts/isdm/campaign/ISDM-HEADLINE-RSZ.json` records FAIL on the four `predict` cases (link, response, newdata offset 0 link, newdata offset 0 response) at 1e-6 absolute, with the logLik case (4.09e-7) and the `beta` case (1.12e-5 at 1e-4) inside their bounds. The question for this note: is there a bug in the Julia `predict` path for `IsdmFit` that, once fixed, would bring the stored comparison inside 1e-6?

## What was read

- The receipt above and the two committed raw files it names, `receipts/isdm/campaign/raw/isdm_R.json.gz` and `isdm_J.toml.gz`, plus the committed data `tools/true_parity/campaign/data/isdm.csv.gz`.
- `tools/true_parity/campaign/run_J.jl` lines 153 to 169 (the Julia isdm cell), `tools/true_parity/campaign/run_R.R` lines 159 to 167 (the R isdm cell), and `tools/true_parity/campaign/write_receipts.py` lines 250 to 257 (alignment of the predict vectors by `(cell_id, trait, source)`).
- `src/families/isdm_predict.jl`, the whole file: `_isdm_newdata_design` (lines 24 to 93) and `predict(fit::IsdmFit; ...)` (lines 127 to 163). `git diff 896a0a228 HEAD -- src/families/isdm_predict.jl` is empty, so the file at HEAD is the file the campaign ran (the receipt's `gllvmodels_commit` is `896a0a22`).

## What the stored numbers say

The analysis below used Python on the committed raw files only. No engine was run.

1. Alignment is exact. The 20000 R keys and 20000 Julia keys are the same set in the same order. Both engines report the offset the same way: `predict_link - predict_newdata_offset0_link` takes exactly two values in each file, `log(1.5) = 0.4054651081` on gbif rows and `log(0.9) = -0.1053605157` on survey rows. The offset is not the source of the gap.

2. The response scale is a pure function of the link scale in both files. Applying `exp` to gbif rows and `1 - exp(-exp(eta))` to survey rows reproduces the stored response vector to 3.6e-15 in the Julia file and 2.6e-13 in the R file. The response-scale FAIL is the link-scale gap carried through the inverse links. There is no second defect on the response scale.

3. The link gap is diffuse, not structured like a coding error. 14681 of 20000 rows differ by more than 1e-6, median 2.1e-6, maximum 7.69e-5 at `(c434, sp01, gbif)`, signed mean -5e-8. gbif and survey rows have the same median gap (1.7e-6 and 2.7e-6) and the same maximum (7.69e-5 and 7.65e-5). A dropped term, a wrong sign, or a wrong source column would give a gap of order 0.1 to 1 on an identifiable subset of rows.

4. The gap decomposes into the two parameter differences the receipt already records. Writing `eta = X b + offset + Lambda[t, :] . z[:, cell]`, and using the receipt's own `beta` vectors with the committed covariates, the fixed part `X (b_Julia - b_R)` alone exceeds 1e-6 on 15788 of 20000 rows, maximum 2.28e-5. The remainder after subtracting that fixed part is identical across the gbif and survey rows of the same cell and trait to 1.0e-14 (median 5e-15), which is exactly what a shared `Lambda z` term gives. Arranged as a 500 by 20 matrix of cells by traits, that remainder has singular values 5.0e-4, 3.7e-4, 9.0e-5, 5.5e-5 and then 9e-14: rank four, the rank of the difference of two rank-2 products `Lambda_J z_J - Lambda_R z_R`. The stored `LLt` matrices differ by 3.4e-5 (receipt case BETA passes, LLt is not a campaign case on this row).

5. The size of the gap matches the size of the logLik gap. The root mean square link gap is 6.11e-6, so `0.5 * sum(d_eta^2) = 3.73e-7`. The receipt's logLik difference is 4.09e-7, Julia above R. Two points on one likelihood surface a small step apart differ by about half the squared step in the linear predictor when the per-row curvature is near one, which is the regime here. The two numbers agreeing to about 10 percent is the fingerprint of a fit-level difference. R's recorded `max_abs_gradient` is 0.0212 after 686 iterations; Julia converged in 97. R's point is the one further from the stationary point, and Julia's logLik is the higher one.

## Why no predict bug can close this row

The stored Julia `predict_link` equals `X b_fix + offset + Lambda z` at the stored Julia estimate (`src/families/isdm_predict.jl:142` returns `fit.eta` in sample; lines 148 to 156 rebuild the same sum for newdata, and the two stored Julia vectors differ by the offset alone, item 1). Any change to that arithmetic that still returns the linear predictor at Julia's estimate leaves the `X (b_Julia - b_R)` component untouched, and that component by itself is above 1e-6 on 15788 rows using only numbers already in the receipt. The only way the comparison lands inside 1e-6 is for the two estimates to agree more closely than the campaign's own `beta` tolerance of 1e-4 allows, which is a property of the two fits, not of `predict`.

One observation on the bounds, recorded and not acted on: with `env` in `[-1.76, 1.77]`, a `beta` difference admitted at 1e-4 absolute can move the link-scale prediction by up to about 3e-4, so a predict bound of 1e-6 on the same row is tighter than the coefficient bound it depends on. The paired test for the public door (`docs/dev-log/2026-10-20-p1-f-isdm-1fo-predict.md`) uses 1e-4 for the five predict vectors and passes at 3.2e-6 to 6.6e-6 on its 120-row cell, the same diffuse pattern at a smaller size. This note does not change either bound.

## Not done

No Julia process was started, so `fit.eta` was not recomputed at R's coefficients on this dataset. The decomposition in item 4 uses the receipt's `beta` vectors and the committed covariates; the latent remainder is inferred from its structure (shared across sources within a cell, rank four), not from stored `z` vectors, which neither raw file carries. The R predict code was not deparse-checked here beyond what the receipt records (`predict.gllvmTMB_multi` identical to P1). Whether a re-run with tighter R convergence, as GATES.md line 889 already asks for, brings the four cases inside 1e-6 is not predicted by this note.

## Next step

Hold the row as it stands. If a diagnostic contract for this row is approved, the measurement that would settle it is a re-run with R's gradient below 1e-5, as the decision packet's N9 block names. A change to `src/families/isdm_predict.jl` would leave the stored comparison where it is.
