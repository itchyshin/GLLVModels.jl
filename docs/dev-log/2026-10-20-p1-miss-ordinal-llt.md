# ORDINAL-LOGIT-RSZ LLt miss

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Diagnosis only. ORDINAL-LOGIT-RSZ is not admitted. The tolerance 0.0001 is unchanged.

Label: **REAL_DIFFERENCE**.

## The two numbers

`maxabs` in `tools/true_parity/campaign/write_receipts.py:88` subtracts the stored matrices element by element:

```python
def maxabs(a, b): return max(abs(x - y) for x, y in zip(a, b))
```

The maximum is one symmetric pair of `Lambda Lambda'`, traits `t01` and `t15`, row-major index 14 and its transpose index 280:

| Side | Entry (t01, t15) |
| --- | ---: |
| R `r_value` | 0.243906750881311 |
| Julia `julia_value` | 0.24403664215890872 |

`abs(0.243906750881311 - 0.24403664215890872) = 0.00012989127759771568`.

`reasons` at `tools/true_parity/campaign/write_receipts.py:343` prints that maximum as `LLt: abs difference 0.000129891 > tolerance 0.0001`. Those are the only 2 of 400 entries above 0.0001. The median absolute entry difference is 2.8306665238453377e-6. Both stored matrices are symmetric, and every entry has the same sign on both sides.

## What was compared

R writes `LLt` at `tools/true_parity/campaign/run_R.R:143-145`: `extract_Sigma(fit, level = "unit", part = "shared", link_residual = "none")$Sigma`. The same lines set `LLt_from_Lambda_B_max_abs_diff` to `max(abs(Lambda_B %*% t(Lambda_B) - S))`. The raw file `ordinal_R.json.gz` records that check as 0, so the compared R matrix is `Lambda_B Lambda_B'`. Trait names are `t01` through `t20`.

Julia writes the same object at `tools/true_parity/campaign/run_J.jl:118`: `rows(fit.Λ * fit.Λ')`. `rows` at `tools/true_parity/campaign/run_J.jl:39` stores one matrix row per TOML row. Julia `trait_levels` are `t01` through `t20` in that order. The raw Julia value at row 1, column 15 is 0.24403664215890872, the same number the receipt compares.

`cases_for` at `tools/true_parity/campaign/write_receipts.py:237` passes `R["LLt"]` and `J["LLt"]` through with no sign flip. `orient` at `tools/true_parity/campaign/write_receipts.py:142-146` is applied only to temporal loadings. The ordinal quantity uses the rule at `tools/true_parity/campaign/write_receipts.py:155`: 1e-4 absolute per element of `Lambda Lambda'`. The receipt `r_value` matches the raw R matrix exactly, including 0.243906750881311 at this entry.

## Why the capture still leaves the row failed

The 2026-10-20 capture in `docs/dev-log/2026-10-20-p1-ordinal-logit-rsz-capture.md` rebuilt the returned Julia objective. Reconstruction absolute error was 0. The maximum absolute gradient was 9.367795428261161e-6. The captured log-likelihood was -11506.928561299756, beside the receipt Julia log-likelihood -11506.928561299746. That capture stops before it writes `LLt`. It shows that the returned Julia point is a point of Julia's own objective.

The receipt already treats the log-likelihood as inside tolerance: R -11506.9285615101, Julia -11506.928561299746, absolute difference 2.1035339159425348e-7, tolerance 1e-6. Cutpoints are inside 0.0001 (maximum absolute difference 5.585089662085707e-6). R convergence is 0 and `pdHess` is true, so R returned a usable matrix. The failing check is the cross-engine covariance entry above.

`Lambda Lambda'` is already the rotation-and-sign invariant named in the tolerance rule. The trait order matches, the sign pattern matches, and a sign change or a swap of latent axes leaves this product unchanged. The R matrix equals `Lambda_B Lambda_B'` with recorded difference 0, and the Julia matrix is `fit.Λ * fit.Λ'`, in the same row order. The remaining gap is the value of that covariance entry.

ORDINAL-LOGIT-RSZ is not admitted.
