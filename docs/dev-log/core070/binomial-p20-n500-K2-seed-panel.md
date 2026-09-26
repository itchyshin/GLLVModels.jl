# Seed panel — Binomial-logit, p=20, n=500, K=2, two intercept DGPs

**Date:** 2026-09-26. Not a receipt; not gate-tier evidence. Answers one question raised in
review of PR #491: was seed 42 picked because it passes under the zero-centered intercept?

**Estimate before running:** about 1 min Julia + 15s R per fit at this size (10 fits total,
5 seeds x 2 DGPs), so under 15 minutes wall; measured total 181s (Julia) + 59s (R) = 240s,
well inside the 30-minute stop condition.

**Method:** seeds {1,2,3,4,5} (seed 42 excluded — it is the receipt seed and is reported
separately in `binomial-p20-n500-K2-separated-regime-seed42.md` and the T4-p6 receipt), same
p=20/n=500/K=2 shape, both DGPs from `tools/core070_realistic_size_cell.jl`:
count-scale intercept (`β_log`, "original") and zero-centered intercept (`β_bin = 0.4*randn`,
"new", the receipt's DGP). Fit with `fit_binomial_gllvm` (Julia) and frozen gllvmTMB 0.7.0
(R, `b4d5fee64`), same formula as the tracked R script. Script: `dgp_variant.jl`, kept
out-of-tree in `~/local-scratch/review-491-fix-scratch/`, not committed; the tracked cell
scripts are unchanged.

## Results

### Count-scale intercept (`β_log`, the discarded first-run DGP)

| seed | Julia converged | Julia logLik | max‖Λ_i‖ | R convergence | R warning |
|---:|---|---:|---:|---|---|
| 1 | true | -5411.755720 | 1.32 | 0 | none |
| 2 | true | -5392.192087 | 1.10 | 0 | none |
| 3 | true | -5498.345363 | 1.10 | 0 | none |
| 4 | true | -5316.760492 | 1.37 | 0 | none |
| 5 | **false** | -5442.516097 | **46.2** | 0 | **runaway loading** (t19, prevalence 0.854, max_loading=46.1, relative_loading=159) |
| 42 (receipt) | **false** | -5292.804135 | **42.5** | 0 | **runaway loading** (t16, prevalence 0.814, max_loading=44.3, relative_loading=120) |

Separates 2 of 6 seeds (33%): 42 and 5. Both engines agree on which seeds separate; R always
reports `convergence = 0` (with a warning on the separating seeds), Julia reports
`converged = false` only on the separating seeds (see #498 for the flag-disagreement itself).

### Zero-centered intercept (`β_bin = 0.4*randn`, the receipt DGP)

| seed | Julia converged | Julia logLik | max‖Λ_i‖ | R convergence | R warning |
|---:|---|---:|---:|---|---|
| 1 | true | -6590.517744 | 1.19 | 0 | none |
| 2 | true | -6660.941460 | 0.80 | 0 | none |
| 3 | true | -6644.712052 | 1.27 | 0 | none |
| 4 | true | -6761.098526 | 1.25 | 0 | none |
| 5 | true | -6720.953341 | 0.90 | 0 | none |
| 42 (receipt) | true | -6744.539487 | 0.81 | 0 | none |

Separates 0 of 6 seeds. Julia and R logLik agree to 5-6 decimal places on every seed (same
pattern as the receipt's 6.6e-08 delta).

## Reading

Seed 42 was not selected to pass: under the zero-centered intercept, all 6 seeds tried
(1-5 plus 42) converge cleanly with no separation on either engine — the DGP is well away
from the quasi-separation boundary at this size, not narrowly tuned to one seed. Under the
count-scale intercept, separation is seed-dependent (2 of 6 here), consistent with the
after-task's design-rule note that this DGP was not "a bug", just an intercept range that
sometimes lands there.
