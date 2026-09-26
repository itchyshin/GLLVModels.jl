# Separated-regime record — Binomial-logit, p=20, n=500, K=2, seed=42

**Date:** 2026-09-26. **Not a receipt; not a PASS/FAIL claim.** This is a labelled record of
PR #491's discarded first run, kept because it is a real parity observation in a hard regime
(quasi-complete separation), reproduced deterministically for this review.

**Claim boundary:** this is a design-choice record, not gate-tier evidence. It exists so the
discarded run is not lost, and so the choice to redraw the intercept (see
`docs/dev-log/after-task/2026-09-25-realistic-binomial-cell.md`) is auditable rather than
asserted.

## Setup

Same `tools/core070_realistic_size_cell.jl` DGP mechanism (p=20, n=500, K=2, seed=42), using
the file's shared count-scale intercept `β_log = log.(2.0 .+ 3.0 .* rand(p))` (logit scale
0.69 to 1.61, i.e. baseline prevalence 0.66-0.83) instead of the zero-centered
`β_bin = 0.4 .* randn(p)` the receipt uses. This is an ordinary logit-scale intercept range
for real presence/absence data, not a malformed DGP; it happens to land this particular draw
in the quasi-separation regime at this size. Reproduced here by running the same
`Λ_true`/`β_log`/`Z` RNG draws the tracked script performs, then building `η` from `β_log`
instead of `β_bin` (script: `dgp_variant.jl` variant `original`, kept out-of-tree in
`~/local-scratch/review-491-fix-scratch/`, not committed — it exists only to reproduce this
one record; the tracked scripts are unchanged).

## Result

| Engine | converged | logLik | Diagnostic |
|---|---|---:|---|
| Julia `fit_binomial_gllvm` | `false` | -5292.804134629608 | none (issue #498) |
| R `gllvmTMB` 0.7.0 (frozen, `b4d5fee64`) | `convergence = 0` (reports converged) | -5292.7859682457 | "runaway trait loading" warning: t16 prevalence=0.814, max_loading=44.3, relative_loading=120, saturated_fit=1 |

logLik delta (Julia minus R): 0.018 (both engines land on the same degenerate solution; only
the flagging differs). Observed prevalence range under this DGP: 0.64 to 0.858. Max row-norm
of Λ at trait 16: 42.5 (Julia) / 44.3 (R's reported `max_loading`) — both far above the true
max row-norm of Λ_true (about 0.5 for K=2, entries ~N(0, 0.35²)).

These numbers match the independent review
(`reviews/pr-491.md` in the review lane) exactly: same prevalence (0.814), same trait (t16),
same order-of-magnitude loading blowup, same 0.02-unit logLik agreement between engines.

## What this does and does not establish

Does: shows that both engines find the same runaway-loading (Heywood/quasi-separation)
solution for this DGP at this seed, and that they report it differently — R converges with a
warning, Julia reports `converged = false` with no equivalent diagnostic. This is the same
class of disagreement raised in #498 (opened from #494's two runaway covariate fits); this
record adds one more data point and is referenced there.

Does not establish: that the count-scale intercept always separates (see the seed panel,
`docs/dev-log/core070/binomial-p20-n500-K2-seed-panel.md`: 4 of 5 additional seeds under this
same DGP converge cleanly); does not itself justify or invalidate either intercept choice on
its own (see the after-task's design-rule note for that); is not used in, and does not gate,
the T4-p6 binomial receipt.
