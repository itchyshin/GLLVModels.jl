# Truncated-NB2 second-order cell: interior-dispersion data screen (2026-09-25)

## Why

The second-order `truncated_nbinom2` cell (`tools/core070_second_order/cells.jl`)
copied its DGP from the frozen NATIVE-12 fixture (`test/parity/test_truncated_nbinom2_parity.jl`,
seed=58, p=5, K=1, n=120) but fit Julia with `fit_truncated_nbinom2_gllvm`, which uses one
shared dispersion `r`. gllvmTMB's `truncated_nbinom2()` has no shared-dispersion mode — it
always fits per-trait `log_phi_truncnb2` (`docs/dev-log/decisions/2026-08-15-truncated-nbinom2-
identity.md`) — so the cell paired two different models. Repointing the cell to
`fit_truncated_nbinom2_gllvm_pertrait` (already exported, `src/GLLVModels.jl`) makes the
comparison like-with-like, but on the NATIVE-12 seed=58/n=120 data, one trait's dispersion sits
at the Poisson-limit boundary on **both** engines (measured on this checkout: Julia `r[5] =
2.93e9`, R `phi[5] = exp(16.331) ≈ 1.24e7`; `se=TRUE` vs `se=FALSE` makes no difference in R;
corrected 2026-09-26 after an independent review of #493 found the boundary is shared, not an
R-side artefact; see "Corrections" below), which would make any SE/CI comparison on that trait
meaningless. This screen looks for data where every trait has finite, interior dispersion on
**both** engines.

## Method

`p=5, K=1`, same intercepts (`beta = log([4, 5, 3.5, 4.5, 4])`) and loadings scale
(`Lambda = 0.2 .* parity_loadings_p5k2()[:, 1:1]`) as the NATIVE-12 design; only the seed and
`n` vary. For each candidate: draw `Y` (zero-truncated NB2, `r_true = 4.0`, rejection sampling),
fit R's `gllvmTMB::truncated_nbinom2()` with `se = TRUE` (per-trait `log_phi_truncnb2` by
construction — this family has no shared-dispersion mode) and Julia's
`fit_truncated_nbinom2_gllvm_pertrait(Y; K, hessian = :observed)`, and record each engine's
5 per-trait dispersion values, R's `pdHess` and fixed-effect precision condition number, and
whether every value stays well inside the Poisson-limit boundary (interior means `< 1e3` on
both engines here; every candidate below is well clear of that line one way or the other).

R library: the frozen `gllvmTMB` 0.7.0 build at commit `b4d5fee64`
(`GLLVM_PARITY_R_LIBS=.../gllvm-owed24-decisions-20260924/.unlazy/r-build/library`). Julia
1.10.0, `JULIA_NUM_THREADS=4`. Total screen wall time: under 3 minutes for 14 candidates across
two passes (a first pass mis-parsed R's dispersion block as "everything but `b_fix`", which
wrongly included the loadings block `theta_rr_B`; corrected to the true tail block
`par_fixed[(p+rr+1):(p+rr+p)]`, `rr = rr_theta_len(p, K) = 5`).

## Results (corrected extraction)

| seed | n | R phi (5 traits) | Julia r (5 traits) | R pdHess | cond(H)_R | interior? |
|---|---|---|---|---|---|---|
| 58 | 200 | 1.04, 0.85, 1.28, 1.08, 1.08 (b_fix-adjacent; see note) | 3.34, 5.07, 4.01, 3.12, **1.35e9** | true | 799 | no (Julia trait 5 boundary) |
| 61 | 150 | 3.09, 2.29, 9.60, 3.93, 5.69 | 3.09, 2.29, 9.60, 3.93, 5.69 | true | 187 | **yes** |
| 65 | 150 | **2.86e7**, 3.80, 2.33, 4.34, 7.01 | 3.99, 5.00, 2.33, 5.74, 7.23 | true | 187,141 | no; **not** an R artefact: R's boundary solution (logLik −1707.020) is *better* than Julia's interior one (−1707.787, `converged = true`); Julia's per-trait fitter stalls at an inferior local optimum here (#499) |
| 66 | 150 | 5.56, 3.78, **1.28e7**, 8.10, 2.78 | 5.56, 3.78, **5.56e9**, 8.10, 2.78 | true | 257,912 | no (trait 3 boundary, both) |
| 67 | 180 | **6.78e7**, 3.89, 6.57, 4.41, 3.90 | **1.47e9**, 3.89, 6.57, 4.41, 3.90 | false | 236,826 | no (trait 1 boundary, both; R not converged) |
| 68 | 180 | 8.41, 4.38, 4.05, 5.60, 6.10 | 8.41, 4.39, 4.05, 5.60, 6.10 | true | 296 | **yes** |
| 70 | 160 | 6.00, 2.77, 4.32, 3.94, 5.43 | 6.00, 2.77, 4.32, 3.94, 5.43 | true | 252 | **yes** |

(seed 60, n=150: R did not converge; seed 62, n=150: trait 1 boundary on **both** engines
(Julia `r[1] = 9.97e9`; not Julia-only, corrected 2026-09-26; see "Corrections" below); seed
63/64, n=200: one trait boundary on Julia; seed 69, n=200: sqrt(negative) in R's `cov.fixed`
diagonal, fit rejected. Not tabulated above; see raw screen output if needed.)

Note on the first (mis-extracted) row for seed=58/n=200: the printed "R phi" there actually
read the loadings block, not the dispersion block; it is left in the table only to record that
this seed/n pair was tried and is NOT the chosen fixture (Julia's own per-trait fit already
shows a boundary trait there regardless).

## Choice

**seed = 61, n = 150** — the lowest fixed-effect precision condition number of the three clean
candidates (187, vs 252 and 296; the contract's each-own-optimum SE/vcov tolerance scales by
`cond(H)_R / 1e3` above 1e3, so all three are equally unscaled here, but the lowest-conditioning
draw is the more conservative pick). R and Julia per-trait dispersion agree to 3-4 significant
figures at each trait (e.g. trait 3: R 9.598321, Julia 9.598451). Stored, hashed, and pinned at
`test/parity/fixtures/truncnb2_interior_seed61_n150.toml`
(`tools/truncnb2_parity_data_draw.jl 61 4.0 <out> 150`), because Julia 1.10 and 1.12+ draw
different numbers from the same seed
(`docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-pins.md`).

## Corrections (2026-09-26, independent review of #493)

An independent review of PR #493 refit this screen's draws against the frozen 0.7.0 R library
and found this doc, the PR body, and `tools/core070_second_order/cells.jl`'s cell note all
misattributed the boundary problem to R. Re-measured on this checkout (Julia 1.10.12) to
confirm:

- **Seed 58/n=120 (the old fixture) and seed 62/n=150: the boundary is shared, not an R
  artefact.** Both engines put the same trait at the Poisson limit and agree on logLik to
  about 1e-6. Seed 58/n=120: Julia `r[5] = 2.93e9`, R `phi[5] = exp(16.331) ≈ 1.24e7` (not the
  `9.9e7` earlier stated in this doc, which was misread off a flat ridge); Julia logLik
  −1375.391373 vs R −1375.391374. `se = TRUE` and `se = FALSE` give R identical results on
  every draw, so `se = TRUE` plays no part in the boundary either.
- **Seed 65/n=150 is a Julia-side defect, not an R boundary artefact.** R's boundary solution
  has the *higher* log-likelihood (−1707.020446) than Julia's interior answer (−1707.787472,
  which Julia reports as `converged = true`), a difference of 0.767. Tightening `g_tol` to
  1e-8 lands Julia on the same inferior point; starting from `r_init = [1e6, ...]` reaches R's
  value. This is the mirror image of #477 (NB2 grouped fits stalling at the boundary below a
  better interior point). The screen discarded this draw (because the filter above also
  selects on first-order agreement between the engines) without noticing that Julia, not R,
  was the one at fault here. **Tracked in #499**, not fixed in this doc or in #493.

Both corrections are folded into the table and footnote above. The chosen fixture (seed=61,
n=150) is unaffected: every trait is genuinely interior on both engines there.

## Not claimed

This is a data-selection screen, not a second-order parity claim. The tolerances in
`second-order-parity-contract.md` §4 are applied unchanged to the resulting cell; see
`docs/dev-log/core070/second-order-batch-out/truncated_nbinom2.json` for the receipt and
`docs/dev-log/core070/second-order-holdouts-2026-09-04.md` for the disposition.
