## 2026-11-01: has_diag with K_W == 0 vs R's unit-tier Sigma, replicated rows per cell

Docs-only note. No source, test, row status, scoreboard, receipt, scope or tolerance change. Base origin/main 3c94a76f7, oracle R gllvmTMB 0.7.1 (Totoro library `~/hsq_work/true-parity-campaign-20261002/Rlib`; the bootstrap-sigma check log verified its `bootstrap-sigma.R` against the pin, but this run did not re-verify the library against 9539352f).

Question: for a Gaussian `has_diag` fit with `K_W == 0`, is `_r_tier_total(fit, :unit)` (= `sigma_y_site(fit)`, `src/extractors.jl:276-288`) equal to R's unit-tier `extract_Sigma(level = "unit", part = "total")`? Earlier evidence covered one row per (site, trait) only (`2026-11-01-bootstrap-sigma-vs-r.md`, follow-up 5).

Result: **only when R's `sigma_eps^2` is about 0, i.e. one row per cell.** With replicated rows R identifies `sigma_eps^2` and keeps it out of the unit Sigma, while the Julia total still contains the residual of whatever each Julia column is.

Setup: p = 6, 200 sites, K = 1, true unit Sigma = LL' + diag(s^2), true residual variance 0.25; m = 1, 3, 10 rows per cell, 3 seeds each (9 datasets). R: `value ~ 0 + trait + latent(0 + trait | site, d = 1)`, `unit = "site"`. Julia: `fit_gaussian_gllvm(Y; K = 1, has_diag = true)`, replicate rows stacked as separate columns ("stacked"), or averaged per site ("site-mean"). `GllvmFit` has no site grouping for replicates (the `fit.jl` docstring points to `fit_twolevel_gaussian`, which has no `has_diag`, `K_W == 0` form), so these are the two ways a user can feed it replicated data.

| m | R sigma_eps^2 (3 seeds) | max abs diff, stacked vs R unit | stacked vs R unit + sigma_eps^2 I | site-mean vs R unit | site-mean vs R unit + sigma_eps^2/m I |
|---|---|---|---|---|---|
| 1 | 8.2e-7 to 1.0e-6 | 2.6e-6 to 9.4e-6 | same | same | same |
| 3 | 0.250 to 0.261 | 0.258 to 0.276 | 0.0089 to 0.0173 | 0.083 to 0.087 | 6.4e-6 to 1.4e-5 |
| 10 | 0.251 to 0.253 | 0.253 to 0.262 | 0.0113 to 0.0141 | 0.0251 to 0.0253 | 6.7e-6 to 1.6e-5 |

The m = 3 and m = 10 diagonal gaps (stacked) are 0.237 to 0.277 on every trait, about R's `sigma_eps^2`. The site-mean fit equals R's unit Sigma plus `sigma_eps^2 / m` on the diagonal to 1.6e-5 or better, the same size as the one-row-per-cell agreement. The stacked off-diagonals differ from R by up to 0.016 (stacking ignores the correlation among replicates, so it is a different estimator, not a rounding gap). R: convergence code 0 and positive-definite Hessian in all nine fits; max gradient 2.5e-4 (m = 1) to 5.4e-3 (m = 10). Julia converged in all 18 fits.

Consequence for the code comment at `src/extractors.jl:271-275` and `docs/src/se-profile-machinery.md:82`: the identity "R fits the whole sum as the unit unique part" holds when R's `sigma_eps` is auto-fixed near 0 (one row per cell). It must not be read as covering replicated rows. This note changes neither text.

Not covered: unbalanced replication, K > 1 with replicates, non-Gaussian families, the TwoLevel route, tightened R convergence, one DGP shape (3 seeds per m). Raw scripts and outputs: Totoro `~/claude-hasdiag-equiv` (`run.sh`, `gen_r.R`, `jl.jl`, `out/`, `logs/`); not committed.
