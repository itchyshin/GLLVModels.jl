# animal_latent overnight check

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Verdict: **WRONG_MODEL**. No source file was edited. No tolerance was changed. `GATES.md` and the receipts were left as they stood. The row stays unbound.

## The row

`namespace/export/animal_latent` is the slope contract. The case map records empty `executable_case_ids`, a null disposition, and three P0 ids: `SLOPE-ANIMAL-LAT-DEFAULT`, `SLOPE-ANIMAL-LAT-NOUNIQUE`, and `SLOPE-ANIMAL-LAT-RANK4` (`docs/dev-log/core070/true-parity-latest/case-map-namespace.json`). The scoreboard line is NOT-MEASURED, with no receipt cited (`docs/dev-log/core070/true-parity-latest/scoreboard.md`). The maintainer ruling of 2026-10-05 says "animal_latent: leave unbound" and later repeats that the row stays unbound (`docs/dev-log/core070/true-parity-latest/GATES.md`).

Those P0 calls are random slopes. `SLOPE-ANIMAL-LAT-NOUNIQUE` and `SLOPE-ANIMAL-LAT-DEFAULT` prepare `animal_latent(1 + x | species, A = C, d = 2)` with free `theta_rr_phy_slope` (10 coordinates at p = 3) and random effect `g_phy_slope`. Both list `native_call` as UNRESOLVED and `native_status` as UNPAID (`docs/dev-log/core070/slopes-required-case-plan.json`). `SLOPE-ANIMAL-LAT-RANK4` is the same formula at `d = 4`. R rejects it because the rank exceeds the number of traits. Its native call is also UNRESOLVED.

The slope contract stores two separate p-by-d loading blocks. With relatedness C, the covariance of two observations is `C[g_o, g_v] * (U0[t_o, t_v] + x_o * x_v * U1[t_o, t_v])`, and `Uj = Lj Lj'`. There is no intercept-slope cross term (`docs/dev-log/core070/slopes-input-contract.md`).

## The Julia function

`fit_animal_latent_gllvm(Y, A, d)` in `src/animal_latent.jl` takes a trait-by-site matrix and a p-by-p relatedness matrix, with `p = size(Y, 1)`. For `d < p` it calls `fit_gaussian_gllvm(Y; K = 1, K_phy = d, Σ_phy = relatedness_cov(A))`. For `d == p` it delegates to `fit_animal_dep_gllvm`. The arguments contain no covariate, no slope column, and no `g_phy_slope` block.

The docstring calls it a Gaussian matrix fitter and says `@formula` `animal_latent()` syntax is absent. The accepted identity of 2026-09-14 locks the same matrix API and places pedigree parsing and slopes outside the slice (`docs/dev-log/decisions/2026-09-14-animal-latent-identity.md`). `src/GLLVModels.jl` includes the file as an animal-by-latent matrix fitter with no formula sugar. A search of `src/` finds the symbol only in that file, a comment in `animal_dep.jl`, and the export list.

## R's slope route

In gllvmTMB, an augmented bar `animal_latent(1 + x | id)` is detected in `R/brms-sugar.R` and rewritten to `phylo_rr` with `.latent_slope = TRUE` and `.animal_source = TRUE`. The Gaussian test `tests/testthat/test-animal-latent-slope-gaussian.R` draws separate intercept and slope scores, `g0` and `g1`, each distributed as `N(0, A)`, builds `u_int = g0 Lam0'` and `u_slp = g1 Lam1'`, and sets the mean to `beta[trait] + u_int + u_slp * x`. The header states that the two columns are block-diagonal, with no intercept-slope cross-column correlation.

That mean and those two loading matrices are absent from `fit_animal_latent_gllvm`. The function's third argument is a single rank `d` on one relatedness trait block. Pointing the same function at `animal_latent(1 + x | species)` would replace the estimand. The slope case plan already records that no native call is dispatch-ready.

## The passing animal_latent receipt

`CORE070-FIT-INPUT-ANIMAL-LATENT-NATIVE-MODEL` passes on a different formula. R fits `value ~ 0 + trait + animal_latent(species, A = A, d = 1)`. Julia fits `fit_gaussian_sources` with one `SourceCovariance` on the id axis. The receipt reports a log-likelihood absolute difference of 3.122732294968955e-09 and a coefficient absolute difference of 5.681706146676291e-07, both under 0.0001 (`docs/dev-log/core070/true-parity-latest/receipts/fit-input/cases/CORE070-FIT-INPUT-ANIMAL-LATENT-NATIVE-MODEL.json`). The formula has no `1 + x`. The Julia call is `fit_gaussian_sources`. The namespace case map does not list this receipt under `namespace/export/animal_latent`.

## What was left alone

No Julia source, test, tolerance, gate file, or receipt was edited. No new fitter was added. No comparison was run, because the slope cases have no Julia native call to execute. HEAD remains `5cfc6754212588222779ccfd66ad7000375cf8d5`.
