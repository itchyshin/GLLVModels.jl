# Second check of RD-SPIDER-NB2 and RD-BEETLE-NB2

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`. HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. No source file was edited. No tolerance, scoreboard verdict, GATES ledger, or receipt was edited. Nothing was committed, pushed, or merged. Totoro was not used. The full suite was not run. The beetle pathology split was not run. The grouped-NB2 Laplace objective was not evaluated again.

RD-SPIDER-NB2: NO_NEW_BUG

RD-BEETLE-NB2: BLOCKED_BY_R_PDHESS

## What this pass checked

The first overnight note already evaluated `nb_grouped_marginal_loglik_laplace` at both stored parameter sets. At R's parameters the objective matches R's log-likelihood to `8.050e-10`. The receipt log-likelihood gap of `1.457e-6` is that same objective at the two stored points. Those two numbers are cited from that note. They were not recomputed here.

This pass read the grouped-NB2 fitter for a different defect in dispersion grouping, covariate column order, or species order. The stored spider receipt and the raw engine files were used only to see the order of names and the length of the dispersion vector.

## Dispersion grouping

`fit_nb_gllvm_grouped_cov` defaults `group` to `collect(1:p)`. It then sets `labels = sort(unique(group))` and `gidx[t] = findfirst(==(group[t]), labels)`. For `1:p` that map is the identity: species `t` uses `r_group[t]`. The formula call does not pass another `group`.

The raw Julia file stores `dispersion_phi` with length 12 and `dispersion_boundary` true at positions 1, 5, and 11. The receipt's engine warning names the same group indices, `[1, 5, 11]`. Those are `sp01`, `sp05`, and `sp11`, one dispersion per species. The fitter did not collapse species into fewer groups, and it did not store `r_group` in a sorted-label order different from species order.

## Covariate column order

`_build_site_modelmatrix` drops the intercept `ConstantTerm` and keeps the remaining formula terms in the order written. For `y ~ 1 + ConWate + BareSand + CovMoss` the design columns are `ConWate`, `BareSand`, `CovMoss`. Inside the fitter, `_build_offset` multiplies column `k` by `γ[k]`. Spider passes no `γ_fixed` mask, so `_slice_fixed_X` keeps every column in that order.

The receipt beta block has 15 entries: twelve intercepts, then three slopes. R's `beta_names` end in `ConWate`, `BareSand`, `CovMoss`. Julia's terms are `beta[1]` through `beta[12]`, then `gamma[1]`, `gamma[2]`, `gamma[3]`. The last three pairs are:

| Slope | R | Julia | Absolute difference |
| --- | ---: | ---: | ---: |
| 1 | -0.00801051230366657 | -0.008009850371207194 | 6.62e-10 |
| 2 | -0.629067181065095 | -0.6290671485852394 | 3.25e-8 |
| 3 | -1.20295756144915 | -1.2029578152485936 | 2.54e-7 |

Those three slopes differ from one another by tenths. A swapped covariate column would miss the existing `0.0001` beta bar. The stored block is inside that bar (`max_abs_diff` `4.951e-5`).

## Species order

Both raw files list traits `sp01` through `sp12` in that order. The fitter stores `β` as `θ[1:p]` in the row order of `Y` and does not sort species. The twelve intercepts in the receipt agree to a maximum absolute difference of `4.951e-5`, and they are distinct (about `-6.55` to `2.60`). A permutation of species would move an intercept by an amount on the scale of the intercepts.

The `Lambda Lambda'` gap of `0.0001617` and the linear-predictor gap of `6.740e-5` stay the parameter gap between the two stored points. Reordering species or columns cannot pull a scalar log-likelihood, a matched diagonal entry, or a matched linear predictor inside bars that those aligned values already miss.

## RD-BEETLE-NB2

`RD-BEETLE-NB2.json` records `pdHess` false and `pass_rule.legs.R_pdHess_true` false. The campaign binds a row only when that leg is true, so the row stays unbound. The receipt data sha256 is `1d7603ddceac61ea7179ed9519cb87bc5b7557ec1dd30f9267bd6224cd952c9b`. The first overnight note recorded a local rebuild hash `5ea37a22fab9ca2fbb599e18e8fa28ed7477675f73db8add3b6dc0df0419f545`. This pass did not open that file and did not evaluate the beetle objective. The row is not admitted.

## Code

No source change. Grouping, covariate column order, and species order on the spider grouped-NB2 path already match the receipt's aligned comparison. There is no formula correction in those three places that would bring the three spider gaps inside the current bars.
