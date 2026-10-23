# bootstrap_Sigma named covariance

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Local reading only. No commit, push, merge, Totoro job, full suite, or bootstrap study. `GATES.md` and receipts were not edited. No tolerance was changed. No source file was edited. The row is not admitted.

Verdict: **NEEDS_NEW_STUDY**.

## What was required

The later target, if any, is R's named random-effect covariance. An existing comparison that already passes would close this check. A failure that is a proven bug in the existing function against that named covariance would be fixed in place. A match that needs a new study or a new estimand stays unbuilt.

## Receipt

`docs/dev-log/core070/true-parity-latest/receipts/namespace/cases/CORE070-NAMESPACE-BOOTSTRAP-SIGMA-NATIVE.json` records `result` PASS and verifier category `registration_twin`. Its meaning is a Tier 0 registration check: the R export is in the pinned NAMESPACE and the named Julia symbol exists. The Julia symbol in that check is `bootstrap_ci`, with first method `src/confint_bootstrap.jl:258`. The R pin is `9539352f66f2db2cc26b1c393e67212a359b60c9`, definition hint line 202 of `R/bootstrap-sigma.R`. The scoreboard row `namespace/export/bootstrap_Sigma` stays REGISTRATION-ONLY and not bound.

## What the Julia function computes

`bootstrap_Sigma` (`src/confint_derived_wald.jl`, lines 913-944) accepts only `level = :unit`. Each upper-triangle entry calls `bootstrap_ci_derived` on `sigma_y_site(fb)[i, j]`. `sigma_y_site` (`src/confint_derived.jl`, lines 154-199) is `Λ_B Λ_B' + Λ_W Λ_W' + diag(σ²_B + σ²_W + σ²_eps)`. The phylogenetic block is excluded. The docstring names this the site-level trait covariance.

The named unit covariance is a different function. `extract_Sigma(fit; level = :unit, part = :total)` (`src/extractors.jl`, lines 105-106 and 163-165) uses `_sigma_unit` (lines 63-69): `Λ Λ'` plus `diag(σ²_B)` when a unit diagonal is present. Residual variance is on the observation tier. `_sigma_unit_obs` (lines 72-83) adds `σ_eps^2 I`.

## What R computes

The receipt's pinned `bootstrap_Sigma` simulates, refits, and extracts named tier summaries. The later `gllvmTMB` copy still stores each kept level as `extract_Sigma(fit, level = lvl, part = "total")$Sigma` inside `.extract_summaries` (`R/bootstrap-sigma.R`). Proposal B records the pinned unit target as the between-unit component `Lambda_B Lambda_B'`, plus `diag(Psi_B)` when that tier is fitted (`/private/tmp/gllvm-p1-current-main/docs/dev-log/plans/2026-10-07-p1-namespace-mc-contracts.md`, Proposal B). On a Gaussian fit, R's `link_residual` option leaves that unit matrix without the observation residual.

## Existing comparisons

The registration receipt passes as a name match. It records no covariance entries and no bootstrap endpoints.

`test/test_se_machinery.jl` (the set starting at line 214) checks that `bootstrap_Sigma` agrees with a direct `bootstrap_ci_derived` call on `sigma_y_site(fb)[1, 1]` at the same seed, and that `level = :unit_obs` throws. That check compares the public wrapper with the site-total functional it already calls. It was left unrun. It has no R unit-tier `Sigma` on either side.

`test/parity/test_gaussian_parity.jl` (lines 85-86) takes R `extract_Sigma(..., part = "shared")` and adds `diag(sigma_eps^2)` before the comparison. That constructed matrix is the site total. It is a log-likelihood parity helper, and it was left unrun.

No committed numeric comparison of `bootstrap_Sigma` against the named random-effect covariance was found. The source audit states that no bootstrap seed or endpoint comparison was judged for this row (`docs/dev-log/audits/2026-10-19-p1-thirteen-row-source-audit.md`, the `namespace/export/bootstrap_Sigma` section).

## Why the match is a new estimand and a new study

On a Gaussian fit with no unit-unique term, the public wrapper's diagonal includes `σ_eps^2` and R's named unit covariance omits it. The two diagonals differ by that residual by construction. The wrapper does what its docstring says, so the gap is an estimand difference. Redirecting `bootstrap_Sigma` to `extract_Sigma(; level = :unit)` would change the returned quantity.

Off-diagonals can agree on a fixture with no within-unit loadings. That limited comparison is one of three unsigned Proposal B alternatives. The other two are a full unit-tier bootstrap of `extract_Sigma` and a site-total comparison that adds residual variance on the R side. The 2026-10-20 signature draft leaves all three unsigned (`docs/dev-log/plans/2026-10-20-p1-thirteen-row-signature-drafts.md`, section 10). N4 still requires the Monte Carlo rule to be fixed before the run it judges (`docs/dev-log/core070/true-parity-latest/GATES.md`, the N4 section). No such rule was signed here, and no replicates were drawn.

## What this leaves in place

HEAD remains `5cfc6754212588222779ccfd66ad7000375cf8d5`. `CORE070-NAMESPACE-BOOTSTRAP-SIGMA-NATIVE` stays PASS as `registration_twin`. The scoreboard row stays REGISTRATION-ONLY and not bound. This note does not bind the row, move a scoreboard count, or mark C2, C3, C4, or X2 MET.
