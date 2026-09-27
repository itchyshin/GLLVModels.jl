# After-task: iSDM unit-level unique variance, R's default `latent(..., unique = TRUE)` (ISDM-PSI, 2026-09-27)

## 1. Goal

Maintainer decision D-301: the first iSDM PR (#546) keeps its refusal of R's `latent()` default; this follow-up ports it. R's default adds gllvmTMB's `theta_diag_B`, a length-p log-SD vector, with `s_B(t, s) ~ N(0, exp(theta_diag_B[t])^2)` added to eta for every (trait, unit) and integrated by Laplace. Acceptance: an identified fixture with at least three traits, R receipts at pin P1 (logLik at both optima, cross-objective both ways, b_fix by name, Λ Λ', exp(theta_diag_B), predict, convergence), `unique = FALSE` unchanged.

## 2. Implemented

- `src/families/isdm_formula.jl`: `latent()` reads `unique` (default TRUE, literal TRUE/FALSE only); the refusal is gone.
- `src/families/isdm_table.jl`: `IsdmTable.unique`; an internal guard for R's `diag_B_skip` gate, which cannot fire on this door.
- `src/families/isdm_laplace.jl`: `_isdm_augment` builds `[Λ diag(exp(theta_diag_B))]`; `isdm_marginal_loglik_laplace(...; theta_diag_B)`.
- `src/families/isdm_grad.jl`: packed `[b; pack(Λ); theta_diag_B]` (R's `opt$par` order, measured); the one-step gradient over it.
- `src/families/isdm_fit.jl`: `IsdmFit.unique`, `.theta_diag_B`, `.s_B`; start `theta_diag_B = log(1)` as R.
- `src/families/isdm_predict.jl`: `s_B` re-added on seen units, not on unseen ones (R/methods-gllvmTMB.R, diag_B block).
- Fixtures and receipts: `test/fixtures/isdm/export_psi_fixtures.R`, `isdm_psi4.csv`, `r_values_psi_p1.toml`, `export_julia_psi_estimates.jl`, `julia_estimates_psi_p1.toml`; tests `test/parity/isdm_unique_cases.jl` and a new testset in `test/test_isdm.jl`.

## 3a. Decisions and Rejected Alternatives

- The augmented-loadings reading was checked against R before coding: `src/gllvmTMB.cpp:1211-1212, 1968-1984, 3124` and a real P1 fit (random names `z_B`, `s_B`; `par` order). It is exact, not an approximation: the Laplace value is invariant to the linear change of variables `s_B = diag(exp θ) u`, and a test compares against a direct Laplace in the `s_B` scale.
- Not ported: R's per-trait `diag_B_skip` (cannot fire: every trait has count rows) and the Gaussian-only exact convolution (no Gaussian arm).
- Rejected: fixing the mode-search floor noise (section 10) here, because it would change `unique = FALSE` fits, which must stay bit-identical.
- Seed for `isdm_psi4.csv`: 20260929, the first of four candidate seeds whose R fit put every unique SD in the interior; recorded in the generating script.

## 4. Files Touched

`src/families/isdm_{formula,table,laplace,grad,fit,predict}.jl`; `test/test_isdm.jl`; `test/parity/isdm_unique_cases.jl` (new); `test/fixtures/isdm/` (five new files, `isdm_fixture_io.jl` extended); `CHANGELOG.md`; `docs/src/api.md`; `docs/dev-log/check-log.md`; `docs/dev-log/decisions/2026-09-27-isdm-port-provenance.md`; this report.

## 5. Checks Run

- Julia 1.10.12 and 1.13.0, each file on its own: `test/test_isdm.jl` 183/183; `test/parity/isdm_cases.jl` admission 41/41, paired 219/219; `test/parity/isdm_unique_cases.jl` 68/68. Full suite not run (brief).
- `unique = FALSE` bit-identity: every output of the four #546 paired fits (`b_fix`, `Λ`, `zhat`, `eta`, `loglik`, iterations, predict) `isequal` between #546 head `983c9979b` and this branch.
- R receipts (psi4): logLik door -1197.2250235201839 (code 0), polished -1197.2250234858263 (code 0); Julia -1197.225023485898. Cross-objective gaps 4.4e-10 and 1.1e-11. Polished: b_fix rel 1.3e-6, Λ Λ' 2.3e-7, sd_B 4.2e-7.

## 6. Tests of the Tests

- The first finite-difference check failed at one step size on one coordinate (0.0012 against a 8e-5 bound). Traced to one cell whose mode was accepted with a residual step of 5.9e-8, moving its value by 2.4e-8; the gradient itself agreed at h = 1e-4 and 1e-6. The reference now polishes modes with ForwardDiff Newton steps rather than widening the bound.
- The augmentation identity (unique SDs at exp(-40) reproduce the loadings-only marginal) and the direct `s_B`-scale Laplace would both fail on a wrong augmentation.

## 7a. Issue Ledger

No GitHub issue opened. D-301 (maintainer decision log) is the scope record.

## 8. Consistency Audit

CHANGELOG, the `fit_isdm_gllvm` and `predict` docstrings, `docs/src/api.md` and the provenance note now describe `unique = TRUE` as supported; the #546 CHANGELOG sentence about the refusal was amended in place. No "issue #N" in docstrings.

## 9. What Did Not Go Smoothly

- A shared session scratchpad already held another lane's R library; a fresh P1 library was built in a subfolder (install 2 min 16 s).
- The FD glitch above cost one diagnosis round.

## 10. Known Residuals

- Mode-search floor noise: `_isdm_cell_mode` (copy of the `_mixed_laplace_mode` rule) accepts a mode at the floating-point floor with a residual step near sqrt(eps)(1 + |z|); the log-determinant is first-order in it, so a cell value can move by ~2.4e-8. Below every receipt tolerance, affects `unique = FALSE` equally, not fixed here.
- Two-trait default fits: theta_diag_B runs to the boundary (not identified); asserted only through logLik, cross-objective, b_fix and Λ Λ'.
- Full suite and Aqua/JET not run in this lane.

## 11. Team Learning

When a model component enters additively with identity loadings, porting it as augmented loadings reuses a verified kernel unchanged; check invariance of the approximation under the reparameterisation, then test it directly.

## 12. Cross-Product Coverage

Covers: `unique = TRUE` on the iSDM door with K = 1, Poisson-log count and Bernoulli-cloglog detection arms, offsets, source-reporting-rate terms, in-sample and `newdata` predict (seen and unseen units), the one-step gradient. This arc does NOT cover: K >= 2 with `unique = TRUE` against R (the kernel accepts it; no R fixture), source observation formulas combined with `unique = TRUE` against R, `se_fit`, R's `diag_B_skip` gate (unreachable on this door), `unique(..., common = TRUE)`, spatial or AGHQ routes, and the `gllvm()` formula door.
