# iSDM port: provenance of code and fixtures taken from gllvmTMB (2026-09-27)

Scope: arc A1b, the integrated species distribution model twin of gllvmTMB's
public door `gllvmTMB(..., family = isdm_sources(...))` at pin P1
(`9539352f66f2db2cc26b1c393e67212a359b60c9`). Design spec: PR #525
(`docs/design/isdm-port-spec.md`); scope decisions: vault D-296. gllvmTMB is a
read-only reference; it was read with `git show 9539352f6:<path>` and installed
into a temporary library from a detached worktree. No gllvmTMB file was edited.

## What was carried over, and how

| Julia | gllvmTMB at P1 | kind |
|---|---|---|
| `_isdm_log_cloglog_p`, `_isdm_dbinom_cloglog` (`src/families/isdm_laplace.jl`) | `gll_log_cloglog_p`, `gll_dbinom_cloglog` (`src/gllvmTMB_cloglog.h:14-57`) | algorithm translated branch for branch (same cut points -20 and 700, same series, same operation order) so the values agree bit for bit; `CppAD::CondExp*` became `ifelse` |
| `_isdm_declared_core`, `_isdm_assert_trait_scale`, `_isdm_prepare_offset`, `_isdm_assert_observed_arms`, `_isdm_observation_design` (`src/families/isdm_table.jl`) | `R/isdm-sources.R:178-277, 412-477`; `R/fit-multi.R:293-318, 1432-1501, 3462-3480`; `R/offset.R:108-195` | logic re-implemented in Julia; error messages keep R's cli-rendered first line |
| `isdm_sources`, `isdm_source` (`src/families/isdm_sources.jl`) | `R/isdm-sources.R:15-172` | constructor checks re-implemented; R's first lines kept |
| `predict(::IsdmFit)` (`src/families/isdm_predict.jl`) | `R/methods-gllvmTMB.R:2786-3220` | semantics re-implemented |
| CSV fixtures (`test/fixtures/isdm/*.csv`) | fixture generators in `tests/testthat/test-isdm-predict.R:6-34`, `test-isdm-multisource.R:4-31`, `test-isdm-source-formula.R:23-58, 171-181` | generators copied into `test/fixtures/isdm/export_p1_fixtures.R`, run once in R at P1, outputs sha256-pinned |
| `test/fixtures/isdm/r_values_p1.toml`, `cloglog_grid_p1.csv`, `admission_p1.toml` | the P1 package, fitted and evaluated | recorded numbers, with the P1 SHA and the sha256 of the R source files read |

The damped mode search copies the convergence rule of this repository's own
`_mixed_laplace_mode` (`src/families/mixed.jl`, after #514), not gllvmTMB code.

## Where R at P1 differs from the spec (recorded, not silently resolved)

- R's `latent()` defaults to `unique = TRUE` (`R/brms-sugar.R:607`), adding a
  per-trait unit-level unique variance (`theta_diag_B`). The spec's model is
  loadings-only. The first PR required `unique = FALSE` and refused the default;
  the paired R fits of `export_p1_fixtures.R` use `unique = FALSE`. Maintainer
  decision D-301 (2026-09-27): port it as a follow-up with a fixture of at least
  three traits. Ported; see the next section.

## The unit-level unique variance (`latent(..., unique = TRUE)`)

Read from gllvmTMB at P1: `PARAMETER_VECTOR(theta_diag_B)` and
`PARAMETER_MATRIX(s_B)` (`src/gllvmTMB.cpp:1211-1212`); the density
`nll -= dnorm(s_B(t, s), 0, exp(theta_diag_B(t)), true)` for every trait and
unit (`1968-1984`); `eta(o) += s_B(t, s)` (`3124`); `s_B` is in TMB's `random`
vector with `z_B`. Measured from a real P1 fit: `names(fit$opt$par)` is
`b_fix..., theta_rr_B..., theta_diag_B...` and the random names are `z_B`, `s_B`;
R starts `theta_diag_B` at `log(1)` on non-Gaussian fits
(`R/fit-multi.R:5765-5869`).

Parameterisation in Julia: with `s_B = diag(exp(theta_diag_B)) u`,
`u ~ N(0, I_p)`, the model is the loadings-only kernel with
`Λ_aug = [Λ diag(exp(theta_diag_B))]` (`p x (K + p)`) and
`z_aug = [z; u] ~ N(0, I_{K+p})` (`_isdm_augment`, `src/families/isdm_laplace.jl`).
The Laplace approximation is invariant to this linear change of variables (the
Jacobian cancels the change in the Hessian determinant), so the per-cell value
equals R's joint Laplace over `(z_B, s_B)`; `test/test_isdm.jl` checks it against
a direct Laplace in the `s_B` scale. Packed `θ = [b_fix; pack(Λ); theta_diag_B]`,
R's order. Not ported, because it cannot fire on this door: R's per-trait gate
that maps the default Psi off for a trait whose rows are all single-trial
Bernoulli (`R/fit-multi.R:7133-7178`, `diag_B_skip`); every declaration has a
count arm and every trait carries every arm (an internal guard in `isdm_table`
errors if that ever changes). Nor the Gaussian-only exact convolution
(`integrate_gaussian_diag_B`): the door has no Gaussian arm.

Fixtures: `test/fixtures/isdm/export_psi_fixtures.R` generates `isdm_psi4.csv`
(4 traits, 80 cells, 3 sources; seed 20260929, the first of 20260927-20260930
whose default R fit put every unique SD in the interior) once at P1, and records
R's default-formula fits of it and of the two-trait `isdm_predict.csv` and
`isdm_ms3.csv` in `r_values_psi_p1.toml` (the latter reproduce the
`default_unique_*` values of `r_values_p1.toml` exactly). Julia's estimates:
`export_julia_psi_estimates.jl`, `julia_estimates_psi_p1.toml`.
- The spec's `gllvm()` door edits `src/formula.jl`, another lane's file; not
  added. The entry point is `fit_isdm_gllvm`.

## Recorded R optima and the polish diagnostic

`r_values_p1.toml` also records a polished R optimum: nlminb restarted from the
door's optimum on the same TMB objective (rel.tol 1e-14). It certified
convergence (code 0) on `ms3` and `srcform_mixed`. It did not certify on the
other two: on `srcform_pois` it moved (max|gradient| 6.1e-4 to 1.0e-5) and then
stopped with code 1; on `predict` it did not move at all (code 1). The paired
test labels `srcform_pois` "polish did not certify" and makes no polished
comparison for `predict`, whose door optimum already passes rel 1e-4.

Factor levels are ordered by Julia byte order (`sort(unique(...))`), not R's
locale collation; a mixed-case observation factor can take a different
reference level, which fails loudly because coefficients pair by name.
