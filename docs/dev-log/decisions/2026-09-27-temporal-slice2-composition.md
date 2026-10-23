# Temporal source, slice 2: ordinary unit / unit_obs composition and provenance

Date: 2026-09-27. Design spec: `docs/design/temporal-port-spec.md` (draft PR #535, head
`af130f704`), its review, and the signed scope (vault decision D-300: slice 2 inside P1,
cross-source pairs deferred, wide `traits()` fenced, no bridge). R reference: gllvmTMB
`9539352f66f2db2cc26b1c393e67212a359b60c9` (P1, 0.7.1), read only.

## Model

The slice 1 marginal gains the ordinary tiers R admits beside the temporal source
(R/temporal.R:154-158):

    V = Z (K_blockdiag ⊗ Sigma_T) Z' + J_unit ∘ Sigma_B + J_unit_obs ∘ Sigma_W
        + sigma_re^2 J_g + sigma_eps^2 I

- `J_unit[o, o'] = 1` when rows share a `unit` level (gllvmTMB's `site_id`), `J_unit_obs`
  likewise for `unit_obs` (`site_species_id`), `J_g` for the `(1 | g)` group.
- `Sigma_B` / `Sigma_W` = `Lambda Lambda'` (rank > 0) + `diag(exp(2 theta_diag))`:
  `indep` is diagonal only (`theta_diag_*`), `dep` is a full lower-triangular factor only
  (`theta_rr_*`, rank `p`), `latent(d)` is both (diag only when `unique = TRUE`, the
  default of R's `latent()`, R/brms-sugar.R:607). This is the covariance of R's own
  dense oracle (tests/testthat/test-temporal-sixth-source-oracles.R:36-106, the
  `ordinary = TRUE` branch) and of the composed-simulation test
  (test-temporal-program-composed-simulation.R:58-82).
- `(1 | g)` is gllvmTMB's `re_int`: one intercept per level shared by every trait, SD
  `exp(log_sigma_re_int)`. Measured on P1: Julia's NLL at R's coordinates equals R's
  `fn` to 3e-9 on the `(1 | series)` cells.
- The likelihood is still exact (all Gaussian). It is evaluated per independent row
  block: rows are linked by series, and by unit, unit_obs and random-intercept level when
  those terms are present (union-find); without ordinary terms the blocks are the series,
  as in slice 1.

## Parameter vector (measured, not assumed)

R's `names(opt$par)` on P1 fits, all 25 receipt cells:

    b_fix, log_sigma_eps, theta_rr_B, theta_temporal_time, theta_temporal_rr,
    theta_temporal_diag, theta_diag_B, theta_rr_W, theta_diag_W, log_sigma_re_int

`theta_rr_B` precedes the temporal blocks and `theta_diag_B` follows them (TMB orders by
template declaration). Every receipt test asserts R's recorded name vector.

## Admission and refusals (R's check order)

1. The slice 1 pre-pass (one temporal term; source providers refused).
2. unit_obs nesting, whenever `unit_obs` is supplied (R/gllvmTMB.R:1178-1186): "Each
   `unit_obs` level must be nested inside one `unit` level."
3. series/unit partition, only when a term groups on `unit` (R/gllvmTMB.R:1253-1264):
   "The temporal `series` column must have the same partition as `unit` when a stable
   unit covariance component is included."
4. Grouping: a term's group must be the `unit` or `unit_obs` column (R: "Unsupported
   grouping ..." from `gllvmTMB_multi_fit`).

Julia default: `unit = nothing` means the temporal series column (every R temporal test
passes `unit = "series"`).

Julia-side limits, stated in the docstring and reference page: one ordinary term per
level, at most one `(1 | g)`, `common = true` refused. R admits more combinations; they
are not receipted here.

## sigma_eps suppression rule (R/fit-multi.R:6959-6967)

`per_row_diag_W` = a unit_obs term with a diagonal (`indep`, or `latent` with `unique`)
whose `(trait, unit_obs)` cells are unique per row; `per_row_diag_B` likewise for unit.
When either holds and the temporal workflow is not unreplicated, `sigma_eps` is fixed at
`max(1e-3 sd(y), 1e-6)` and `log_sigma_eps` leaves the vector (so `df` drops by one). The
unreplicated temporal workflow keeps it free even with a per-row diagonal term. Receipts:
`sim_rw__Tindep_Wrow` and `sim_rw__Tindep_Wrowlatent` (R's `opt$par` has no
`log_sigma_eps`; the full `log_sigma_eps` equals `log(max(1e-3 sd(y), 1e-6))`).

## Optimiser change (applies to slice 1 fits too)

LBFGS now runs from the same start with both the Hager-Zhang and the cubic-backtracking
line search and keeps the lower objective, then polishes with Newton steps on the exact
ForwardDiff Hessian until the gradient passes `g_tol`. Reason, measured: on
`sim_u__Tdep_Bindep` (temporal_dep + unit indep), Hager-Zhang drifted to the
`sigma_eps -> 0` limit (objective 687.0329, Hessian min eigenvalue ~0) while backtracking
reached gllvmTMB's optimum (687.02046); and on five interior cells LBFGS stopped on a
small function change with gradients 2e-8 to 8e-7, above `g_tol = 1e-8`, which the Newton
polish brings to 1e-13.

## R gradient noise at a fixed sigma_eps

On `sim_rw__Tindep_Wrowlatent`, R's TMB gradient at its optimum differs by 1.9e-8 from a
256-bit central difference of the exact NLL, while Julia's ForwardDiff gradient agrees
with that reference to 8e-14. With `sigma_eps` fixed near `1e-3 sd(y)` TMB's inner
Laplace solve has residual precision near 1e6. The receipt test checks the suppressed
cells against the 256-bit reference at 1e-10 and against R at 5e-8; every other cell
stays at 1e-8.

The 256-bit reference shows only that Julia's gradient is consistent with Julia's NLL.
Two pieces of evidence tie the discrepancy to R's side. First, Julia's NLL and gradient
equal R's `fn` / `gr` to 1e-8 at the deterministic off-optimum coordinates on these same
two cells (max 3.0e-9 / 3.4e-9 over all 25 cells), so the objective is R's. Second, an
R-only check (independent review of PR #563): TMB's `gr` against a central difference of
TMB's own `fn` at R's `opt$par`, h = 1e-4, P1 temporary library:

    sim_rw__Tindep_Wrowlatent  sigma_eps fixed  max|gr - fd(fn)| = 1.13e-05
    sim_rw__Tindep_Wrow        sigma_eps fixed  max|gr - fd(fn)| = 9.6e-06
    sim_u__Tindep_Bindep       sigma_eps free   max|gr - fd(fn)| = 4.4e-07

R's own `fn` and `gr` are more than an order of magnitude less self-consistent on the
suppressed cells than on a free-sigma cell. With h = 1e-4 this carries truncation error
as well, so it is directional support for the inner-Laplace explanation, not a bound on
R's gradient error. Reproduce from the repository root:

    RLIB=<scratch>/Rlib Rscript test/fixtures/temporal_p1/generate_temporal_p1_slice2.R fdcheck

(stage `fdcheck` prints the three lines above and writes nothing). The 5e-8 tolerance is
unchanged.

## Ordination

`extract_ordination(fit; level = :unit)` follows R/extractors.R:505-566: with a unit term
carrying loadings, the conditional unit scores `E[z_B | y]` (raw, no rotation) and the
raw loadings, one row per unit level; otherwise the rank-one temporal state scores; else
`nothing`. Measured against R's `extract_ordination(level = "unit")` at R's coordinates:
9e-16.

## Provenance

Ported semantics (no code copied) from gllvmTMB P1: R/temporal.R:154-158,
R/gllvmTMB.R:1138-1186 and 1253-1264, R/fit-multi.R:5765-5790 (start scales),
6940-6967 (suppression), R/methods-gllvmTMB.R:2706-2716 (tier names), R/extractors.R:
505-566, and the R test files named above. Receipts:
`test/fixtures/temporal_p1/generate_temporal_p1_slice2.R` against a temporary install of
P1 from a detached worktree.
