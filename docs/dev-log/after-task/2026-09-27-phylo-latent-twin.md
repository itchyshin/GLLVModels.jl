# After-task: phylo_latent twin at gllvmTMB P1, gate rows A14 and A15 (2026-09-27)

Lane: Claude, branch `claude/phylo-latent-build` from `origin/main` `97e11be04`, worktree
`~/local-scratch/worktrees/GLLVM.jl-phylo-latent-build`, lane lease on the new files plus
`src/precision_multivariate_fit.jl`. Draft PR #547. Built from the reviewed spec
`docs/design/phylo-latent-port-spec.md` (draft #545, head `c0fd5f5be`), its review, and the
signed scope D-300 (packet 1d, answers 1 to 12 as recommended).

## 1. Goal

Build the Julia twin of R's bare Gaussian `phylo_latent(species, d = K)` at P1
(`9539352f6`, 0.7.1): named entry `fit_phylo_latent_gllvm` on the R-shaped `PrecisionPhy` /
`fit_precision_multivariate` path, R's argument names and refusals, label matching, R's dense
ridge, polytomies via the raw-triplet constructor, the `rho` scope fence, `extract_phylo_signal`
returning `H2 = 1` on a bare fit; red-first twins per the spec's map; paired P1 receipts for A14
(both optima, both cross objectives) and A15 (signed shape, wall time).

## 2. Implemented

- `src/phylo_latent.jl`: entry, tree precision by R's rule (Newick or `AugmentedPhy`, general
  trees), dense route with the `1e-8` ridge and a condition-number warning, coverage refusal
  with R's `droplevels()` and "genuine mismatch" branches, refusal table, the `rho` fence, and
  in-keyword `Ainv` as R's `vcv = solve(as.matrix(Ainv))`.
- `src/phylo_latent_postfit.jl`: `extract_phylo_signal(::PrecisionMultivariateFit)`.
- `src/precision_multivariate_fit.jl`: two appended fields and a 23-argument outer constructor
  that fills them from the precision, plus `_pmv_with_labels`.
- `src/destination_b_postfit.jl`: `level = :phy` accepted.
- Tests: `test/test_phylo_latent_twin.jl` (24 of 26 mapped R blocks plus Julia-only checks),
  `test/test_phylo_latent_paired_p1.jl` (parity tag P1).
- Tools and receipts: `tools/phylo_latent/r_reference_p1.R`,
  `tools/phylo_latent/compare_phylo_latent_p1.jl`, `docs/dev-log/core070/phylo-latent-p1/`.
- Docs: api, low-level reference, tutorial subsection, README, capability status, ROADMAP,
  CHANGELOG, check-log, decisions note.

## 3a. Decisions and Rejected Alternatives

- `Ainv`: P1 rewrites in-keyword `Ainv` to the dense ridged route (`R/brms-sugar.R:3311-3319`),
  contradicting the spec's sparse-route mapping. First refused (brief said stop); after review,
  ported literally per the maintainer rule. R's global sparse route has no keyword twin.
- No unary-node refusal: P1's validator has none; the spec's claim is wrong.
- `extract_Sigma` extended with `:phy` rather than a second method (one already existed).
- `species_levels` added so R's declared-but-unobserved levels have an analogue.
- A15 stationarity gap asserted with explicit measured bounds (first as `@test_broken`), not fixed: a Newton polish or scale-aware
  stopping rule would change `fit_precision_multivariate` for every caller (bridge included);
  out of scope for this build.
- Rejected: passing a looser `g_tol` for the A15 receipt so `converged` reads true.

## 4. Files Touched

New: `src/phylo_latent.jl`, `src/phylo_latent_postfit.jl`, `test/test_phylo_latent_twin.jl`,
`test/test_phylo_latent_paired_p1.jl`, `tools/phylo_latent/*`, `docs/dev-log/core070/phylo-latent-p1/*`,
`docs/dev-log/decisions/2026-09-27-phylo-latent-parameterisation.md`, this report.
Edited: `src/GLLVModels.jl`, `src/precision_multivariate_fit.jl`, `src/destination_b_postfit.jl`,
`test/runtests.jl`, `CHANGELOG.md`, `README.md`, `ROADMAP.md`, `docs/src/api.md`,
`docs/src/low-level-reference.md`, `docs/src/tutorial.md`, `docs/design/capability-status.md`,
`docs/dev-log/check-log.md`. Not touched (hard stops): `_laplace_mode`, `src/families/mixed.jl`,
`grouped_dispersion.jl`, `model_selection.jl`, `cv.jl`, `src/formula.jl`, confint entry points,
`Project.toml`.

## 5. Checks Run

Per file, one process at a time, `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, JSON3 and
StableRNGs from a stacked scratch environment:
- `test_phylo_latent_twin.jl`: red first (UndefVarError; 2 fail, 9 error), then 81/81 on 1.10.12
  and 1.13.0.
- `test_phylo_latent_paired_p1.jl` (as the P1 workflow runs it): 113/113, 0 broken, on both
  versions, live A15 refit included. Twin file after review: 89/89 in the test environment.
- Adjacent, 1.10.12: `test_precision_multivariate_fit` 47, `test_destination_b_postfit` 29,
  `test_destination_b_fixed_effects` 25, `test_precision_shared_residual` 19,
  `test_bridge_precision_multivariate` 5, `test_destination_b_tree_precision` 19; all pass.
- Tutorial snippet executed on 1.10.12.
- Full suite and local Documenter build not run (brief: never the full suite; Documenter
  check left to branch CI).

## 6. Tests of the Tests

- The twin file was run against the stashed (unimplemented) tree and failed as expected.
- A first 1.13 run failed the recovery test because `randperm` / `randexp` streams differ
  across Julia versions; the generator now uses only version-stable StableRNGs draws, and
  both versions produce the identical fit (logLik -278.87121982930773).
- The replay pins SHA-256 of every fixture and receipt and would fail on any byte change.
- A15 bounds are set from measured values with headroom (2.0e-4 against 1e-3, 4.0e-3 against
  1e-2, 1.6e-5 against 1e-4); the new Ainv testsets failed before the implementation.

## 7a. Issue Ledger

- New finding, not filed: `fit_precision_multivariate` LBFGS stops above an absolute
  `g_tol = 1e-5` at large objective scale (A15: gradient 2.0e-4, objective 7372); one
  FD-Newton step reaches 6e-6. Same class as the #485 convergence-rule family.
- Maintainer question: the dated A14/A15 promotion block.

## 8. Consistency Audit

- Docstrings on the new export and both internal builders; entries in api.md and
  low-level-reference.md; extractor method docstring picked up by the existing
  `extract_phylo_signal` block.
- The bridge still reports the signal as `:estimand_not_admitted`
  (`_pmv_phylogenetic_signal`) while the public extractor answers `H2 = 1`; recorded in the
  decisions note, bridge out of scope.
- No "issue #N" phrases in docstrings; no em dashes in new prose.

## 9. What Did Not Go Smoothly

- `obj$report(par)` in TMB needs the full parameter vector; the first R receipt attempt failed
  until it used `obj$env$last.par`.
- The first replay structure check used a Frobenius-norm `atol = 1e-10` on two dense inverses
  of the order-198 A15 precision; it failed on roundoff and became norm-relative `1e-10`, with
  an added check against the fixture's `ape::vcv` matrix.

## 10. Known Residuals

- A15 stationarity gap (both engines above the 1e-4 cross-gradient bar; Julia
  `converged = false`).
- Deferred twins: `test-species-unused-levels-guard.R:78` (R's global sparse route, no keyword twin) and
  `test-phylo-signal-categorical.R:137` (would change `extract_phylo_signal(::GllvmFit)`).
- Spec section 4.2 deferrals unchanged (`unique = TRUE` pairing, site latent, intervals).
- Receipts `qualified = false` until the maintainer signs.
- The full suite, Aqua/JET and Documenter have not run locally on this branch.

## 11. Team Learning

- Reading R at P1 before building caught two spec claims that R does not support; the
  brief's stop rule kept them out of the code.
- For cross-version StableRNGs fixtures, avoid `randperm` and `randexp`.

## 12. Cross-Product Coverage

Covered: bare Gaussian `phylo_latent` via tree (Newick, `AugmentedPhy`) and dense `vcv` / `A`,
`d` from 1 to T, label matching with declared levels, R's refusals, `unique = true` as an
unpaired extra. This arc does NOT cover: R's global sparse `phylo_vcv` route, `rho != 1` (refused), non-Gaussian families,
`unique = TRUE` pairing, `lv = ~x`, slopes, formula grammar, the R bridge, intervals of any
kind, phylo signal intervals, `phylo_scalar` (PROPTO, fenced), and multi-seed recovery
evidence (one StableRNGs fixture only).
