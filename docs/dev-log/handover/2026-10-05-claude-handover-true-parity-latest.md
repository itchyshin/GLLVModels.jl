# Handover: true-parity-latest lane, 2026-10-05

Goal: GLLVModels.jl at true parity with gllvmTMB main pinned at P1 = 9539352f6 (0.7.1); boundary D-295 (temporal and phylo latent inside; column grammar and spatial outside). Status: PAUSED at the maintainer's rulings. Every row agents can bind alone is merged; each remaining row waits on a ruling, a classification, or an approved capability build. Not done: C2, C3, C4 and C6 are NOT MET on origin/main.

## Where truth lives

- Kit `LOOP/lanes/true-parity-latest/` in the lane worktree `~/local-scratch/lanes/GLLVM.jl-true-parity-latest`: `GOAL.md`, `checkpoint.md`, `rulings-needed-2026-10-04.md` (every open decision, each with a recommendation; updated 2026-10-05), `wave-plan-2026-10-03.md`, `post-nobs-fallback-evidence.md`, `scripts/merge_train_v2.sh`.
- Ledger and checker on main: `docs/dev-log/core070/true-parity-latest/GATES.md`, `tools/true_parity_check.mjs`, `tools/true_parity_assemble.py`, the assembled `scoreboard.md` and `case-map-assembled.json`, and the receipt tools `tools/core070_*_p1_receipts.py`.

## Landed on 2026-10-05

- #806 (integration of #804 and #805): the direct profile and bootstrap entry points refuse a `lambda_constraint` fit; masked Gaussian fits return `+Inf` at non-positive-definite trial points instead of throwing (#716 closed).
- #807: `getLV`, `predict` and `residuals` on 11 shared-dispersion fit types use the stored training offset (refs #788; #788 stays open for `simulate`, Delta `predictor = :shared` and training-data identity).
- #808: `fit_mixed_gllvm` takes an offset with gllvmTMB's row-wise admission rule; binds DATA-OFF-MIXED.
- #809: the namespace-2 batch runs at P1; `namespace/export/gllvmTMB` is measured but recorded `numeric_non_discriminating` under the shared rule (its intercept comparison is degenerate on centred data).
- #810: six covariance twins bound (COV-ORD-LATENT-BARE, -DEFAULT, -COMMON; COV-PHYLO-DEP, -A-ALIAS, -FOLDED-UNIQUE). The phylo twin asserts stationarity (positive-definite Hessian, Newton decrement at most 1e-9, gradient at most 1e-4), because the optimiser's `converged` flag at the rounding floor differs between macOS and Linux.
- #814: family rows re-measured at P1 on Totoro (a registered second oracle build, `GLLVM_PARITY_ORACLE_BUILD=totoro`); binds FAMILY-02-LOG, -05-LOG, -07-LOGIT and BETA-ALIAS. Bridge route-vs-native entries sit in `route_consistency`, not in the R-vs-Julia `comparison` block.
- #817 (integration of #815 and #816): corrected `phylo_dep` and `phylo_latent` docstrings; repaired the postfit (stale counts) and isdm (campaign row) receipt checks; the true-parity workflow now runs 11 receipt-tool `--check`s.

## Gates on origin/main at 0852df0b5

Measured with `node tools/true_parity_check.mjs <clause>`.

| Clause | Result |
|---|---|
| C0, C1, C5, C7, C8 | MET |
| C2, X2 | NOT MET, 205 of 317 rows done |
| C3 | NOT MET (campaign rows; N9 convergence rule pending) |
| C4 | NOT MET (bridge leg; C4 text or harness decision pending) |
| C6 | NOT MET (60 held names need a decision each) |
| F-gates | informational in GATES.md; they fold into C2 once the scoreboard carries their ids |

Receipt checks on main (aghq, data, family, inference, isdm, namespace_2, postfit): all CURRENT. `true_parity_assemble.py --check`: ASSEMBLE_OK. Negative controls (`node tools/test_true_parity_check.mjs`): all pass.

## Open, for the maintainer

All in `rulings-needed-2026-10-04.md`. The largest single item is N1 (about 13 rows). New on 2026-10-05: DATA-OFF-PREDICT (keep training modes on new offsets), POST-NOBS-FALLBACK (unreachable through the public door; disposition recommended), `namespace/export/gllvmTMB` (add a loadings comparison), `namespace/export/animal_latent` (needs a random-slope `animal_latent`), CI-ROUTE-034, the W1-1 mutation-control repoint, and the FAMILY-00 random-effect-name expectation.

## Follow-ups found on the way (not filed)

- Delta `predictor = :shared`: post-fit `getLV`/`predict` ignore the occurrence loadings.
- `fit_phylo_latent_gllvm` with `d = T` stops at the rounding floor with a gradient above `g_tol`, so a user sees `converged = false` on a correct optimum.
- `src/animal_dep.jl`: its docstring claims gllvmTMB's `animal_dep` estimand, but it uses the row-phylogeny path, as `phylo_dep` did.
- `core070_grouping_p1_receipts.py --check` fails because its run commit lives only on the squash-merged branch of #593 (not part of the assembled ledger).
- P1 generators assert `packageVersion == "0.7.1"`, not the commit.

## Lessons

- A receipt tool whose `--check` is not in CI drifts silently: postfit counts and the isdm campaign row both broke main unnoticed. The workflow now runs them.
- A test that asserts an optimiser's `converged` flag can pass on macOS and fail on Linux at the same optimum. Assert stationarity instead.
- A comparison that cannot discriminate (for example intercepts on centred data) blocks binding under the shared rule; check `mark_degenerate` before calling a row bound.
