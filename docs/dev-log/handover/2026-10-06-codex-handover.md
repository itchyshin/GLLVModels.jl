# Codex handover: true-parity-latest lane, 2026-10-06

You are Codex, picking up the true-parity-latest lane from a Claude Code session. You did not see that session; this file and the files it names are all you need. Read `AGENTS.md` first (native for you), then this file.

## Goal

GLLVModels.jl at true parity with gllvmTMB main pinned at P1 = `9539352f6` (0.7.1). Boundary D-295: temporal and phylo latent inside P1; column grammar and spatial outside (closed by signed disposition, revisited at P2). Every P1 capability inside the boundary gets a scoreboard row with a receipt made at P1, or a 0.7.0 receipt whose sources are byte-identical at P1. Then re-pin to the newest gllvmTMB main and repeat (issue #833).

DONE WHEN: C0 to C8 and the F-gates reverify MET from tracked files on origin/main, the negative controls fail as expected, and a handover is on main.

STOP FOR (ask the maintainer first): every merge; releases, tags, `Project.toml`; public parity claims; edits to the shared Laplace mode code beyond the approved weights change; DRAC jobs (time estimate first); runs over 3 h; another lane's files; case-map classifications. Agents never sign; signatures and classifications are the maintainer's.

## Current state on origin/main @ `9269f4622`

Measured with `node tools/true_parity_check.mjs <clause>`:

| Clause | Result |
|---|---|
| C0, C1, C5, C7, C8 | MET |
| X2 | 297 of 317 done |
| C2 | 283 done, NOT MET |
| C3 | 6 of 8 (ORDINAL-LOGIT-RSZ, ISDM-HEADLINE-RSZ open) |
| C4 | 4 of 8 (urbanisation, spider, beetle, fungi miss tolerances) |
| C6 | 37 Julia-only exports held for the maintainer |
| F-gates | informational in GATES.md; they fold into C2 |

Negative controls (`node tools/test_true_parity_check.mjs`) all pass. CI now runs 12 receipt-tool `--check`s plus the behaviour-receipts tests in `.github/workflows/true-parity-check.yml`.

## What landed this session (2026-10-04 to 2026-10-06)

- **Bug fixes:** #806 (profile/bootstrap refuse pinned fits; masked Gaussian PosDef, closes #716), #807 (training offset on 11 shared-dispersion fit types; #788 stays open), #817 (phylo docstrings; postfit and isdm receipt checks repaired; receipt checks in CI).
- **Parity rows:** #808 DATA-OFF-MIXED; #809 namespace-2 batch at P1; #810 six covariance twins (COV-ORD-LATENT x3, COV-PHYLO x3, stationarity asserted instead of the platform-dependent `converged` flag); #814 four family rows re-measured on Totoro (second oracle build `GLLVM_PARITY_ORACLE_BUILD=totoro`); #819/#821 C6 inputs refresh, 23 signed C6 names, `predict(...; modes = :training)` and DATA-OFF-PREDICT, FAMILY-00 expectation.
- **Maintainer rulings (vault D-319, 2026-10-05, plus follow-ups 2026-10-06):** wired in #838 (X2 206 to 284). Rule text is in `docs/dev-log/core070/true-parity-latest/GATES.md`, section "Rulings of 2026-10-05". Follow-ups: N9 applies only to receipts carrying a `convergence_parity` block; N6's internal predicates are ISDM-NO-TRAITS, -WRONG-ID, -WRONG-LINK; the weights edit to the shared Laplace mode was approved under review; the six helper-internal weights rows bind on fit-level twins.
- **Weights:** #839, observation weights on the Poisson Laplace route through the shared mode code; unweighted fits proven bit-identical to main (253/253); all 13 DATA-W rows bound (X2 284 to 297).
- **Handover:** this file; the earlier lane handover is `docs/dev-log/handover/2026-10-05-claude-handover-true-parity-latest.md`.

## Next immediate steps (yours)

Each is agent-only unless marked. Run each as one PR: branch, independent review, then the maintainer's merge decision. After every change: regenerate with the owning receipt tool, then `python3 tools/true_parity_assemble.py`, and pass every `--check`.

1. **FAMILY-11-LOG.** Its numeric cases pass; it is held because its bridge context case is NOT_EXECUTED, and N1 accepts only an R public-bridge refusal recorded as `R_BOUNDARY_UNCHANGED`. Record a P1 R probe of the truncated-NB2 bridge refusal (GJL-GATE-FAMILY) through `gllvmTMB(..., engine = "julia")` and write it as an `R_BOUNDARY_UNCHANGED` receipt; then the family tool's `n1_context` binds the row. Totoro already has the bridge environment at `~/hsq_work/gllvm-w2-3` (Julia 1.10.12, gllvmTMB built from the tracked P1 archive, TMB 1.9.21). Linux needs libunwind preloaded for JuliaCall and `LD_LIBRARY_PATH` cleared for Julia children started from R (see PR #814).
2. **ORDINAL-LOGIT-RSZ and ISDM-HEADLINE-RSZ (C3).** Under N9 they bind once their campaign receipts carry the `convergence_parity` block (both engines' max-abs gradient at or below 1e-5) at a Newton-polished point. The polish harness is in draft PR #715 (`polish_J.jl`, `run_R.R` changes), not merged. Land it, teach `tools/true_parity/campaign/write_receipts.py` to record `compared_point: "newton_polished"` and Julia's gradient, then re-run. Estimate: ORDINAL about 10 to 15 min single-threaded; ISDM similar. Totoro is fine (threads capped).
3. **check_auto_residual (N10).** In the behavioural scope but its receipt has no behaviour block. Add one through `tools/core070_behaviour_receipts.py` and the postfit tool.
4. **Five iSDM public-door rows (#836).** ISDM-COUNT, -EXTRA-SOURCE, -MISSING-IN-TRAIT, -MISSING-SOURCE, -WRAPPER-LAW need Julia public-door runs (`fit_isdm_gllvm`) and behaviour blocks. ISDM-EXTRA-SOURCE's compared class is R's earlier `length(family)` guard.
5. **Remaining C4 rows** (urbanisation, spider, beetle, fungi) miss tolerances; diagnose per row before any re-run. Spider and beetle were judged out of reach as specified in the 2026-10-03 wave plan.
6. **Open for the maintainer, not agents:** the 37 held C6 names (#831; proposal file `c6-proposed-decisions-2026-10-05.md` in the lane kit); the icc profile withdrawal (#827); POST-SIMULATE-DEFAULT seed sensitivity (#828). Also namespace simulate multi, bootstrap_Sigma, extract_phylo_signal (#834); animal_latent stays unbound by ruling (#835).

Future-work issues filed this session: #820, #822 to #837.

## Key decisions and rationale

- A comparison that cannot discriminate (for example intercepts on centred data, or a 0 = 0 length) does not bind (`mark_degenerate`, `DEGENERATE_ABS = 1e-10` in `tools/core070_postfit_p1_receipts.py`). That is why `namespace/export/gllvmTMB` needed a loadings comparison and POST-COEF-EMPTY stays non-discriminating.
- Bridge readback (R returning what Julia computed) binds only for `fitted`, `predict`, `residuals`; `coef`, `logLik`, `summary`, `confint`, `simulate` close by signed disposition (ruling 1).
- Monte-Carlo rows bind only on a rule committed before the run (N4); POST-SIMULATE-DEFAULT is held because its twin fails 7 of 40 alternative seeds.
- Tests assert stationarity (positive-definite Hessian, Newton decrement at most 1e-9, gradient at most 1e-4) rather than an optimiser `converged` flag, which differs between macOS and Linux at the same optimum (#822).

## Live toolchain for Codex

```bash
export PATH="$HOME/.juliaup/bin:$PATH"          # julia / julialauncher
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # the Mac is shared; keep light
export GLLVM_P1_RLIB="$HOME/local-scratch/gllvmTMB-p1-scratch/Rlib"     # frozen gllvmTMB 0.7.1 at P1
export GLLVMTMB_DIR="$HOME/Dropbox/Github Local/gllvmTMB"               # read-only R clone; read P1 with git show 9539352f6:<path>
export NOT_CRAN=true
```

- Install: `julia --project=. -e 'using Pkg; Pkg.instantiate()'` (worktrees have no Manifest; copy the git-ignored `Manifest.toml` from the main checkout if needed).
- Full suite: `julia --project=. -e 'using Pkg; Pkg.test()'`. P1-tagged twin tests (`# gllvm-parity-tag: P1`) run as `julia --project=. test/<file>` in the root env and may use only root dependencies (no StableRNGs, no JSON3; TOML fixtures).
- Heavy runs: Totoro via the existing socket only, `ssh -o ControlPath=~/.ssh/cm-snakagaw@totoro.biology.ualberta.ca:22 -o BatchMode=yes snakagaw@totoro.biology.ualberta.ca`; at most 150 cores, cap BLAS/OMP threads explicitly (one run here hit 192 by default), kill what you start. Estimate before any run; over 3 h needs approval.
- Merges in this lane went through a head-pinned merge train (`scripts/merge_train_v2.sh <PR>:<sha>` in the lane kit) after independent review; use `MERGE_METHOD=--merge` for integration PRs.

## Gotchas

- Generated files (`case-map-*.json`, `case-map-assembled.json`, `scoreboard.md`, `reverse-gap*.json`, `behaviour-equivalence.json`) are never hand-merged: on conflict take either side and regenerate with the owning tool.
- Run `python3 tools/wrong_close_guard.py --body-file <body> --git-range origin/main..HEAD` before every push; put "refs #N" on its own line. Agents are never named as GitHub @handles.
- `tools/core070_grouping_p1_receipts.py --check` fails on main for an unrelated reason (#824).
- GitHub occasionally cancels CI jobs en masse; re-run cancelled jobs once before treating it as a failure.

## Where truth lives

- Ledger and rules: `docs/dev-log/core070/true-parity-latest/GATES.md`, `tools/true_parity_check.mjs`, `tools/true_parity_assemble.py`, the assembled `scoreboard.md`.
- Lane kit (local, not in git): `~/local-scratch/lanes/GLLVM.jl-true-parity-latest/LOOP/lanes/true-parity-latest/` (`GOAL.md`, `checkpoint.md`, `signed-rulings-2026-10-05.md`, `wave-plan-2026-10-03.md`, `weights-design-2026-10-05.md`, `c6-proposed-decisions-2026-10-05.md`).
- Maintainer decisions: vault `memory/DECISIONS.md`, D-295, D-300, D-319 and its follow-ups.

## How to resume

Start Codex in the repository root and paste:

> Rehydrate from docs/dev-log/handover/2026-10-06-codex-handover.md + the AGENTS.md snapshot, then continue with the Next Immediate Steps.
