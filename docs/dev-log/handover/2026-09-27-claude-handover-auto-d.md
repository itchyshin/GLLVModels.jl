# Handover: auto-d lane (estimate the number of latent dimensions), 2026-09-27

You are Claude, picking up the auto-d lane in GLLVModels.jl and its gllvmTMB twin. The authoring session
ran in the glmmTMB folder; this handover moves the lane's home to GLLVModels.jl. Nothing here is merged.

## Mission-control

| repo | branch (draft PR) | pushed head | state |
|---|---|---|---|
| GLLVModels.jl | `claude/lane-auto-d-20260926` ([#518](https://github.com/itchyshin/GLLVModels.jl/pull/518)) | `db6b31003` | guarded `select_lv`, `:bic_sites` default, `fit_gllvm` without K, loading ridge for binary data; ridge under review |
| gllvmTMB | `claude/lane-auto-d-r-20260926` ([#1324](https://github.com/itchyshin/gllvmTMB/pull/1324)) | `3b1e8e61f` | same guard, `bic_sites` default, `latent(d = "auto")`, `binary_ridge`, docs cascade; review-fix round was in progress (uncommitted edits in the worktree) |

Worktrees: `~/local-scratch/lanes/GLLVM.jl-auto-d-20260926` (Julia) and
`~/local-scratch/lanes/gllvmTMB-auto-d-20260926` (R). Lane kit (goal, checkpoint, plan, evidence):
`LOOP/lanes/auto-d-20260926/` on the Julia branch. Decision memo: `docs/design/74-auto-latent-dimension.md`.

## Critical context

- **Decisions (vault):** D-292 (gllvmTMB moved into this lane; R and Julia built side by side),
  D-293 (Shinichi's sign-off: `:bic_sites` default in both; omitting K estimates it in Julia; R gets
  `d = "auto"` with the `latent()` default unchanged at d = 1; binary data via a loading ridge; push and
  open draft PRs, never merge).
- **Merges need Shinichi's word.** Do not merge either PR.
- **Evidence:** DRAC recovery grid, 17 569 complete datasets (`LOOP/lanes/auto-d-20260926/pilot/harvest-report-complete.md`).
  Mean exact recovery, guarded rule with `:bic_sites`: Gaussian 0.948, Poisson 0.999, NB 0.895;
  binomial ≤ 0.56 without the ridge. R ridge experiment: true d = 2 found in 8/10 datasets (20 traits,
  n = 120) with the ridge vs 4/10 without. Literature: NotebookLM `c2c564e3`, vault note
  `dr-auto-d-latent-dimension-selection`.

## Current working state

**Working (committed, pushed):** everything in the mission-control table except the items below.

**In progress when this handover was written (authoring session may still be running them):**
1. **R review-fix round** for `latent(d = "auto")`: uncommitted edits in the R worktree (`R/gllvmTMB.R`,
   `R/brms-sugar.R`, `R/select-lv.R`, `man/*`, `NEWS.md`). The nine confirmed findings and evidence are in
   the review journal `~/.claude/projects/-Users-z3437171-Dropbox-Github-Local-glmmTMB/4f055e97-b708-44a2-bba7-707dda5c4eff/subagents/workflows/wf_190cad38-518/journal.jsonl`
   (summary: `species =` alias dropped on the auto path; scan accepts formulas `select_lv()` refuses and
   the hint misleads; wrong `?latent` claim; the `document()` run reverted the #1298 `?latent` wording;
   a roxygen heading swallows ~90 lines of `?gllvmTMB`; docs silent on which families have evidence;
   internal decision ID in reader-facing help; the real-fit test uses true d = 1; the message test
   ignores d, criterion and caveat).
2. **Julia ridge review** (3 lenses, 2 skeptics per finding) on commit `0b3f5d117`: results in
   `~/.claude/projects/-Users-z3437171-Dropbox-Github-Local-glmmTMB/4f055e97-b708-44a2-bba7-707dda5c4eff/subagents/workflows/wf_b2d4ba93-5d7/journal.jsonl`.
3. **Julia binary recovery check** (`LOOP/lanes/auto-d-20260926/ridge/ridge_binary_julia.jl 10 1.5`,
   output `ridge/ridge_binary_julia_L1.5.csv`): compares the Julia sweep with and without the ridge on
   the R experiment's cells.
4. **NB re-run on DRAC** (1 631 datasets missing or half-finished from the grid; ~1 600 core-h expected,
   range 800–3 000): narval 4098607 (tasks 1–544), nibi 22779067 (tasks 1–544), rorqual 21902752
   (543 moved tasks; ids in `pilot/rorqual_moved_ids.txt`). Outputs in each cluster's
   `~/projects/def-snakagaw/snakagaw/auto-d-pilot/out-rerun/`. Pre-lane fitting code on all three.

## Next immediate steps (classify each OWED / DONE on arrival)

1. Run `tools/lane_preflight.sh` in GLLVModels.jl and gllvmTMB; check `git status` in both worktrees.
2. **R fix round:** if the R worktree has uncommitted edits and no commit after `3b1e8e61f` addresses the
   review findings, finish it: re-read the journal above, complete the fixes, run
   `bash LOOP/lanes/auto-d-20260926/gates/r_tests.sh` (from the Julia worktree; expect `R-TESTS-PASS`),
   commit, push to #1324.
3. **Julia ridge review:** read the journal; fix every confirmed finding on the Julia branch; run
   `node ~/shinichi-brain/skills/unlazy/scripts/gate-check.mjs --root . --cwd . --approve --reverify --timeout 3600 .unlazy/auto-d/gates/leaf-julia.md .unlazy/auto-d/gates/leaf-docs.md`
   (expect ALL MET); push to #518.
4. **Recovery check:** if `ridge/ridge_binary_julia_L1.5.csv` is complete (80 rows), compare with the
   R result and add the numbers to design/74 T7 and the PR body; if missing, re-run it (≈ 60–90 min local).
5. **Docs pass: DONE** (Julia `fe09db2e3`: README note, after-task report; `api.md` already lists `select_lv`, `LVSelection`, `fit_binomial_gllvm` with `loading_ridge` documented. R `3b1e8e61f`: check-log, validation-debt rows MS-03 to MS-06, formula-grammar note, `vignettes/articles/model-selection-latent-rank.Rmd` example with `eval = FALSE`, after-task report). Not run: Documenter build, `devtools::check()`, 3-OS CI.
6. **NB re-run:** when all three arrays finish, rsync each cluster's `out-rerun/` to separate local dirs
   (task ids overlap across clusters), then
   `python3 LOOP/lanes/auto-d-20260926/pilot/analyze.py harvest,harvest-rerun-narval,harvest-rerun-nibi,harvest-rerun-rorqual harvest-report-final2.md`
   (later dirs fill incomplete datasets). Report core-hours (`sacct … elapsedraw,alloccpus`) against the
   estimate. Update design/74, the MORNING-REPORT numbers and both PR bodies (#518 says 17 687; the
   complete-only count is 17 569 before the re-run).
7. Release leases (`claude:GLLVM.jl:auto-d*`, `claude:gllvmTMB:auto-d*`) when done; update
   `LOOP/lanes/auto-d-20260926/checkpoint.md`.

## Blockers / open questions (Shinichi's)

- Merge of #518 and #1324.
- Whether `select_lv` parity is signed by this lane or the new true-parity programme (that session will ask).

## Gotchas / failed approaches

- **Never `git stash` in a worktree shared with another agent**: it sweeps up their unsaved edits.
- rorqual compute nodes have no internet: install packages on the login node with
  `JULIA_PKG_PRECOMPILE_AUTO=0`, precompile inside a job.
- DRAC per-user submit limit is 1 000 jobs; the grid used cost-balanced task maps.
- gllvmTMB `latent()` adds a per-trait Ψ by default (`unique = TRUE`); compare with GLLVModels only with
  `unique = FALSE`.
- Julia's default Gaussian fit has no species intercepts (fixed on draft #519, another lane); re-run the
  Gaussian grid cells on uncentred data after #519 lands.
- Under a ridge, the "logLik cannot fall with K" check must use the penalised value (done in both packages).
- `agent_mention_check.py` takes a file path, not text.

## Files created / modified (this lane)

Julia (`git diff --name-only origin/main...claude/lane-auto-d-20260926`): `src/model_selection.jl`,
`src/families/fit_gllvm.jl`, `src/boundary_inference.jl`, `src/families/binomial.jl`,
`src/families/aghq_binomial_fit.jl`, `test/test_model_selection.jl`, `test/test_binomial_ridge.jl`,
`test/runtests.jl`, `docs/src/tutorial.md`, `docs/design/74-auto-latent-dimension.md`, `CHANGELOG.md`,
`README.md`, `docs/dev-log/after-task/2026-09-27-auto-d.md`, `docs/dev-log/plan-actual/2026-09-27-auto-d.md`,
`LOOP/lanes/auto-d-20260926/**`, and this file.
R (`git diff --name-only origin/main...claude/lane-auto-d-r-20260926`): `R/select-lv.R`, `R/gllvmTMB.R`,
`R/brms-sugar.R`, `man/select_lv.Rd`, `man/latent.Rd`, `man/gllvmTMB.Rd`, `NEWS.md`,
`tests/testthat/test-select-lv-guard.R`, `tests/testthat/test-select-lv-ridge.R`,
`tests/testthat/test-latent-auto.R`, plus the docs cascade (check-log, after-task, validation-debt
register, formula-grammar note, example).
Vault: `memory/DECISIONS.md` (D-292, D-293), `memory/AGENT_LOG.md`, `memory/PROJECT-NOTEBOOKS.md`,
`projects/deep-research/README.md`, `projects/deep-research/dr-auto-d-latent-dimension-selection.md`.
Snapshot pointer: not refreshed (many lanes are active in GLLVModels.jl; see
`docs/dev-log/coordination-board.md`).

## Environment

Julia 1.10 (`julia --project=.`), `OPENBLAS_NUM_THREADS=1`, `JULIA_NUM_THREADS=4` on the Mac. R via
`devtools::load_all()`. DRAC via existing `~/.ssh/cm-snakagaw@<cluster>.alliancecan.ca:22` sockets
(never trigger Duo). Do not stage `.unlazy/` or `LOOP/lanes/auto-d-20260926/pilot/harvest/` (git-ignored).

## How to resume

```text
Read AGENTS.md and docs/dev-log/handover/2026-09-27-claude-handover-auto-d.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```
