# Checkpoint — OVERWRITTEN every arc (a pointer to truth, not a log)

- DONE: G0 — ultra-plan approved (ultra-plan.md). Decisions: omitting d means estimate it; gllvmTMB gets the same rule via a spec handed to the Cursor lane; A1 is a feasibility question first.
- D-292 (2026-09-26): Claude owns gllvmTMB too; build auto-d in R and Julia side by side, each cross-checking the other. A8b becomes a real R build, not a spec.
- PARKED FOR A POSSIBLE NEW LANE (Shinichi 2026-09-26): the big LASSO/shrinkage investigation (LASSO vs ridge priors, FA and latent-variable models generally). If this lane cannot address it, open a follow-up lane.
- DONE (00:xxZ 09-27): warm-start safeguard in select_lv (caddc8653): guard rejects failed/unconverged/non-monotone K with recorded status; warm-start retry where fitter takes β_init/Λ_init. 39/39 model_selection tests; sentinel-defect tests pass. Real check: binomial n300 p20 K_true3 → K=4 unconverged excluded, BIC and AIC both pick 3 (AIC previously would pick the unconverged K=4).
- GAP: NB default route (fit_nb_gllvm_grouped, grouped_dispersion.jl = overnight lane file) rejects Λ_init (MethodError) → guard excludes bad K but cannot retry. Needs init kwargs in the grouped kernels; raise with overnight lane / after 11:00Z.
- IN PROGRESS: DRAC nibi array 22744942 (960 tasks, existing API from origin/main code, no safeguard) — measures the raw failure rate. Smoke 22744711 passed (14 rows, 0 fails).
- NEXT: harvest nibi CSVs; NotebookLM synthesis (agent running); then runaway detector (K=5 NB jump to -16720) as the second safeguard; then R twin (D-292).
- OPEN GATE: none
- WHERE TRUTH LIVES: this worktree's branch; artefacts under LOOP/lanes/auto-d-20260926/
- RESUME: read GOAL.md → checkpoint.md → ultra-plan.md, then continue from NEXT.
