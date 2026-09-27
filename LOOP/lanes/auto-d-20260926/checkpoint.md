# Checkpoint — OVERWRITTEN every arc (a pointer to truth, not a log)

- DONE: G0 — ultra-plan approved (ultra-plan.md). Decisions: omitting d means estimate it; gllvmTMB gets the same rule via a spec handed to the Cursor lane; A1 is a feasibility question first.
- D-292 (2026-09-26): Claude owns gllvmTMB too; build auto-d in R and Julia side by side, each cross-checking the other. A8b becomes a real R build, not a spec.
- PARKED FOR A POSSIBLE NEW LANE (Shinichi 2026-09-26): the big LASSO/shrinkage investigation (LASSO vs ridge priors, FA and latent-variable models generally). If this lane cannot address it, open a follow-up lane.
- DONE (to 01:00Z 09-27): Julia select_lv guard + warm start + runaway detector + :bic_sites + fit_gllvm omitted-K (AWAITING G1) + D-43 panel (Opus stats, Sonnet code, Sonnet tests: all PASS WITH REQUIRED FIXES, all fixed: bf8940ad2, 7dd50f157). Ledger leaf-julia ALL MET (J1 tests, J2 explicit-K bit-identical, J3 NB measured). R twin gllvmTMB 978f4bba2 + review fixes 21319c652; leaf-r ALL MET (R1 130 pass, R2 cross-check). Design memo 74 with T2–T7 recommendations. Chen–Li JIC read (not v1). Chi-bar docstring fixed.
- FINDINGS: (1) NB default route: poor optima at K=2–4, K=1/2/4 runaways; guard picks K=5 (truth 3) — flagged task_1cb53b14. (2) Julia Gaussian default route fits no species intercepts — flagged task_b59ac9c8. (3) gllvmTMB latent() adds per-trait Psi by default (unique = TRUE): not the same model as Julia. (4) Binary data: 70–87% of K≥2 fits unconverged, all unconverged are runaways → T7 (ridge). (5) Gaussian: BIC log(n sites) beats log(p·n).
- IN PROGRESS: nibi 22744942 (1–480), narval 4064148 (481–960) — 251/960 at 00:51Z. R ridge experiment ridge/ridge_binary.R (10 reps, ~50 min).
- NEXT: harvest; fill design/74 tables; after 11:00Z overnight lane closes → NB/Beta init kwargs possible; morning report for Shinichi (G1 decisions T2 criterion, T4 API Julia/R, T6, T7).
- OPEN GATE: none
- WHERE TRUTH LIVES: this worktree's branch; artefacts under LOOP/lanes/auto-d-20260926/
- RESUME: read GOAL.md → checkpoint.md → ultra-plan.md, then continue from NEXT.
