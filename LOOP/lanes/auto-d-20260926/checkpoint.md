# Checkpoint — OVERWRITTEN every arc (a pointer to truth, not a log)

- DONE: G0 — ultra-plan approved (ultra-plan.md). Decisions: omitting d means estimate it; gllvmTMB gets the same rule via a spec handed to the Cursor lane; A1 is a feasibility question first.
- D-292 (2026-09-26): Claude owns gllvmTMB too; build auto-d in R and Julia side by side, each cross-checking the other. A8b becomes a real R build, not a spec.
- PARKED FOR A POSSIBLE NEW LANE (Shinichi 2026-09-26): the big LASSO/shrinkage investigation (LASSO vs ridge priors, FA and latent-variable models generally). If this lane cannot address it, open a follow-up lane.
- DONE 09-26/27 (Shinichi: "work autonomously till 5 am"): guard + warm start (caddc8653); runaway detector (9653b1778); chi-bar docstring fix (14d207517); design memo 74 draft (67447f15a); fit_gllvm omitted-K path, AWAITING G1 (5ed53d4b7); harvest script + interim Gaussian table (2f47ecb74). model_selection 55/55; fit_gllvm 11, unified API 24, formula 30 pass.
- FINDINGS: NB n300 p20 K_true3: K=2 fit (the one every criterion picked) is a runaway (max latent SD 26.9, median 2.7). Interim Gaussian grid (3000 datasets): BIC log(p·n) under-selects badly at small n; BIC log(n) best at n ≥ 60 with p = 20; AIC overshoots ~10–15%.
- GAP: NB default route rejects Λ_init (grouped_dispersion.jl, overnight lane's file until 11:00Z).
- IN PROGRESS: nibi 22744942 (tasks 51–480 after hand-off), narval 4064148 (481–960, waits on smoke 4064147; watcher cancels nibi 481–960 once narval smoke passes). R port in gllvmTMB worktree ~/local-scratch/lanes/gllvmTMB-auto-d-20260926 (Sonnet builder). NB K=5 runaway probe.
- NEXT: harvest as tasks land (python3 LOOP/lanes/auto-d-20260926/pilot/analyze.py harvest harvest-report.md); fill design/74 recovery table; D-43 panel on the Julia branch; Chen–Li JIC read.
- OPEN GATE: none
- WHERE TRUTH LIVES: this worktree's branch; artefacts under LOOP/lanes/auto-d-20260926/
- RESUME: read GOAL.md → checkpoint.md → ultra-plan.md, then continue from NEXT.
