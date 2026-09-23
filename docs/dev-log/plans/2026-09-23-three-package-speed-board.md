# End-of-arc speed board — three packages (inventory + cell matrix)

**Status:** DRM + H² filled earlier 2026-09-23. GLLVM unstructured `speed_bench` tip abs Totoro @ `8d58a0c94` (post-#449/#450); 9/10 GLLVM cells `has_receipt`; only `gllvm-profile-ci-small` still `needs_run`.
**Sibling plans:** `2026-09-23-ultra-plan-three-package-speed.md` (Q1 verdict + Phases 0–2), `2026-09-23-ultra-plan-next-after-448.md` (GLLVM Latte arc).
**Thread discipline (standing):** Mac Studio lanes measure at `JULIA_NUM_THREADS=4` + `OPENBLAS_NUM_THREADS=1` unless a receipt says otherwise. Totoro H² selinv cells used `threads=1`. Do not mix regimes in one speedup column.

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.

---

## 1. What exists (last ~2 weeks)

### GLLVModels (`GLLVM.jl`)

| Artifact | Path / PR | Role |
|---|---|---|
| Speed78 lane | `~/local-scratch/lanes/GLLVM.jl-speed78-20260919` · **PR #430 MERGED** | S6–S8: phylo EM sparse loglik, CHOLMOD reuse, warm start, analytic outer gradient |
| S9a Hessian | `~/local-scratch/lanes/GLLVM.jl-s9a-hessian-20260921` · **PR #446 MERGED** | D-274 Hessian-by-grad-FD; `glmm_200x5` / `glmm_5000x3_g500` |
| Moment / NM | **PR #447 MERGED** | NM demotion abandoned; `moment_start` opt-in (not a shipped wall win) |
| S9c coverage | `~/local-scratch/lanes/GLLVM.jl-s9cov-20260921` · **PR #448 MERGED** | coverage certification; not a fit-wall receipt |
| `origin/main` tip (lane fetch) | `8d58a0c94` | post-#448/#449/#450 |
| Primary scripts | `bench/profile_em_phylo_scaling.jl`, `bench/profile_grouped_glmm.jl`, `bench/speed_bench.jl`, `bench/sparse_phy*_bench.jl` | EM scaling; grouped Laplace section walls; self-grid families |
| Banked TSVs | `bench/results/em_phylo_{scaling,after}_*.tsv`, `grouped_warm_68c2f067c.tsv`, `grouped_sections_after_*.tsv`, `s9_hessian_2x2_*.tsv`, `board_gllvm_20260923_8d58a0c94.tsv` | attested before/after + tip abs |

### DRModels (`DRM.jl`)

| Artifact | Path / PR | Role |
|---|---|---|
| Speed6 lane | `~/local-scratch/lanes/DRM.jl-speed6-20260919` · **PR #781 OPEN** | S3 section profile + S5* CHOLMOD/u0 plumbing; **no shipped fit-wall win on main** |
| Lane HEAD (sample) | `45ff42de8` / receipts at `9d709f008` | p∈{100,1000,5000} q4 phylo ML walls |
| `origin/main` tip | `2050350d4` (docs/reader); Dropbox checkout may lag | last 2 weeks = rename/docs/CI, not speed6 merge |
| Primary scripts | `bench/profile_q4_sections.jl`, `bench/head_to_head_q4_scaling.jl`, `bench/fit_phylo_{poisson,nb2,binomial,gamma_beta}.jl`, `bench/fit_crossed_*.jl`, `bench/bridge_six_cell_timing.jl` | q4 Gaussian phylo; non-Gaussian phylo; crossed RE; bridge H2H |
| Older evidence (still useful cells) | `docs/dev-log/evidence/fit-speed-h2h.tsv`, `engine-speed-grid.tsv` (2026-08-24) | Julia vs TMB medians; **re-anchor SHA before quoting as current** |

### HSquared.jl

| Artifact | Path / PR | Role |
|---|---|---|
| Speed12 | `~/local-scratch/lanes/HSquared.jl-speed12-20260919` | selinv three-arm + AI-REML section profile |
| Speed2b | `~/local-scratch/lanes/HSquared.jl-speed2b-20260919` · after-task `2026-09-17-reliability-memory-fix-and-selinv-bottleneck.md` | Totoro fill-471 follow-up |
| Landed on `main` | **#361** (28× selinv), **#363** (76× w/ SIMD), **#370** (post-fit 4.37→0.60 s), **#371** (AI-REML workspace 31.7→5.9 ms) | attested kernel / post-fit |
| `origin/main` tip | `e05fcf0e` (#371) | post-fit + selinv wave complete |
| Primary scripts | `bench/selinv_arms.jl`, `sim/cpu_fit_benchmark.jl`, `sim/profile_ai_reml_sections.jl`, `sim/phase_s6_asreml_wallclock_ladder.jl`, `sim/phase5_sparse_aireml_benchmark.jl` | kernel; fit wall; section; ASReml ladder (gated) |

---

## 2. Board matrix

Columns: `package | cell_id | DGP kind | script path | baseline SHA/date | current SHA | wall before | wall after | speedup | notes | status`

Paths are relative to the owning repo root (lane or Dropbox twin). Absolute lane roots named in §1.

### 2.1 GLLVModels — 10 cells

| package | cell_id | DGP kind | script path | baseline SHA/date | current SHA | wall before | wall after | speedup | notes | status |
|---|---|---|---|---|---|---|---|---|---|---|
| GLLVModels | `gllvm-gauss-unstruct-small` | Gaussian unstructured GLLVM; `(p,n,K)=(8,40,1)` | `bench/speed_bench.jl` | — | `8d58a0c94` Totoro | n/a | **0.0001 s** | n/a (abs) | closed-form; J=4 OB=1; TSV `board_gllvm_20260923_8d58a0c94.tsv` | `has_receipt` |
| GLLVModels | `gllvm-gauss-unstruct-large` | Gaussian unstructured; `(p,n,K)=(30,100,2)` | `bench/speed_bench.jl` | — | `8d58a0c94` Totoro | n/a | **0.0016 s** | n/a (abs) | same host/threads; evidence `board_cell2_large_8d58a0c94.log` | `has_receipt` |
| GLLVModels | `gllvm-gauss-phylo-em-p200` | Gaussian structured phylo EM; p=200, n=200, K_B=1 | `bench/profile_em_phylo_scaling.jl --gate` | `d56bccea4` (before sparse loglik) | `1f4ae6632` / #430 | 3.4 ms/iter | 3.5 ms/iter | ~1.0× | flat at small p; #430 body | `has_receipt` |
| GLLVModels | `gllvm-gauss-phylo-em-p1000` | Gaussian structured phylo EM; p=1,000 | same | `d56bccea4` | `1f4ae6632` / #430 | 83 ms/iter | 3.2 ms/iter | ~26× | TSV `em_phylo_*` | `has_receipt` |
| GLLVModels | `gllvm-gauss-phylo-em-p5000` | Gaussian structured phylo EM; p=5,000 | same | `d56bccea4` | `1f4ae6632` / #430 | 8,235 ms/iter | 17.9 ms/iter | ~460× | dramatic headline; EM not public default fitter | `has_receipt` |
| GLLVModels | `gllvm-pois-glmm-200x5` | non-Gaussian grouped Poisson GLMM; N=1000, G=200, p=1 | `bench/profile_grouped_glmm.jl` (warm / sections) | pre-S7c banked / `68c2f067c` before | `68c2f067c` after #430 | 0.1847 s | 0.1504 s | 1.23× | vs Latte 0.015 s → 10× remaining; `grouped_warm_68c2f067c.tsv` | `has_receipt` |
| GLLVModels | `gllvm-pois-glmm-5000x3` | non-Gaussian grouped; N=5000, G=500, p=3 (`glmm_5000x3_g500`) | `bench/profile_grouped_glmm.jl` + S9a real entry | BEFORE fd Hessian `edf7d39e0` | HESS `edf7d39e0` / #446 | 5.713 s | 4.300 s | 1.329× | bit-identical loglik; `s9_hessian_2x2_edf7d39e0.tsv` | `has_receipt` |
| GLLVModels | `gllvm-pois-glmm-200x5-hess` | same small fixture; Hessian path only | same | fd Hessian | grad_fd Hessian #446 | 0.113 s | 0.110 s | 1.028× | modest; do not conflate with S7c 1.23× | `has_receipt` |
| GLLVModels | `gllvm-nb2-or-binom-unstruct` | NB / Binomial unstructured Laplace; analytic vs finite | `bench/speed_bench.jl` | :finite same run | `8d58a0c94` Totoro | NB 0.2008 / Bin 0.1410 s (:finite @ 8×40×1) | NB **0.0222** / Bin **0.0130** s (:analytic) | **9.04× / 10.84×** vs :finite | Δll ≤1.7e-13; large grid companion NB 1.60 s / Bin 2.52 s analytic | `has_receipt` |
| GLLVModels | `gllvm-profile-ci-small` | post-fit profile CI `beta[1]` on 8×40×1 count | `bench/speed_bench.jl` (`PROFILE_CI=1`) | — | `8d58a0c` Totoro | n/a | Poisson **1.5809 s**; NB 2.5044; Binom 1.9279 | n/a (abs) | tip abs; log `board_profile_ci_8d58a0c.log` | `has_receipt` |

### 2.2 DRModels — 10 cells

| package | cell_id | DGP kind | script path | baseline SHA/date | current SHA | wall before | wall after | speedup | notes | status |
|---|---|---|---|---|---|---|---|---|---|---|
| DRModels | `drm-gauss-q4-phylo-p100` | Gaussian q=4 phylo ML; p=100 | `bench/profile_q4_sections.jl` | `a734d2b90` (S3 bank) | `cf058168b` (#781 merge) | 0.874 s | 0.976 s | ~0.90× (no gain) | Julia-vs-Julia same DGP; chol fallbacks 0; evidence `q4_sections_cf058168b.tsv` + `…a734d2b90.tsv` | `has_receipt` |
| DRModels | `drm-gauss-q4-phylo-p1000` | Gaussian q=4 phylo; p=1,000 | same | `a734d2b90` | `cf058168b` | 8.331 s | 10.375 s | ~0.80× (no gain) | Julia-vs-Julia; speed6 reuse = identity on this grid | `has_receipt` |
| DRModels | `drm-gauss-q4-phylo-p5000` | Gaussian q=4 phylo; p=5,000 | same | `a734d2b90` | tip abs Totoro `12ee8a8c2`; paired bank `9d709f008` | 38.772 s | tip Totoro **82.4455 s**; paired after 42.700 s | ~0.91× paired (no gain); tip abs n/a cross-host | Julia-vs-Julia paired on speed6 bank; tip abs Totoro JULIA_NUM_THREADS=4; evidence `docs/dev-log/evidence/2026-09-23-q4-p5000-totoro/` | `has_receipt` (paired + tip abs Totoro) |
| DRModels | `drm-gauss-locscale-n1000` | Gaussian unstructured loc-scale; n=1000 | `bench/results/speed6_end_arc_20260923/board_julia_reanchor_4a5840c15.tsv` | 2026-08-24 H2H julia 0.002 s | `4a5840c15` tip | 0.002 s | 0.00147 s | ~1.36× | **provisional pair**: tip DGP is testlike seed 20260815, not byte-identical R H2H fixture; do not headline | `has_receipt` (tip abs) / pair provisional |
| DRModels | `drm-gauss-relmat-G25` | Gaussian structured `relmat`; G=25 | same reanchor TSV | 2026-08-24 grid julia 0.006 s | `4a5840c15` | 0.006 s | 0.0174 s | ~0.35× | **provisional**: tip K=I+0.3 off-diag testlike, not engine-speed-grid fixture; tip absolute only is load-bearing | `has_receipt` (tip abs) / pair provisional |
| DRModels | `drm-bridge-gauss-locscale` | Gaussian bridge fixture n=180 | `bench/bridge_six_cell_timing.jl` (#372) | — | `cf058168b` | — | Julia 0.000446 s | 49.3× vs drmTMB 0.7.1 | **vs TMB**, not Julia-vs-Julia; sibling end-arc receipt | `has_receipt` (vs TMB) |
| DRModels | `drm-bridge-nbinom2` | NB2 bridge fixture n=180 | same + plus5 | — | `cf058168b` | — | Julia 0.001149 s | 18.3× vs TMB | non-Gaussian; vs TMB | `has_receipt` (vs TMB) |
| DRModels | `drm-bridge-poisson` | Poisson bridge fixture n=180 | `plus5` cohort | — | `cf058168b` | — | Julia 0.000235 s | 55.2× vs TMB | non-Gaussian; vs TMB | `has_receipt` (vs TMB) |
| DRModels | `drm-bridge-meta-V` | Gaussian structured `meta_V` | #372 six | — | `cf058168b` | — | Julia 0.001032 s | 15.5× vs TMB | structured; vs TMB | `has_receipt` (vs TMB) |
| DRModels | `drm-crossed-poisson` | non-Gaussian crossed RE Poisson | `bench/fit_crossed_poisson.jl` | #70 report (host≠) | `12ee8a8c2` Totoro | — | crossed_large 0.1884 s; fixedq_n20k 0.1742 s | — | tip abs Totoro JULIA_NUM_THREADS=1; evidence `docs/dev-log/evidence/2026-09-23-crossed-poisson-totoro/` (CSV+JSON) | `has_receipt` |

### 2.3 HSquared — 12 cells

| package | cell_id | DGP kind | script path | baseline SHA/date | current SHA | wall before | wall after | speedup | notes | status |
|---|---|---|---|---|---|---|---|---|---|---|
| HSquared | `hsq-animal-fit-q500` | Gaussian animal AI-REML; q≈500 | `sim/e2e_wall_receipts.jl --core` · TSV `sim/results/e2e_wall_receipts_e05fcf0e.tsv` | 2026-06-20 cpu_fit hist (soft) | `e05fcf0e` / #379 | 0.0230 s† | 0.0049 s | 4.7×† | † soft: June hist DGP ≠ gene-drop; Mac-only Totoro=N; absolute after attested | `has_receipt` |
| HSquared | `hsq-animal-fit-q2000` | Gaussian animal; q≈2,000 | same TSV | 2026-06-20 cpu_fit hist (soft) | `e05fcf0e` / #379 | 0.0840 s† | 0.0098 s | 8.6×† | † soft hist DGP; Mac-only | `has_receipt` |
| HSquared | `hsq-animal-fit-q10000` | Gaussian animal; q≈10,000 | same TSV + Totoro `e2e_wall_receipts_101aa483.tsv` | — | `101aa483` / Totoro | n/a | Mac 0.0312 s; Totoro 0.0345 s | n/a | absolute after only; Totoro=Y banked (no cross-host ×) | `has_receipt` |
| HSquared | `hsq-aireml-iter-workspace` | Gaussian AI-REML per-iter assemble+factorize | #371 body / `sim/profile_ai_reml_sections.jl` | pre-#371 | `e05fcf0e` / #371 | 31.7 ms | 5.9 ms | ~5.4× | workspace reuse across fit | `has_receipt` |
| HSquared | `hsq-reml-eval-once` | one `sparse_multi_reml_loglik` | #370 body | pre-#370 | #370 | 43.2 ms | 9.08 ms | ~4.8× | | `has_receipt` |
| HSquared | `hsq-postfit-uncertainty-3call` | post-fit uncertainty (3 separate calls) | #370 body | pre-#370 | #370 | 4.37 s | 0.60 s | ~7.3× | bitwise identical | `has_receipt` |
| HSquared | `hsq-postfit-multi-effect` | `multi_effect_uncertainty` path | #370 | — | #370 | — | 0.17 s | — | absolute after only in PR text | `has_receipt` |
| HSquared | `hsq-selinv-fill471-kernel` | selected-inverse kernel; q=20k, fill≈471 | `bench/selinv_arms.jl --totoro-arm` | pre-#361 | #361 / `b68bde5a` TSV | 416.8 s | 14.9 s | 28× | Totoro; bit-identical scatter | `has_receipt` |
| HSquared | `hsq-selinv-fill471-simd` | same + aligned-tail SIMD | same | pre-#361 | #363 | 416.8 s | 5.45 s | 76× | rtol-gated; with #361 | `has_receipt` |
| HSquared | `hsq-pev-reliability-q500` | post-fit PEV / reliability via selinv | same TSV + #350/#362 | dense banked 2026-06-20 | `e05fcf0e` / #379 | 0.0125 s‡ | 0.0003 s | 40.2×‡ | ‡ dense before = banked (live dense OpenBLAS hang); selinv after live; Mac-only | `has_receipt` |
| HSquared | `hsq-multi-effect-K2-q500` | multi-effect K=2; q≈500 | same TSV | dense NelderMead live | `e05fcf0e` / #379 | 2.552 s | 0.0036 s | 702×§ | § estimator confound (NelderMead vs AI-REML) disclosed; not pure LA claim; Mac-only | `has_receipt` |
| HSquared | `hsq-animal-fit-q20000-large` | large pedigree animal; q≈20,000 | same TSV + Totoro `e2e_wall_receipts_101aa483.tsv` | — | `101aa483` / Totoro | n/a | Mac 0.0402 s; Totoro 0.0477 s | n/a | absolute after only; Totoro=Y banked (no cross-host ×) | `has_receipt` |

---

## 3. Table summary (counts)

| Package | Cells | `has_receipt` | `needs_run` | `blocked` |
|---|---:|---:|---:|---:|
| GLLVModels | 10 | 10 | 0 | 0 |
| DRModels | 10 | 10 (3 Julia-vs-Julia q4 + 2 tip-abs provisional + 4 vs-TMB bridge + 1 Totoro crossed) | 0 | 0 |
| HSquared | 12 | 12 | 0 | 0 |
| **Total** | **32** | **32** | **0** | **0** |

\*DRM #781 MERGED. Julia-vs-Julia q4 shows **no wall gain** (identity). Attested multi-cell speedups on tip are **vs drmTMB**, not vs prior Julia.

---

## 4. Gaps (what the board still owes)

1. **Paired Julia-vs-Julia for DRM q4 — DONE (honest no gain).** `a734d2b90`→`cf058168b` p100/1000; p5000 paired on speed6 bank (`a734d2b90`→`9d709f008`). #781 **MERGED** `cf058168b`. Tip p5000 Totoro absolute DONE @ `12ee8a8c2` (82.4455 s).
2. **DRM bridge cohort re-anchored vs drmTMB 0.7.1** (10 OK cells, median 18.3×) in `docs/dev-log/evidence/2026-09-23-speed6-end-arc-cells/`. Board now carries 4 vs-TMB bridge rows + 2 provisional tip-abs locscale/relmat. Exact Aug-24 H2H Julia re-anchor (byte-identical fixtures) still owed; R harness failed (`Package DRM not found` — rename to DRModels).
3. **GLLVM unstructured + profile-CI** tip abs Totoro @ `8d58a0c94` / profile follow-up `8d58a0c` (`board_gllvm_20260923_8d58a0c94.tsv` + `board_profile_ci_8d58a0c.log`). All 10 GLLVM board cells `has_receipt`.
4. **H² end-to-end fit wall** — board `needs_run` animal/PEV cells flipped `has_receipt` via #379 TSV `e05fcf0e` (gene-drop; soft hist †/‡ fenced). Totoro absolute walls for q10k/q20k banked at `101aa483` (0.0345 s / 0.0477 s). Still owed: same-DGP before/after; optional Totoro re-time of q500/q2000/PEV/multi-effect; public README/NEWS speed claim remains withheld. Projected SelectedInversion (#378) stays fenced (unwired).
5. **Cross-package comparability.** Do not put Latte gap, TMB H2H, and selinv kernel ms in one “headline ×” without labeling comparator. Board rows already separate them in `notes`.
6. **ASReml public compare (H²)** stays gated (`phase_s6_asreml_wallclock_ladder.jl`); not a board cell until a paired receipt + Rose wording.
7. **Spatial / animal** GLLVM structured cells: phylo EM covers structured Gaussian; spatial/animal fit-wall cells are not in the Sept speed TSV set → add later or mark blocked if engine route not ready.
8. **Machine mix:** Mac M1 Ultra (GLLVM/DRM TSVs) vs Totoro EPYC (H² selinv). Never compute a cross-host speedup.

---

## 5. Suggested fill order (after G0; not this inventory)

1. Re-anchor DRM evidence H2H + q4 paired tip (Lane B).
2. Run GLLVM `speed_bench` quick+one large + one NB2/Binomial (Lane A tooling).
3. Run H² `cpu_fit_benchmark` three q-rungs + PEV at q=500 on `e05fcf0e` (Lane C).
4. Freeze board TSV directory convention: `bench/results/board_<pkg>_<yyyymmdd>_<sha>.tsv` with header threads/BLAS/host.

---

## 6. Source index (PRs with measured numbers, ~2026-09-09…23)

| PR | Package | Measured claim (attested) |
|---|---|---|
| #430 | GLLVModels | EM p5k 8235→17.9 ms/iter; warm GLMM 0.185→0.150 s |
| #446 | GLLVModels | `glmm_5000x3_g500` 5.71→4.30 s (1.33×); small Hess 1.03× |
| #447 | GLLVModels | NM alone slower on large; moment_start not default win |
| #448 | GLLVModels | coverage; not fit-wall |
| #781 | DRModels | **MERGED** `cf058168b`; q4 Julia-vs-Julia identity/no gain; 10 bridge cells vs drmTMB 0.7.1 median 18.3× |
| #361 | HSquared | selinv 416.8→14.9 s (28×) |
| #363 | HSquared | +SIMD → 5.45 s (76×) |
| #370 | HSquared | post-fit 4.37→0.60 s; reml eval 43.2→9.08 ms |
| #371 | HSquared | AI-REML iter 31.7→5.9 ms |

Rose fence: no README/NEWS speed claim from this board until each quoted cell is `has_receipt` with dated TSV + after-task.
