# Three-package speedup report (publishable scratch)

**Status:** DRAFT toward publishable scratch · **Phase B UNLOCKED** (Shinichi *finish all parts*; see `2026-09-23-speed-then-20-report-sequence.md`) · bank as siblings return · do **not** claim “20 done” until every package hits banked aim · **not** a README/NEWS claim surface  
**Date:** 2026-09-23  
**Filled now:** GLLVM **20** · DRM **20** · H² **20** = **60** attested rows (aims met). Fir DRAC GLLVM job `61148081` 7/7 ok (cross-host abs only); Fir DRM job `61148082` harness errored — board authority remains Totoro CSV `e9d50a110`.  
**Authority for numbers:** board `2026-09-23-three-package-speed-board.md` (floor 32/32) + H² kinds expand TSV `e2e_wall_receipts_fdc43845.tsv`  
**Closeout digest:** `docs/dev-log/plans/2026-09-23-arc-status-digest.md` (floor closeout; Phase B continues here)

Active lenses: Shannon, Ada, Rose (perspectives). No subagents for this scaffold.

---

## 1. Purpose

This report tracks **measured wall times** for the three Julia twins against their R counterparts **where a paired cell exists**, and Julia-vs-Julia before/after walls where the arc improved an engine path.

| Twin (Julia) | R pair (when cited) |
|---|---|
| **GLLVModels.jl** (repo folder `GLLVM.jl`) | **gllvmTMB** |
| **DRModels.jl** (repo folder `DRM.jl`) | **drmTMB** |
| **HSquared.jl** | **hsquared** (R surface; ASReml / other comparators gated until paired receipt) |

What this file is for:

- A single place to paste attested × and absolute walls as the board grows toward 20 cells per package.
- Honest labels for **comparator** (Julia-vs-Julia · vs TMB · vs Latte oracle · kernel-only · soft-hist).

What this file is **not**:

- A public README speed claim.
- Proof that every model class is fast.
- A license to mix Mac and Totoro walls into one ×.

---

## 2. Methods

### Hosts and threads

| Host | Typical regime (standing board discipline) |
|---|---|
| **Mac Studio** (M1 Ultra) | `JULIA_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1` unless a receipt says otherwise |
| **Totoro** (EPYC; D-50 compute) | Many H² selinv cells used `threads=1`; DRM/GLLVM tip abs often `JULIA_NUM_THREADS=4` or `1` as named on the cell |

**Rule:** never form a speedup by dividing walls from different hosts. Absolute walls may be listed side-by-side with host labelled; × stays **within-host**.

### Tip SHAs (attested at board close · 2026-09-23)

Use these as the “current” column unless a newer cell receipt names another SHA.

| Package | Tip / receipt SHAs (short) | Role |
|---|---|---|
| GLLVModels | `8d58a0c94` (board unstructured / PROFILE_CI Totoro); prior wave `#430` `1f4ae6632`, `#446` `edf7d39e0` | fit-wall board cells |
| GLLVModels | `#449` `f36def049…` Latte-kernel (default **OFF**); `#451` `4e976e259…` Totoro board receipts | post-board merges (digest) |
| DRModels | `#781` merge `cf058168b`; tip abs Totoro `12ee8a8c2` (q4 p5000 / crossed); H2H re-anchor `1b8e81c` | Julia-vs-Julia q4 + vs-TMB H2H |
| DRModels | `#803`–`#807` evidence PRs (`3107f0c66…` … `e9d50a110…`) | end-arc attested walls |
| HSquared | `e05fcf0e` (#371 / #379 Mac e2e); Totoro e2e `101aa483` (#379/#380); selinv `#361`/`#363`; kinds expand **`fdc43845`** | fit + kernel + Phase B kinds |

### How a cell enters this report

1. Board row status `has_receipt` with dated TSV / after-task / PR body.
2. Comparator labelled in the notes column.
3. Rose wording check before any reader-facing quote outside this scaffold.

---

## 3. Floor tables (board 32/32)

Source tables copied from the speed board §2 (2026-09-23). Phase B additions are in §4.

### 3.1 GLLVModels: cells 1–10

| # | cell_id | Comparator | Wall / metric | Speedup | PR / SHA | Notes |
|---:|---|---|---|---|---|---|
| 1 | `gllvm-gauss-unstruct-small` | abs (Totoro) | after **0.0001 s** `(p,n,K)=(8,40,1)` | n/a (abs) | tip `8d58a0c94` · #451 | closed-form; J=4 OB=1 |
| 2 | `gllvm-gauss-unstruct-large` | abs (Totoro) | after **0.0016 s** `(30,100,2)` | n/a (abs) | `8d58a0c94` | same host/threads |
| 3 | `gllvm-gauss-phylo-em-p200` | Julia-vs-Julia EM | 3.4 → 3.5 ms/iter | ~1.0× | #430 `1f4ae6632` | flat at small p |
| 4 | `gllvm-gauss-phylo-em-p1000` | Julia-vs-Julia EM | 83 → 3.2 ms/iter | ~26× | #430 | TSV `em_phylo_*` |
| 5 | `gllvm-gauss-phylo-em-p5000` | Julia-vs-Julia EM | 8,235 → 17.9 ms/iter | ~460× | #430 | EM path; not public default fitter |
| 6 | `gllvm-pois-glmm-200x5` | Julia-vs-Julia warm | 0.1847 → 0.1504 s | 1.23× | #430 `68c2f067c` | vs Latte 0.015 s residual ~10× (not a shipped × claim) |
| 7 | `gllvm-pois-glmm-5000x3` | Julia-vs-Julia HESS | 5.713 → 4.300 s | 1.329× | #446 | bit-identical loglik |
| 8 | `gllvm-pois-glmm-200x5-hess` | Julia-vs-Julia Hess path | 0.113 → 0.110 s | 1.028× | #446 | do not conflate with 1.23× warm |
| 9 | `gllvm-nb2-or-binom-unstruct` | analytic vs `:finite` | NB 0.2008→0.0222 s; Bin 0.1410→0.0130 s | **9.04× / 10.84×** | tip `8d58a0c94` Totoro | Δll ≤1.7e-13 |
| 10 | `gllvm-profile-ci-small` | abs (Totoro) | Poisson **1.5809 s** (NB 2.5044; Binom 1.9279) | n/a (abs) | tip `8d58a0c` · PROFILE_CI | post-fit profile CI |

### 3.2 DRModels: cells 1–10

| # | cell_id | Comparator | Wall / metric | Speedup | PR / SHA | Notes |
|---:|---|---|---|---|---|---|
| 1 | `drm-gauss-q4-phylo-p100` | Julia-vs-Julia | 0.874 → 0.976 s | ~0.90× (no gain) | #781 `cf058168b` | identity on this grid |
| 2 | `drm-gauss-q4-phylo-p1000` | Julia-vs-Julia | 8.331 → 10.375 s | ~0.80× (no gain) | #781 | same |
| 3 | `drm-gauss-q4-phylo-p5000` | Julia-vs-Julia paired + tip abs | paired 38.772 → 42.700 s; tip Totoro **82.4455 s** | ~0.91× paired; tip abs n/a as × | tip `12ee8a8c2` · #806 | no cross-host × |
| 4 | `drm-gauss-locscale-n1000` | **vs drmTMB 0.7.1** | tmb 0.024 s → julia 0.000553 s | **43.4×** | tip `1b8e81c` · #807 | Aug-24 H2H re-anchor Totoro |
| 5 | `drm-gauss-relmat-G25` | Julia abs + Δll H2H | julia **0.000906 s**; TMB wall load-fenced | n/a quiet × (TMB load~114) | `1b8e81c` · #807 | Julia abs + Δll≈0 load-bearing |
| 6 | `drm-bridge-gauss-locscale` | **vs drmTMB 0.7.1** | Julia 0.000446 s | 49.3× | `cf058168b` · #803 | bridge cohort |
| 7 | `drm-bridge-nbinom2` | **vs drmTMB 0.7.1** | Julia 0.001149 s | 18.3× | same | non-Gaussian |
| 8 | `drm-bridge-poisson` | **vs drmTMB 0.7.1** | Julia 0.000235 s | 55.2× | same | non-Gaussian |
| 9 | `drm-bridge-meta-V` | **vs drmTMB 0.7.1** | Julia 0.001032 s | 15.5× | same | structured |
| 10 | `drm-crossed-poisson` | abs (Totoro) | crossed_large 0.1884 s; fixedq_n20k 0.1742 s | n/a (abs) | tip `12ee8a8c2` · #805 | J=1 |

### 3.3 HSquared: cells 1–12

| # | cell_id | Comparator | Wall / metric | Speedup | PR / SHA | Notes |
|---:|---|---|---|---|---|---|
| 1 | `hsq-animal-fit-q500` | soft hist → tip | 0.0230 → 0.0049 s | 4.7×† | `e05fcf0e` · #379 | † soft: June hist DGP ≠ gene-drop; Mac |
| 2 | `hsq-animal-fit-q2000` | soft hist → tip | 0.0840 → 0.0098 s | 8.6×† | same | † soft hist; Mac |
| 3 | `hsq-animal-fit-q10000` | abs | Mac 0.0312 s; Totoro 0.0345 s | n/a | Totoro `101aa483` · #380 | no cross-host × |
| 4 | `hsq-aireml-iter-workspace` | Julia-vs-Julia | 31.7 → 5.9 ms | ~5.4× | #371 | workspace reuse |
| 5 | `hsq-reml-eval-once` | Julia-vs-Julia | 43.2 → 9.08 ms | ~4.8× | #370 | one loglik eval |
| 6 | `hsq-postfit-uncertainty-3call` | Julia-vs-Julia | 4.37 → 0.60 s | ~7.3× | #370 | bitwise identical |
| 7 | `hsq-postfit-multi-effect` | abs | after 0.17 s |: | #370 | absolute after only |
| 8 | `hsq-selinv-fill471-kernel` | Julia-vs-Julia kernel | 416.8 → 14.9 s | 28× | #361 Totoro | bit-identical scatter |
| 9 | `hsq-selinv-fill471-simd` | Julia-vs-Julia kernel | 416.8 → 5.45 s | 76× | #363 | with #361; rtol-gated |
| 10 | `hsq-pev-reliability-q500` | dense banked → selinv | 0.0125 → 0.0003 s | 40.2×‡ | `e05fcf0e` · #379 | ‡ dense before banked; Mac |
| 11 | `hsq-multi-effect-K2-q500` | NelderMead vs AI-REML | 2.552 → 0.0036 s | 702×§ | same | § estimator confound disclosed |
| 12 | `hsq-animal-fit-q20000-large` | abs | Mac 0.0402 s; Totoro 0.0477 s | n/a | `101aa483` · #380 | no cross-host × |

### Board / Phase B count (live)

| Package | Floor (board) | Phase B new | **Filled in this report** | Aim | Empty slots |
|---|---:|---:|---:|---:|---|
| GLLVModels | 10 | **10** | **20** | ~20 | (GLLVM aim met; Fir `61148081` abs confirm) |
| DRModels | 10 | **10** | **20** | ~20 | (DRM aim met; Totoro `e9d50a110`) |
| HSquared | 12 | **8** | **20** | ~20 | (H² aim met) |
| **Total** | **32** | **28** | **60** | ~60 | (aims met) |

---

## 4. Phase B cells (fill as receipts land)

Leave rows blank until a dated TSV / after-task exists. Do **not** invent ×.

### 4.1 GLLVModels: cells 11-20 BANKED (Phase B first-wave)

Lane: `GLLVM.jl-speed-diversity-20260923` · `bench/speed_board_firstwave.jl` · Totoro tip `4e976e2` · J=4 OB=1 · median of 3 warm reps · evidence `docs/dev-log/evidence/2026-09-23-speed-firstwave-totoro/board_gllvm_firstwave_4e976e2.tsv`. Plan §4.1 cell_ids. No Latte default flip. No R pair (julia_abs_only). Load at launch ≈229 (walls may be load-inflated).

| # | cell_id | Comparator | Wall / metric | Speedup | PR / SHA | Notes |
|---:|---|---|---|---|---|---|
| 11 | `gllvm-binom-glmm-200x5` | abs (Totoro) | **0.2369 s** | n/a (abs) | tip `4e976e2` | grouped Binomial; identity green #448 |
| 12 | `gllvm-gauss-lv-t4-p20n500` | abs (Totoro) | **0.0088 s** | n/a (abs) | same | closed-form; post-compile median (not T4 campaign total) |
| 13 | `gllvm-pois-lv-t4-p20n500` | abs (Totoro) | **1.6459 s** | n/a (abs) | same | Laplace LV |
| 14 | `gllvm-nb2-lv-t4-p20n500` | abs (Totoro) | **4.8812 s** | n/a (abs) | same | slower path; quieter-host T4 hist was ~97.9 s total |
| 15 | `gllvm-gauss-unstruct-p50n2k` | abs (Totoro) | **0.5197 s** | n/a (abs) | same | (50,2000,2) |
| 16 | `gllvm-gauss-phylo-fit-p200` | abs (Totoro) | **0.0418 s** | n/a (abs) | same | `fit_phylo_gaussian` non-EM; label ≠ EM ms/iter |
| 17 | `gllvm-profile-ci-glmm` | abs (Totoro) | **0.0091 s** | n/a (abs) | same | **Wald** via `grouped_nongaussian_intervals` (profile not admitted) |
| 18 | `gllvm-latte-gap-200x5` | labelled vs Latte banked 0.015 s | tip **0.3830 s** → **25.53×** labelled | labelled only | same | no default flip; load-inflated vs quiet ~8× hist |
| 19 | `gllvm-pois-phylo-small` | abs (Totoro) | **0.0191 s** | n/a (abs) | same | `fit_phylo_glm` Poisson 6×12 |
| 20 | `gllvm-spatial-gauss-small` | abs (Totoro) | **0.0130 s** | n/a (abs) | same | SPDE Matérn 8×8 M=16 |

Sibling diversity inventory (alternate cell_ids, not these slots): 8 kinds in `…/speed-diversity-totoro/` (ordinal/ZIP/ZINB/…).

Fir DRAC confirm (job `61148081`, Julia 1.11.3, J=4): binom 0.158 s; gauss-lv 0.0067 s; pois-lv 1.209 s; nb2-lv 4.256 s; unstruct-p50n2k 0.100 s; phylo-fit-p200 0.020 s; profile-ci-glmm 1.191 s. Evidence `…/speed-firstwave-drac/gllvm_fir_61148081_combined.tsv`. Same cells as Totoro 11–17; **no cross-host ×**.

### 4.2 DRModels: cells 11–20 — BANKED (Phase B first-wave)

Lane: `DRM.jl-speed-kinds-20260923` · Totoro tip `e9d50a110` · Julia 1.12.6 · `JULIA_NUM_THREADS=1` · evidence
`docs/dev-log/evidence/2026-09-23-speed-kinds-toward20/board_drm_first_wave_20260923_e9d50a110.csv`
(on DRModels PR #808). Absolute walls only. **No ×.** Fir job `61148082` completed but every cell
errored in `bench/speed_board_firstwave_drm.jl` (MethodError on `tree=`/`se=` kwargs); Fir TSVs are
negative receipts only — do not quote Fir walls.

| # | cell_id | Comparator | Wall / metric | Speedup | PR / SHA | Notes |
|---:|---|---|---|---|---|---|
| 11 | `drm-phylo-poisson` | abs (Totoro) | **0.0223 s** | n/a (abs) | tip `e9d50a110` · #808 | phylo Poisson tip smoke p=100 |
| 12 | `drm-phylo-nb2` | abs (Totoro) | **0.0348 s** | n/a (abs) | same | phylo NB2 p=100 |
| 13 | `drm-phylo-binomial` | abs (Totoro) | **0.0232 s** | n/a (abs) | same | phylo binomial p=128 |
| 14 | `drm-h2h-q4-vs-tmb-p1000` | abs (Totoro) | **20.8147 s** | n/a (abs) | same | Julia arm q4 p=1000; TMB pair not run this receipt |
| 15 | `drm-phylo-gamma` | abs (Totoro) | **0.0717 s** | n/a (abs) | same | phylo gamma p=128 |
| 16 | `drm-crossed-binomial` | abs (Totoro) | **0.0374 s** | n/a (abs) | same | crossed binomial G=H=20 n=1000 |
| 17 | `drm-biv-gauss-rho12` | abs (Totoro) | **0.0572 s** | n/a (abs) | same | bivariate residual rho12 |
| 18 | `drm-profile-ci-locscale` | abs (Totoro) | **0.0160 s** | n/a (abs) | same | profile CI loc-scale gaussian n=600 |
| 19 | `drm-animal-gauss` | abs (Totoro) | **0.0177 s** | n/a (abs) | same | animal() Gaussian A supplied G=60 |
| 20 | `drm-lss-sd-slope` | abs (Totoro) | **0.0029 s** | n/a (abs) | same | sd(id) ~ sex LSS n=480 |

Honest standing fact: q4 Julia-vs-Julia on the speed6 grid shows **no wall gain**. These Phase B rows are tip-abs diversity, not a Julia-vs-Julia × claim.

### 4.3 HSquared: cells 13–20 — BANKED (Phase B)

Source: `sim/results/e2e_wall_receipts_fdc43845.tsv` · after-task `2026-09-23-speed-kinds-expand.md` · host **Totoro** · Julia **1.12.6** · `JULIA_NUM_THREADS=1` · `OPENBLAS_NUM_THREADS=1` · `taskset -c 0-15`. Absolute walls only (`pair = measured_now`). **No ×** (no before column).

| # | cell_id | Comparator | Wall / metric | Speedup | PR / SHA | Notes |
|---:|---|---|---|---|---|---|
| 13 | `hsq-maternal-q80` | abs (Totoro) | **0.0392 s** | n/a (abs) | `fdc43845` | direct–maternal dense; experimental; **conv=false** fenced |
| 14 | `hsq-genomic-greml-q200` | abs (Totoro) | **0.0565 s** | n/a (abs) | `fdc43845` | GBLUP / GREML supplied Ginv; conv=true |
| 15 | `hsq-repeatability-sparse-q200` | abs (Totoro) | **0.0033 s** | n/a (abs) | `fdc43845` | sparse animal+PE; conv=true |
| 16 | `hsq-fa-t4k1-q48` | abs (Totoro) | **0.9104 s** | n/a (abs) | `fdc43845` | FA t=4 K=1; dense MV; **conv=false** fenced |
| 17 | `hsq-multivar-us-t2-q80` | abs (Totoro) | **1.7585 s** | n/a (abs) | `fdc43845` | unstructured multi-trait; **conv=false** fenced |
| 18 | `hsq-animal-depth3-q500` | abs (Totoro) | **0.0108 s** | n/a (abs) | `fdc43845` | 3-gen window-mated pedigree; conv=true |
| 19 | `hsq-reliability-selinv-q2000` | abs (Totoro) | **0.0041 s** | n/a (abs) | `fdc43845` | `reliability(:selinv)` post-fit; not projected SelectedInversion.jl |
| 20 | `hsq-halfsib-q5000` | abs (Totoro) | **0.0410 s** | n/a (abs) | `fdc43845` | larger halfsib animal REML; conv=true |

Still gated (not in this table): ASReml ladder (needs paired receipt + Rose wording). Projected SelectedInversion.jl stays unwired (#378). Do not form Julia 1.10↔1.12 or Mac↔Totoro × from these abs walls.

---

## 5. Rose fences (standing)

1. **Curated cells ≠ all models.** Forty banked rows (or sixty if aim fills) are a **sample of fittable DGPs**, not the capability ledger.
2. **No cross-host ×.** Mac Studio and Totoro walls may appear as absolute columns; speedups stay same-host.
3. **No cross-version ×.** H² Phase B walls are Julia 1.12.6 Totoro; prior #380 large walls used 1.10.12 — abs only.
4. **Advisory Frozen-R.** GLLVModels CI Frozen-R 0.7.0 family smoke may be red (rebuilt oracle); **advisory**, not a speed receipt.
5. **Comparators differ.** Do not collapse Julia-vs-Julia, vs drmTMB/gllvmTMB, vs Latte oracle, selinv kernel ms, and tip-abs into one package headline ×.
6. **Latte `diag_precision_kernel` default OFF** (#449). No public wall quote from that path until owner default-ON (+ optional retime).
7. **Soft † / ‡ / § / conv=false footnotes stay.** Soft-hist DGP mismatch, banked dense, estimator-confound, and non-converged validation-scale walls are not pure LA claims.
8. **README / NEWS.** Withheld until Rose signs a claim surface. This file is publishable **scratch** evidence inventory, not marketing.
9. **Phase B unlock ≠ public claim.** All three packages at 20 attested abs/× rows in this scratch report; README/NEWS still withheld.

---

## 6. R-package issue links

Tracking issues for reader-facing follow-up in the R twins (verified open 2026-09-23).

| R package | Issue | Purpose | Status |
|---|---|---|---|
| gllvmTMB | [#1319](https://github.com/itchyshin/gllvmTMB/issues/1319) | Gaussian TMB-Laplace vs Julia closed-form / `engine=julia` docs | **OPEN** |
| gllvmTMB | [#1320](https://github.com/itchyshin/gllvmTMB/issues/1320) | Profile CI cost documentation | **OPEN** |
| drmTMB | [#1420](https://github.com/itchyshin/drmTMB/issues/1420) | `beta`→`beta_family`: bare `family=beta()` / NEWS overclaim of `drmTMB::beta()` | **OPEN** |
| hsquared | — | **none filed** (deliberate) | skip |

**Skips (deliberate — do not file):** no fresh × claim from this scaffold; NB/Binom analytic speedup not escalated as an R-issue (board cell 9 stays Julia-only evidence); **no ×-gap speed issues for drmTMB** (paired TMB walls often ms-scale — large ×, small absolute walls; #1420 is the API/NEWS footgun only); **no hsquared issue** this arc (Julia abs kinds only; no R twin parallel harness).

---

## 7. Source index (PRs / receipts with measured numbers)

| PR / receipt | Package | Attested claim (short) |
|---|---|---|
| #430 | GLLVModels | EM p5k 8235→17.9 ms/iter; warm GLMM 0.185→0.150 s |
| #446 | GLLVModels | `glmm_5000x3_g500` 5.71→4.30 s (1.33×); small Hess 1.03× |
| #447 | GLLVModels | NM alone slower on large; moment_start not default win |
| #448 | GLLVModels | coverage; not fit-wall |
| #449 | GLLVModels | Latte-kernel diag / identical log-link (default OFF) |
| #451 | GLLVModels | Totoro unstructured + PROFILE_CI board receipts |
| #781 | DRModels | MERGED; q4 Julia-vs-Julia no gain; bridge vs drmTMB 0.7.1 |
| #803–#807 | DRModels | end-arc attested walls / Totoro tip / H2H re-anchor |
| #361 | HSquared | selinv 416.8→14.9 s (28×) |
| #363 | HSquared | +SIMD → 5.45 s (76×) |
| #370 | HSquared | post-fit 4.37→0.60 s; reml eval 43.2→9.08 ms |
| #371 | HSquared | AI-REML iter 31.7→5.9 ms |
| #378–#380 | HSquared | e2e wall receipts + Totoro q10k/q20k |
| kinds expand `fdc43845` | HSquared | 8 tip-abs Totoro kinds (report cells 13–20); no × |

---

## 8. Changelog

| Date | Change |
|---|---|
| 2026-09-23 | Scaffold created from board 32/32 + arc-status digest. Empty cells 11–20 (GLLVM/DRM) and 13–20 (H²). R-issue links left TODO. No invented ×. |
| 2026-09-23 | §6: filed gllvmTMB #1319 (Gaussian TMB-Laplace vs Julia / engine=julia docs) and #1320 (profile CI cost docs). Skips noted: no fresh × claim; NB/Binom analytic. drmTMB / hsquared still TODO. |
| 2026-09-23 | §6: drmTMB [#1420](https://github.com/itchyshin/drmTMB/issues/1420) (`beta`→`beta_family` bare `family=beta()` / NEWS overclaim). Skip note: no ×-gap speed issues (TMB walls often ms-scale). hsquared still TODO. |
| 2026-09-23 | Phase B UNLOCKED (*finish all parts*). Banked H² cells 13–20 from `fdc43845` TSV (abs only; conv=false fenced on 3 rows). Counts **10 / 10 / 20**. GLLVM+DRM 11–20 still TODO. §6 hsquared = none (deliberate). Rose fences refreshed. No invented ×. |
| 2026-09-23 | GLLVM Phase B first-wave **10/10** Totoro (`4e976e2`; load≈229). Report slots 11–20 filled. Counts **20 / 10 / 20**. Profile-ci Wald fallback; latte-gap labelled only (no default flip). No invented ×. |
