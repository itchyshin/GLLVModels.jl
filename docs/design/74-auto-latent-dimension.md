# 74: Estimating the number of latent dimensions from data (auto-d)

Status: DRAFT for maintainer sign-off (G1). Lane `auto-d-20260926` (Julia) with the R twin on
gllvmTMB branch `claude/lane-auto-d-r-20260926` (vault D-292). Nothing here is merged.

## Destination

A user fits a GLLVM without supplying the number of latent dimensions. The package chooses it by
one documented rule, returns the chosen value with the per-candidate criterion values and every
rejected candidate with its reason, the rule's recovery rate is measured on simulated data across
n, p and family, and every place a user reads an interval says it is conditional on the choice.

## What the literature says (vault `dr-auto-d-latent-dimension-selection`, NotebookLM `c2c564e3`, 34 sources)

- **Feasible, and the gap is real.** Gaussian factor-number rules are well validated (Bai & Ng
  2002; Ahn & Horenstein 2013; Dobriban & Owen 2019: near-100% recovery once n, p are in the
  hundreds, failing for weak factors). **No published benchmark checks any rule on count or
  binary GLLVM data at ecological n, p.** Ecology packages (gllvm, boral, glmmTMB `rr()`, HMSC)
  leave d to the user and suggest "compare AIC/BIC/CV"; on one real dataset (gllvm 2.0 beetles,
  n = 88, m = 68) AIC picks 3 and BIC picks 2.
- **Chen & Li (2022, Biometrika; arXiv:2010.02326, read directly 2026-09-27)** give the only
  criterion with a consistency proof for binary and count factor models:
  JIC(K) = −2 l̂_K + K (N ∨ J) log{n / (N ∨ J)}, where l̂_K is the **joint** likelihood (factor
  scores estimated as parameters, constrained joint MLE), N persons, J items, n observed cells.
  Their simulation is binary only, J = 100–400 with N = J or 5J, K* = 3; the paper itself reports
  over-selection when N = J and J is small, and under-selection of a weak factor when N = 5J.
  Our fits maximise the **marginal** (Laplace) likelihood and ecological p is 10–20, so JIC does
  not transfer directly; the paper notes the marginal likelihood approaches the joint one only
  when both N and J are large. Not a v1 candidate.
- **Ordered factor LASSO** (Hui, Tanaka & Warton 2018, Biometrics) is the only GLLVM-native
  single-fit rule. No software implementation was found and no precedent for a non-smooth group
  penalty inside a Laplace/TMB fit. Parked as a follow-up lane (Shinichi, 2026-09-26).
- **The d vs d+1 likelihood-ratio test is non-standard** (Hayashi, Bentler & Yuan 2007; Drton
  2009). Only a parametric bootstrap is a valid reference. `src/boundary_inference.jl` no longer
  lists K-selection as a chi-bar-squared use case.
- **Nothing addresses inference after choosing d in GLLVMs.**

## What we measured (this lane)

- Existing NB fits can sit in poor optima while reporting `converged = true` (NB2, n = 300,
  p = 20, true K = 3; `LOOP/lanes/auto-d-20260926/pilot/runaway_probe.txt`):

  | K | logLik | converged | max latent SD | median | ratio |
  |---|---|---|---|---|---|
  | 2 | −18874 | true | 26.9 | 2.69 | 9.1 |
  | 3 | −19113 | true | 8.7 | 1.40 | 6.1 |
  | 4 | −20240 | true | (not probed) | | |
  | 5 | −16720 | true | 2.45 | 1.41 | 1.75 |

  K = 5 (20 min) reached a region ~2400 logLik units above K = 2–4 with healthy loadings, so the
  K = 2–4 fits are poor optima, not the maxima. The old `select_lv` picks K = 2, itself a runaway.
  The guarded `select_lv` rejects K = 2 (runaway) and K = 4 (non-monotone) and would choose K = 5
  by BIC: it stops the runaway being chosen but cannot recover K = 3, because the K = 3 fit is a
  bad optimum. **Recovery here needs better NB optimisation (multi-start or warm start in the
  per-species NB route), which the guard cannot supply.** An earlier note in this lane called
  K = 5 a likely runaway; the probe shows it is not.
- Binomial, same size: K = 4 unconverged, latent SD 156, ratio 106 (a clear separation runaway,
  caught by both checks); AIC on the old `select_lv` would have chosen it. K = 3 and K = 5 healthy.
- Recovery grid (4 families × n {30,60,120,300} × p {10,20} × true K {1,2,3} × 200 reps, K fitted
  1..K+2, existing code): running on DRAC nibi, array 22744942. Table goes here when harvested.

## Built so far (lane branch, not merged)

1. `select_lv` guard: every attempted K recorded with a status; failed, unconverged and
   non-monotone K are never chosen; interrupts no longer swallowed (commit caddc8653).
2. Warm-start retry from the (K−1) solution plus a lower-triangular new column, where the
   fitter accepts `β_init`/`Λ_init` (same commit).
3. Runaway detector: scale check (max loading row norm = latent SD on the link scale > 10, any
   non-identity link) and binomial ratio check (≥ 25), from the gllvmTMB runaway study
   (commit 9653b1778). Thresholds provisional; recalibrate on the grid's healthy fits.
4. R twin: the same guard and detector in gllvmTMB `select_lv()` (branch claude/lane-auto-d-r-20260926, commit 978f4bba2; 15 new + 70 existing tests pass). Warm start uses `control(start_from = <accepted fit>)`: matching blocks carry over, the new column starts at the default. The table keeps rejected rows with `status` and `message`.

Tests: 50/50 in `test/test_model_selection.jl` (16 existing + 34 new).

## Decisions needed at G1 (each with a recommendation)

| ticket | question | recommendation | why |
|---|---|---|---|
| T2 route | fit-and-compare, shrinkage, spectral, or hybrid? | **Fit-and-compare with the two safeguards (v1). Spectral guess to narrow the K window later (v1.1). OFAL in a separate lane.** | Reuses `select_lv`; no new estimator; the failure we found is in fits, not in criteria. |
| T2 criterion | default criterion | **Decide from the grid.** Provisional: BIC. Candidates on the grid: AIC, BIC log(p·n) (current), BIC log(n sites). Chen–Li JIC read and set aside (joint likelihood, J ≥ 100). | No published discrete-data evidence exists; our grid is the evidence. |
| T4 API, Julia | what does omitting K do? | **Omitting K runs `select_lv` and returns the chosen fit**, printing the candidate table once (`@info`); `select_lv` stays the way to get the full record. Today omitting K throws, so nothing that works now changes. Default `Kmax = min(5, p − 1)`. | Shinichi: "OK what if we do not supply d - yes". Return type stays a fit, so downstream code is unchanged. |
| T4 API, R | what does omitting `d` in `latent()` do? | **Flag: today omitting `d` silently means d = 1**, so switching it to auto changes existing users' fits. Recommend `d = "auto"` now, and switching the default in a later minor release with a NEWS warning. | Julia's switch is error → behaviour; R's would be behaviour → different behaviour. |
| T5 caveat | wording only, or propagate K uncertainty? | **Wording now** (below). A bootstrap that re-selects K per replicate is a later option. | Honest now; the bootstrap costs Kmax fits per replicate. |
| T6 scope | which routes get auto in v1? | **Default family route, including NB/Beta via their per-species dispersion route.** Those routes need `β_init`/`Λ_init` in `grouped_dispersion.jl` for the warm-start retry; without it the guard still rejects bad K. That file belongs to the overnight lane until 11:00Z; raise after. | NB is the most-used count family. |

## Caveat text (for the docstring and docs page)

> The number of latent dimensions is chosen from the same data the model is then fitted to.
> Intervals, p-values and tests reported for that fit are conditional on the chosen number and do
> not include uncertainty about it; when the top two candidates are close, treat the choice as
> uncertain. Species correlations, variance partitions and ordination axes all depend on the
> chosen number, and individual axes can change meaning when it changes. Under model
> misspecification the chosen number tends to grow with sample size, so read it as "dimensions
> the data support at this sample size", not as the number of true gradients. Recovery rates in
> the documentation hold for the simulated settings reported there.

## Out of scope

OFAL / shrinkage build (follow-up lane), grouped/phylo/row-effect/per-variance routes, selective
inference theory, HSquared reuse (after this lands).
