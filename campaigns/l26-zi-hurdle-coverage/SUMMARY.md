# L26 — Julia-only zero-inflated / hurdle Wald coverage sweep (extends #926)

**Package:** GLLVModels.jl `033d4e1da4c20f67e24d7308f442506065a8d768` (the `pkg_sha` column). Julia 1.11.9, Linux x86_64, 4 cores, one thread per worker. Distributions 0.25.131, Optim 1.13.3, ForwardDiff 0.10.39 (`versioninfo.txt`).
**Run:** cloud session, 2026-10-16. The sweep ran from 19:07 to 23:08 UTC, about 14.4 core-hours of fit + Wald time.
**`harness_head` column:** the session-branch HEAD the supervisor recorded. Those commits add data-only files under `campaigns/`; excluding `campaigns/`, the diff against 033d4e1 is empty, and the `src/` and `Project.toml` trees are identical.

## Design
- **Families:** `fit_zip_gllvm`, `fit_zinb_gllvm` and `fit_hurdle_poisson_gllvm` (all exported), with default settings.
- **Grid:** π ∈ {0.05, 0.2, 0.5} × n ∈ {100, 300, 1000}, with p = 10 and K = 2. Cells are numbered 1–27 (`l26core.jl::cell_spec`; n is the innermost index).
- **Truth (fixed across cells):**
  - Λ (10×2, lower-triangular): `[0.8 0; 0.5 0.6; -0.4 0.5; 0.6 -0.3; 0.3 0.6; -0.5 0.4; 0.7 -0.2; 0.2 -0.5; -0.3 0.5; 0.4 0.3]`
  - βc = range(0.5, 1.5); ZINB size r = 2.
  - betaz truth = logit(π) for ZIP/ZINB. For the hurdle model it is logit(1 − π), the occurrence logit, where π is the probability of a zero.
- **Generator:** Λz = 0, which matches the fitted models. Z ~ N(0, I). For each site and each species, one uniform decides the zero part, then a Poisson, NB2 or zero-truncated Poisson count is drawn.
- **Seeds:** RNG is `Xoshiro(seed)` with seed = 20261016 + 10000·cell + rep.
- **Wald intervals:** taken from `confint(fit, Y; method = :wald)`; `vcov(fit, Y)` comes from the same Hessian. Both were extracted in a single `_family_wald(…; covariance = true)` call, verified bit-identical to the public calls (Δ = 0). ΛΛ' intervals use the delta method from `vcov`, with a ForwardDiff Jacobian.
- **Coverage flag:** coverage outside [0.92, 0.98] **and** |cov − 0.95| > 1.96 × the rep-clustered MC SE.
- **Hard timeout (fit + Wald):** 1200 / 2400 / 3600 s at n = 100 / 300 / 1000. There were **no timeouts, errors or crashes**: all 337 cell×rep receipts have `status = ok`.

## Why the rep counts are below 200
One fit plus its finite-difference Wald Hessian takes about 11–1100 s on one core here, depending on family, π and n. Pilot timings put the full 27 × 200 grid at about 230 h of 4-core compute, against the 3.5 h estimate. I ran a pre-declared reduced design, in priority order: ZIP first (extends #926), then ZINB, then hurdle. Reps per cell are the first R reps of the 200-rep seed sequence, so the sweep can be extended (resumable `supervisor.py`; edit `plan.txt`).

| cell | family | π | n | reps run / planned |
|---|---|---|---|---|
| 1 | ZIP | 0.05 | 100 | 20 / 200 |
| 2 | ZIP | 0.05 | 300 | 8 / 200 |
| 3 | ZIP | 0.05 | 1000 | 3 / 200 |
| 4 | ZIP | 0.2 | 100 | 30 / 200 |
| 5 | ZIP | 0.2 | 300 | 15 / 200 |
| 6 | ZIP | 0.2 | 1000 | 5 / 200 |
| 7 | ZIP | 0.5 | 100 | 30 / 200 |
| 8 | ZIP | 0.5 | 300 | 15 / 200 |
| 9 | ZIP | 0.5 | 1000 | 5 / 200 |
| 10 | ZINB | 0.05 | 100 | 6 / 200 |
| 11 | ZINB | 0.05 | 300 | 2 / 200 |
| 12 | ZINB | 0.05 | 1000 | 1 / 200 |
| 13 | ZINB | 0.2 | 100 | 10 / 200 |
| 14 | ZINB | 0.2 | 300 | 8 / 200 |
| 15 | ZINB | 0.2 | 1000 | 4 / 200 |
| 16 | ZINB | 0.5 | 100 | 10 / 200 |
| 17 | ZINB | 0.5 | 300 | 8 / 200 |
| 18 | ZINB | 0.5 | 1000 | 4 / 200 |
| 19 | HurdlePoisson | 0.05 | 100 | 30 / 200 |
| 20 | HurdlePoisson | 0.05 | 300 | 15 / 200 |
| 21 | HurdlePoisson | 0.05 | 1000 | 6 / 200 |
| 22 | HurdlePoisson | 0.2 | 100 | 30 / 200 |
| 23 | HurdlePoisson | 0.2 | 300 | 15 / 200 |
| 24 | HurdlePoisson | 0.2 | 1000 | 6 / 200 |
| 25 | HurdlePoisson | 0.5 | 100 | 30 / 200 |
| 26 | HurdlePoisson | 0.5 | 300 | 15 / 200 |
| 27 | HurdlePoisson | 0.5 | 1000 | 6 / 200 |

## Results

### ZIP

| π | n | reps | betaz cov (MC SE) / bias | betac cov (MC SE) / bias | ΛΛ' diag cov (MC SE) / bias | ΛΛ' off-diag cov (MC SE) / bias | betaz median bias | converged | PD Hessian | timeouts | median s/rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 20 | 0.909 (0.024) / -0.884 | 0.940 (0.022) / -0.013 | 0.955 (0.017) / +0.018 | 0.948 (0.018) / +0.007 | -0.010 | 6/20 | 18/20 | 0 | 192 |
| 0.05 | 300 | 8 | **0.887** (0.030) / -0.031 | 0.925 (0.025) / -0.004 | 0.925 (0.016) / -0.000 | 0.928 (0.019) / -0.006 | +0.118 | 7/8 | 8/8 | 0 | 367 |
| 0.05 | 1000 | 3 | 0.767 (0.145) / +0.237 | 0.833 (0.120) / +0.002 | 0.867 (0.088) / -0.007 | 0.948 (0.041) / -0.009 | +0.214 | 3/3 | 3/3 | 0 | 614 |
| 0.2 | 100 | 30 | **0.910** (0.016) / +0.088 | 0.920 (0.017) / -0.017 | **0.900** (0.025) / -0.000 | 0.924 (0.017) / -0.008 | +0.117 | 30/30 | 30/30 | 0 | 55 |
| 0.2 | 300 | 15 | **0.800** (0.034) / +0.174 | 0.947 (0.013) / -0.010 | 0.940 (0.016) / -0.005 | 0.956 (0.016) / -0.001 | +0.184 | 15/15 | 15/15 | 0 | 166 |
| 0.2 | 1000 | 5 | **0.640** (0.068) / +0.160 | 0.980 (0.020) / +0.003 | 0.980 (0.020) / -0.002 | 0.973 (0.013) / -0.001 | +0.146 | 5/5 | 5/5 | 0 | 747 |
| 0.5 | 100 | 30 | **0.883** (0.019) / +0.175 | 0.943 (0.016) / -0.030 | 0.920 (0.018) / -0.005 | **0.914** (0.015) / -0.002 | +0.183 | 30/30 | 30/30 | 0 | 88 |
| 0.5 | 300 | 15 | **0.687** (0.048) / +0.192 | 0.940 (0.019) / -0.019 | **0.820** (0.039) / -0.045 | **0.856** (0.031) / -0.011 | +0.201 | 15/15 | 15/15 | 0 | 226 |
| 0.5 | 1000 | 5 | **0.300** (0.055) / +0.200 | 0.940 (0.040) / -0.021 | 0.900 (0.032) / -0.022 | 0.929 (0.016) / -0.006 | +0.186 | 5/5 | 5/5 | 0 | 956 |

### ZINB

| π | n | reps | betaz cov (MC SE) / bias | betac cov (MC SE) / bias | ΛΛ' diag cov (MC SE) / bias | ΛΛ' off-diag cov (MC SE) / bias | r (NB size) cov (MC SE) / bias | betaz median bias | converged | PD Hessian | timeouts | median s/rep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 6 | **0.875** (0.020) / -2.626 | 0.933 (0.021) / -0.066 | 0.917 (0.031) / +0.018 | **0.904** (0.020) / -0.023 | 0.833 (0.167) / +0.439 | +0.248 | 2/6 | 3/6 | 0 | 261 |
| 0.05 | 300 | 2 | 0.950 (0.050) / -0.026 | 0.950 (0.050) / -0.070 | 0.950 (0.050) / +0.023 | 0.956 (0.044) / +0.012 | **1.000** (0.000) / +0.122 | +0.219 | 2/2 | 2/2 | 0 | 447 |
| 0.05 | 1000 | 1 | 0.900 (nan) / -0.077 | 0.800 (nan) / -0.046 | 0.900 (nan) / +0.021 | 0.822 (nan) / +0.020 | 1.000 (nan) / -0.033 | -0.192 | 1/1 | 1/1 | 0 | 1092 |
| 0.2 | 100 | 10 | **0.830** (0.040) / +0.164 | 0.960 (0.022) / -0.049 | 0.950 (0.022) / +0.005 | 0.916 (0.023) / -0.013 | 0.800 (0.133) / +0.492 | +0.224 | 10/10 | 10/10 | 0 | 77 |
| 0.2 | 300 | 8 | **0.875** (0.025) / +0.091 | 0.963 (0.018) / -0.031 | **0.988** (0.012) / -0.012 | 0.967 (0.013) / -0.007 | 0.875 (0.125) / +0.075 | +0.157 | 8/8 | 8/8 | 0 | 292 |
| 0.2 | 1000 | 4 | **0.775** (0.048) / +0.081 | **0.875** (0.025) / -0.058 | 0.875 (0.063) / -0.020 | 0.911 (0.035) / -0.005 | **1.000** (0.000) / -0.061 | +0.074 | 4/4 | 4/4 | 0 | 841 |
| 0.5 | 100 | 10 | 0.940 (0.022) / -0.049 | **0.850** (0.031) / -0.249 | **0.850** (0.037) / +0.272 | **0.809** (0.038) / -0.015 | 0.700 (0.153) / +0.824 | +0.064 | 10/10 | 10/10 | 0 | 120 |
| 0.5 | 300 | 8 | 0.900 (0.033) / +0.098 | 0.950 (0.019) / -0.099 | **0.850** (0.033) / -0.017 | 0.931 (0.017) / -0.016 | 0.875 (0.125) / +0.022 | +0.117 | 8/8 | 8/8 | 0 | 370 |
| 0.5 | 1000 | 4 | 0.850 (0.065) / +0.099 | 0.950 (0.029) / -0.061 | 0.825 (0.111) / -0.053 | 0.867 (0.074) / +0.000 | **1.000** (0.000) / -0.097 | +0.112 | 4/4 | 4/4 | 0 | 1058 |

### HurdlePoisson

| π | n | reps | betaz cov (MC SE) / bias | betac cov (MC SE) / bias | ΛΛ' diag cov (MC SE) / bias | ΛΛ' off-diag cov (MC SE) / bias | betaz median bias | converged | PD Hessian | timeouts | median s/rep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 30 | **0.983** (0.007) / +0.198 | 0.950 (0.012) / -0.006 | 0.947 (0.012) / +0.001 | 0.948 (0.011) / -0.001 | +0.000 | 29/30 | 29/30 | 0 | 25 |
| 0.05 | 300 | 15 | 0.967 (0.013) / +0.030 | 0.967 (0.016) / +0.010 | 0.960 (0.016) / +0.009 | 0.948 (0.018) / -0.001 | +0.000 | 15/15 | 15/15 | 0 | 67 |
| 0.05 | 1000 | 6 | 0.967 (0.021) / +0.021 | 0.900 (0.037) / +0.007 | 0.950 (0.034) / -0.010 | 0.937 (0.031) / -0.005 | -0.000 | 6/6 | 6/6 | 0 | 196 |
| 0.2 | 100 | 30 | 0.957 (0.015) / +0.035 | 0.953 (0.013) / -0.013 | 0.923 (0.020) / -0.010 | 0.943 (0.011) / -0.003 | +0.000 | 30/30 | 30/30 | 0 | 14 |
| 0.2 | 300 | 15 | 0.980 (0.011) / +0.011 | 0.947 (0.017) / +0.004 | 0.927 (0.023) / -0.002 | 0.961 (0.010) / -0.008 | +0.021 | 15/15 | 15/15 | 0 | 41 |
| 0.2 | 1000 | 6 | **0.867** (0.033) / +0.008 | 0.917 (0.048) / +0.005 | 0.900 (0.068) / -0.010 | 0.919 (0.040) / +0.001 | +0.016 | 6/6 | 6/6 | 0 | 133 |
| 0.5 | 100 | 30 | 0.950 (0.011) / +0.002 | 0.953 (0.011) / -0.010 | 0.927 (0.012) / +0.016 | 0.940 (0.010) / -0.013 | -0.000 | 30/30 | 30/30 | 0 | 11 |
| 0.5 | 300 | 15 | 0.933 (0.019) / -0.003 | 0.967 (0.016) / -0.000 | 0.940 (0.024) / -0.011 | 0.944 (0.014) / -0.005 | +0.000 | 15/15 | 15/15 | 0 | 32 |
| 0.5 | 1000 | 6 | 0.933 (0.042) / -0.008 | 0.950 (0.022) / +0.002 | 0.950 (0.034) / -0.005 | 0.956 (0.013) / -0.004 | +0.002 | 6/6 | 6/6 | 0 | 105 |

**Bold** marks a flagged coverage. Rows with 1–4 reps are thin. For r, the "1.000 (0.000)" cells with 1–4 intervals carry no information.

## Findings and where they are filed
1. **ZIP betaz bias does not shrink with n; Wald coverage falls to 0.64 (π = 0.2) and 0.30 (π = 0.5) at n = 1000.**
   - Cause: the ZIP Laplace log-det uses the expected count information (`_tp_observed_Wc(::Any, …) = Wc`). At the truth that implies a betaz shift of +0.19 to +0.21. An observed-curvature Laplace and AGHQ both give ≈ 0 (`mechanism_zi.jl`).
   - At π = 0.5 the same cause attenuates the loadings by about 5%, so the ΛΛ' diagonal covers only 0.82 at n = 300.
   - Filed as comments on #926. Cross-referenced on #739, where draft PR #913 implements the observed curvature. On #913's head (03b0813) the mechanism check gives δ* ≈ 0, i.e. the shift is removed.
2. **ZINB has the same mechanism; not covered by #913.**
   - betaz coverage is 0.78–0.88 at π = 0.2.
   - An n = 5000 fit settles at betaz +0.125 and betac −0.034, with the ΛΛ' diagonal −5%.
   - Filed as **#932**, plus two comments.

## Outside the band but not filed (explained)
- **Hurdle betaz at π = 0.05, n = 100: 0.983.** With Λz = 0, the hurdle betaz is exactly the intercept-only binomial logit; package estimate and SE equal logit(k/n) and 1/√(n p̂(1−p̂)) to about 1e-6. That textbook Wald-logit interval's exact coverage there is 0.966 (0.972 given a finite MLE). This is known small-count behaviour, not a package defect (`hurdle_betaz_note.txt`).
- **Hurdle betaz at π = 0.2, n = 1000: 0.867 (6 reps, 60 intervals).** The estimator is the same closed-form binomial logit, whose exact coverage there is 0.947. P(≥ 8 misses out of 60) = 0.013, which is expected somewhere among about 100 cell × block tests, so I treat it as MC noise.
- **ZIP / ZINB at π = 0.05, n = 100:** weak identification of the structural-zero component.
  - ZIP: 14 of 20 fits stop at the 500-iteration cap; ZINB: 4 of 6.
  - Non-PD Hessians: 2 of 20 (ZIP), 3 of 6 (ZINB).
  - Boundary betaz estimates have huge SEs and cover trivially.
  - The fitters flag this honestly (`converged = false`). It was reported in the #926 and #932 comments, not as a separate issue.
- **Non-PD rate above 5% at n = 1000:** not observed (0 of 3–6 per cell), but the n = 1000 cells have too few reps to test the criterion well.

## Files
- `receipts.csv`: one row per cell×rep (status, logLik, converged, iterations, pd_hessian, NaN SEs, max|grad| at the estimate, r̂, zero share, fit / Wald / total seconds, `pkg_sha`, Julia version, `harness_head`).
- `params.csv`: one row per interval (betaz, betac, ΛΛ' unique entries, r): truth, estimate, SE, bounds, covered (−1 = no finite interval).
- `summary_by_cell*.csv` and `table_*.md` (in `diagnostics/`): per-cell summaries.
- `l26core.jl`, `worker.jl`, `supervisor.py`, `analyse.py`, `normalise_params.py`, `family_table.py`: the harness.
- `repro_standalone.jl`, `mechanism_zi.jl`, `mechanism_zinb_joint.jl`, `largen_fit.jl`, `diag_score.jl`: repros and diagnostics, with outputs in `diagnostics/` and repro datasets in `data/`.
