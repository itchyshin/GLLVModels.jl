# R ↔ Julia cross-check of guarded select_lv (2026-09-27)

Data: three simulated sets (pilot_sim.jl), species × sites, shared CSVs here. Julia: lane branch
claude/lane-auto-d-20260926 `select_lv(Y; family, Kmax = 4)`. R: gllvmTMB branch
claude/lane-auto-d-r-20260926 (978f4bba2) `select_lv(..., d_max = 4)` with
`value ~ 0 + trait + latent(0 + trait | unit, d, ...)`. Scripts: run_julia.jl, run_r.R (default
`unique = TRUE`), run_r_nopsi.R (`unique = FALSE`). Raw: julia_results.csv, r_results.csv,
r_results_nopsi.csv.

## Default R model (`unique = TRUE`) is a different model
gllvmTMB's `latent()` adds a per-trait residual Ψ by default (since 0.2.0); GLLVModels has no Ψ.
Poisson and Gaussian log-likelihoods and criteria therefore differ (R npar +8); chosen d agreed
(2, 1, 2) but that agreement is not guaranteed. Binomial matched exactly because gllvmTMB skips Ψ
for binary traits.

## Like-for-like (`unique = FALSE`)
| data | d | Julia logLik | R logLik | Julia npar | R npar | chosen (J / R) |
|---|---|---|---|---|---|---|
| Poisson n100 p8 K2 | 1 | −5522.06 | **−6663.87** | 16 | 16 | 2 / 2 |
| | 2 | −2135.7286 | −2135.7286 | 23 | 23 | |
| | 3 | −2126.7169 | −2126.7169 | 29 | 29 | |
| | 4 | −2125.8668 | −2125.8668 | 34 | 34 | |
| Binomial n120 p8 K1 | 1–4 | identical to 1e-5 | | 16/23/29/34 | same | 1 / 1 |
| Gaussian n80 p8 K2 | 1 | −1125.49 | −1122.26 | 9 | 17 | 2 / 2 |
| | 2 | −1051.35 | −1048.39 | 16 | 24 | |

Verdicts: Poisson AGREE for d ≥ 2; at d = 1 R converged to a poor optimum 1142 units below
Julia (same poor-optimum class as the NB finding, now on the R side). Binomial AGREE exactly.
Gaussian: Julia's default Normal route fits NO species intercepts (β = []), R fits 8; on these
mean-zero data the gap is ~3 units, but on data shifted by 5 Julia's logLik drops by 335
(−962.3 → −1297.1). Flagged as its own task (not this lane's files; changes healthy fits).
