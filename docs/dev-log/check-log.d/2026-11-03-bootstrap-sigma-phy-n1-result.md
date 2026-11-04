# N1 result: R gllvmTMB versus Julia bootstrap_Sigma_r, phy tier

Date: 2026-11-03. Run by a Claude session. Evidence note only: no row status, scoreboard, receipt or tolerance is changed.

Registration: `docs/dev-log/plans/2026-11-02-bootstrap-sigma-phy-n1-prereg-DRAFT.md` on branch `claude/n1-prereg-draft-20261102` (draft PR #1078). Thresholds and rules were applied exactly as drafted. Nothing was changed after seeing results.

Approval record: approved to run by Shinichi in chat on 2026-11-03; signature block left for Shinichi (Section 10 of the draft is not signed by this session and is not edited here).

## Verdict

**INCONCLUSIVE.** All four datasets are Inconclusive; none is Disagree and none is Agree. Overall rule: no dataset Disagree and not all four Agree, so Inconclusive.

The reason is the pre-registered noise-floor rule (1c): on every dataset at least one engine has a within-engine seed-2001-versus-seed-2002 largest standardised endpoint difference above 0.35 (values 0.25 to 0.77, Section 4). Dataset 1104 also fails rule 1b (Julia failed-replicate rate 3.8 percent against 0 percent for R, a gap of 3.8 percentage points, limit 2.5). Rules 1a and 1d did not fire on any dataset. Rule 1 comes first in the draft, so rules 2 and 3 are not reached.

What this means in plain terms: at n_boot = 999 the Monte Carlo noise in the largest endpoint comparison out of 40 is already larger than the 0.35 level the draft allowed, so the design as registered cannot separate the two engines. It does not show that the engines agree, and it does not show that they disagree.

## 1. What was run

- 4 datasets from `sim.jl` seeds 1101 to 1104 (30 tips, d = 5, 2 replicates per species, 300 rows). Dataset 1101 is byte-identical to the pilot copy. No replacement seeds were needed.
- R: gllvmTMB 0.7.1 at pin 9539352f66f2db2cc26b1c393e67212a359b60c9, library `/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library`, R 4.6.1. Formula, BFGS control and `bootstrap_Sigma` call as in Section 4 of the draft (`n_boot = 999`, `n_cores = 1`, `keep_draws = TRUE`).
- Julia: GLLVModels.jl at origin/main commit `85e47a2a9fa3f7882cb0ae420c13f810a79eaa23`, Julia 1.12.6, Manifest.toml sha256 `ba1f3285b0191304b17a003cb8baf07024b566c27c55caac6bb8913a3fea4c40` (the frozen pilot Manifest, copied into the run tree). `fit_precision_multivariate(rank = 1, mode = :explicitunique, residual_mode = :shared)`, then `bootstrap_Sigma_r(n_boot = 999, level = :phy, what = [:Sigma, :R, :communality], keep_draws = true)`, single-threaded.
- Bootstrap seeds 2001 (primary) and 2002 (repeat) on both sides: 16 runs of 999 refits.
- Oracle check on launch day (2026-11-03): the draft's `verify_noversion.py verify --destination .../oracle-build` printed `CORE070_ORACLE_VERIFY_PASS` (log stored with the raw outputs). The stock verify was not run; it is known to fail on the missing `gllvmTMB_version` key (PR-F1).
- Compute: Totoro through the existing ControlMaster, 8 single-thread processes (one per dataset and engine, `taskset -c 0-7`, `nice`, `OPENBLAS_NUM_THREADS=1`), load average 10 to 18 during the run. Estimate before launch: 2.0 to 2.3 core-hours, under the 3 h line. Actual: about 1.6 core-hours, about 27 minutes wall. No overrun.
- Raw outputs, scripts and logs: `~/hsq_work/claude-n1-20261103/` on Totoro (`out/`, `logs/`, `run_r.R`, `run_j.jl`, `export_r.R`). Julia stderr (failed-replicate warnings) is in `logs/J_*.err`.

Eligibility (Section 3 of the draft, applied before the comparison). All four datasets pass: both engines converged (R `convergence == 0`, Julia `converged == true`) and the largest elementwise difference between the two Sigma_phy point estimates is 2.1e-07 (1101), 8.2e-07 (1102), 6.0e-07 (1103), 2.9e-07 (1104), all below 1e-5. Boundary flags: 1101 (R unique variance for trait 4 is 1.2e-08, Julia 1.4e-10) and 1104 (trait 1: R 6.5e-09, Julia 1.6e-10; Julia reports `hessian_positive_definite = false`). 1102 and 1103 are interior (smallest unique variances 0.024 and 0.033).

## 2. Failed replicates, per side and seed

The R count equals the number of draws that are not finite; the Julia count equals `n_failed` and the number of non-finite draws.

| Dataset | R failed 2001 | R failed 2002 | Julia failed 2001 | Julia failed 2002 | Rates 2001 (R, Julia) | Rule 1a | Rule 1b |
|---|---|---|---|---|---|---|---|
| 1101 | 0 | 0 | 5 | 0 | 0.0%, 0.5% | pass | pass |
| 1102 | 0 | 0 | 7 | 6 | 0.0%, 0.7% | pass | pass |
| 1103 | 0 | 1 | 12 | 11 | 0.0%, 1.2% | pass | pass |
| 1104 | 0 | 4 | 38 | 52 | 0.0%, 3.8% | pass | fail (3.8 pp) |

Minimum finite draws in any cell on the primary seed: 994, 992, 987, 961 (rule 1d needs at least 950; pass). The packages' own reported bounds match the outside type-7 recomputation to at most 4.2e-15 on both sides, so quantile code does not enter.

## 3. Summary of the rules per dataset

| Dataset | Endpoints z <= 0.25 (of 40) | Endpoints z <= 0.50 (of 40) | SE ratio in [0.90, 1.10] (of 20) | SE ratio in [0.80, 1.25] (of 20) | Communality abs <= 0.05 | Largest z | SE ratio range | Rule 1 result |
|---|---|---|---|---|---|---|---|---|
| 1101 | 35 | 39 | 19 | 20 | no | 0.909 (comm5 upper) | 0.892 to 1.098 | 1c |
| 1102 | 36 | 40 | 19 | 20 | yes | 0.327 (Sigma35 lower) | 0.873 to 1.036 | 1c |
| 1103 | 37 | 40 | 20 | 20 | no | 0.370 (Sigma13 lower) | 0.937 to 1.091 | 1c |
| 1104 | 38 | 39 | 19 | 20 | no | 0.642 (Sigma33 upper) | 0.939 to 1.128 | 1b, 1c |

These counts are descriptive. The verdict rule stops at rule 1, so they carry no verdict. No Sigma_phy or communality cell had SE_pool below 1e-3, so the absolute-difference substitute for boundary cells was never used. R_phy was stored and is not analysed (descriptive-only in the draft).

## 4. Noise floor (seed 2001 against 2002, same engine)

| Dataset | Engine | Largest z | Cell | Endpoint | Failed 2001 | Failed 2002 |
|---|---|---|---|---|---|---|
| 1101 | R | 0.770 | comm5 | hi | 0 | 0 |
| 1101 | Julia | 0.457 | comm5 | hi | 5 | 0 |
| 1102 | R | 0.448 | Sigma23 | lo | 0 | 0 |
| 1102 | Julia | 0.245 | Sigma14 | hi | 7 | 6 |
| 1103 | R | 0.499 | comm2 | lo | 0 | 1 |
| 1103 | Julia | 0.588 | comm3 | hi | 12 | 11 |
| 1104 | R | 0.440 | Sigma23 | lo | 0 | 4 |
| 1104 | Julia | 0.725 | Sigma33 | hi | 38 | 52 |

Seven of the eight engine-dataset floors exceed the 0.35 limit. The draft's own rationale predicted a single-comparison SD of about 0.12 draw-SD, and the maximum over 40 correlated endpoints (about 4 SD, 0.50) as the plausible ceiling, so 0.35 as a floor for the maximum over cells and endpoints looks tight for n_boot = 999 on this design; that is a comment on the registered threshold, not a basis for changing it after the fact. The noisiest cells are the communality upper endpoints and the boundary-adjacent Sigma entries, where percentiles sit in thin tails.

## 5. Every cell, primary seed 2001

Columns: R and Julia 2.5 percent and 97.5 percent endpoints (type 7, recomputed from the stored draws), SE of finite draws, finite draw count n; dLo and dHi are Julia minus R; zLo and zHi are |d| / SE_pool; SE J/R is the ratio of SEs; Truth is the generating value (lam lam' + diag(s) for Sigma_phy, lam^2 / (lam^2 + s) for communality); R cov and J cov say whether the interval contains the truth (descriptive only). Cell names: SigmaIJ is entry (I, J) of Sigma_phy; commT is communality of trait T. An asterisk would mark SE_pool < 1e-3 (none occurred).

#### Dataset 1101 (primary seed 2001)

| Cell | R lo | R hi | R SE | R n | J lo | J hi | J SE | J n | dLo | dHi | zLo | zHi | SE J/R | Truth | R cov | J cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sigma11 | 0.3672 | 1.6255 | 0.3241 | 999 | 0.3477 | 1.5750 | 0.3136 | 994 | -0.0195 | -0.0505 | 0.061 | 0.158 | 0.968 | 1.310 | y | y |
| Sigma12 | -0.0591 | 0.5916 | 0.1729 | 999 | -0.1222 | 0.5934 | 0.1789 | 994 | -0.0632 | +0.0018 | 0.359 | 0.010 | 1.035 | 0.540 | y | y |
| Sigma22 | 0.2812 | 1.3086 | 0.2643 | 999 | 0.2617 | 1.3308 | 0.2648 | 994 | -0.0194 | +0.0222 | 0.073 | 0.084 | 1.002 | 0.760 | y | y |
| Sigma13 | -0.5895 | 0.0920 | 0.1690 | 999 | -0.5733 | 0.0700 | 0.1579 | 994 | +0.0162 | -0.0220 | 0.099 | 0.135 | 0.934 | -0.450 | y | y |
| Sigma23 | -0.5082 | 0.0735 | 0.1521 | 999 | -0.4984 | 0.0612 | 0.1438 | 994 | +0.0098 | -0.0123 | 0.066 | 0.083 | 0.946 | -0.300 | y | y |
| Sigma33 | 0.1923 | 0.9552 | 0.2008 | 999 | 0.1856 | 1.0456 | 0.2205 | 994 | -0.0068 | +0.0905 | 0.032 | 0.429 | 1.098 | 0.850 | y | y |
| Sigma14 | -0.0310 | 0.7911 | 0.2152 | 999 | -0.0271 | 0.7815 | 0.2022 | 994 | +0.0039 | -0.0096 | 0.019 | 0.046 | 0.939 | 0.630 | y | y |
| Sigma24 | -0.0136 | 0.6876 | 0.1882 | 999 | -0.0243 | 0.7480 | 0.1975 | 994 | -0.0106 | +0.0604 | 0.055 | 0.313 | 1.049 | 0.420 | y | y |
| Sigma34 | -0.6301 | 0.0236 | 0.1729 | 999 | -0.6300 | 0.0191 | 0.1755 | 994 | +0.0001 | -0.0045 | 0.000 | 0.026 | 1.015 | -0.350 | y | y |
| Sigma44 | 0.2089 | 1.0749 | 0.2211 | 999 | 0.2125 | 1.0718 | 0.2221 | 994 | +0.0036 | -0.0030 | 0.016 | 0.014 | 1.004 | 0.790 | y | y |
| Sigma15 | -0.2804 | 0.2583 | 0.1226 | 999 | -0.2762 | 0.2099 | 0.1170 | 994 | +0.0042 | -0.0484 | 0.035 | 0.404 | 0.954 | 0.360 | n | n |
| Sigma25 | -0.2468 | 0.1924 | 0.1102 | 999 | -0.2557 | 0.2004 | 0.1069 | 994 | -0.0089 | +0.0080 | 0.082 | 0.074 | 0.970 | 0.240 | n | n |
| Sigma35 | -0.2077 | 0.2362 | 0.1030 | 999 | -0.1983 | 0.2292 | 0.1044 | 994 | +0.0094 | -0.0069 | 0.091 | 0.067 | 1.013 | -0.200 | y | n |
| Sigma45 | -0.3103 | 0.2642 | 0.1355 | 999 | -0.3018 | 0.2522 | 0.1334 | 994 | +0.0085 | -0.0120 | 0.063 | 0.090 | 0.984 | 0.280 | n | n |
| Sigma55 | 0.1296 | 0.7797 | 0.1729 | 999 | 0.1171 | 0.7939 | 0.1792 | 994 | -0.0125 | +0.0143 | 0.071 | 0.081 | 1.036 | 0.660 | y | y |
| comm1 | 0.0065 | 1.0000 | 0.2543 | 999 | 0.0052 | 1.0000 | 0.2702 | 994 | -0.0013 | +0.0000 | 0.005 | 0.000 | 1.062 | 0.618 | y | y |
| comm2 | 0.0031 | 1.0000 | 0.2710 | 999 | 0.0037 | 1.0000 | 0.2652 | 994 | +0.0006 | +0.0000 | 0.002 | 0.000 | 0.979 | 0.474 | y | y |
| comm3 | 0.0085 | 1.0000 | 0.2915 | 999 | 0.0061 | 1.0000 | 0.2866 | 994 | -0.0024 | +0.0000 | 0.008 | 0.000 | 0.983 | 0.294 | y | y |
| comm4 | 0.0673 | 1.0000 | 0.2925 | 999 | 0.0490 | 1.0000 | 0.2874 | 994 | -0.0183 | +0.0000 | 0.063 | 0.000 | 0.983 | 0.620 | y | y |
| comm5 | 0.0001 | 0.8178 | 0.1948 | 999 | 0.0002 | 0.6500 | 0.1738 | 994 | +0.0001 | -0.1678 | 0.000 | 0.909 | 0.892 | 0.242 | y | y |

#### Dataset 1102 (primary seed 2001)

| Cell | R lo | R hi | R SE | R n | J lo | J hi | J SE | J n | dLo | dHi | zLo | zHi | SE J/R | Truth | R cov | J cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sigma11 | 0.5058 | 2.3711 | 0.4868 | 999 | 0.4763 | 2.3810 | 0.4821 | 992 | -0.0295 | +0.0099 | 0.061 | 0.021 | 0.990 | 1.310 | y | y |
| Sigma12 | -0.0853 | 1.0636 | 0.2928 | 999 | -0.0380 | 1.0246 | 0.2833 | 992 | +0.0473 | -0.0390 | 0.164 | 0.135 | 0.968 | 0.540 | y | y |
| Sigma22 | 0.1806 | 1.2375 | 0.2771 | 999 | 0.1766 | 1.2534 | 0.2779 | 992 | -0.0040 | +0.0159 | 0.014 | 0.057 | 1.003 | 0.760 | y | y |
| Sigma13 | -0.8995 | 0.2279 | 0.2873 | 999 | -0.8923 | 0.2091 | 0.2872 | 992 | +0.0072 | -0.0188 | 0.025 | 0.066 | 1.000 | -0.450 | y | y |
| Sigma23 | -0.4567 | 0.1912 | 0.1609 | 999 | -0.4767 | 0.1772 | 0.1634 | 992 | -0.0200 | -0.0141 | 0.124 | 0.087 | 1.015 | -0.300 | y | y |
| Sigma33 | 0.2200 | 1.4173 | 0.3119 | 999 | 0.2545 | 1.4629 | 0.3230 | 992 | +0.0345 | +0.0456 | 0.109 | 0.144 | 1.036 | 0.850 | y | y |
| Sigma14 | -0.0659 | 0.9997 | 0.2873 | 999 | -0.0331 | 0.9371 | 0.2605 | 992 | +0.0328 | -0.0626 | 0.120 | 0.228 | 0.907 | 0.630 | y | y |
| Sigma24 | -0.1173 | 0.5357 | 0.1671 | 999 | -0.0983 | 0.5455 | 0.1616 | 992 | +0.0191 | +0.0098 | 0.116 | 0.060 | 0.967 | 0.420 | y | y |
| Sigma34 | -0.5362 | 0.1769 | 0.1705 | 999 | -0.4842 | 0.1749 | 0.1609 | 992 | +0.0520 | -0.0020 | 0.314 | 0.012 | 0.943 | -0.350 | y | y |
| Sigma44 | 0.1704 | 1.1724 | 0.2696 | 999 | 0.1887 | 1.1524 | 0.2562 | 992 | +0.0183 | -0.0200 | 0.070 | 0.076 | 0.950 | 0.790 | y | y |
| Sigma15 | -0.3307 | 0.5322 | 0.2187 | 999 | -0.3652 | 0.5181 | 0.2216 | 992 | -0.0345 | -0.0141 | 0.157 | 0.064 | 1.013 | 0.360 | y | y |
| Sigma25 | -0.2130 | 0.3443 | 0.1280 | 999 | -0.2272 | 0.3086 | 0.1272 | 992 | -0.0142 | -0.0356 | 0.112 | 0.279 | 0.994 | 0.240 | y | y |
| Sigma35 | -0.3053 | 0.1907 | 0.1154 | 999 | -0.2684 | 0.2079 | 0.1105 | 992 | +0.0369 | +0.0172 | 0.327 | 0.152 | 0.957 | -0.200 | y | y |
| Sigma45 | -0.2067 | 0.3430 | 0.1287 | 999 | -0.2370 | 0.3070 | 0.1235 | 992 | -0.0303 | -0.0360 | 0.240 | 0.285 | 0.960 | 0.280 | y | y |
| Sigma55 | 0.0838 | 0.8681 | 0.2044 | 999 | 0.0984 | 0.8932 | 0.2064 | 992 | +0.0145 | +0.0251 | 0.071 | 0.122 | 1.010 | 0.660 | y | y |
| comm1 | 0.0657 | 1.0000 | 0.3164 | 999 | 0.0663 | 1.0000 | 0.3082 | 992 | +0.0006 | +0.0000 | 0.002 | 0.000 | 0.974 | 0.618 | y | y |
| comm2 | 0.0048 | 1.0000 | 0.3167 | 999 | 0.0052 | 1.0000 | 0.3090 | 992 | +0.0004 | +0.0000 | 0.001 | 0.000 | 0.976 | 0.474 | y | y |
| comm3 | 0.0004 | 1.0000 | 0.2681 | 999 | 0.0005 | 1.0000 | 0.2565 | 992 | +0.0002 | +0.0000 | 0.001 | 0.000 | 0.957 | 0.294 | y | y |
| comm4 | 0.0029 | 1.0000 | 0.3294 | 999 | 0.0035 | 1.0000 | 0.3083 | 992 | +0.0006 | +0.0000 | 0.002 | 0.000 | 0.936 | 0.620 | y | y |
| comm5 | 0.0001 | 1.0000 | 0.2658 | 999 | 0.0001 | 1.0000 | 0.2320 | 992 | -0.0000 | +0.0000 | 0.000 | 0.000 | 0.873 | 0.242 | y | y |

#### Dataset 1103 (primary seed 2001)

| Cell | R lo | R hi | R SE | R n | J lo | J hi | J SE | J n | dLo | dHi | zLo | zHi | SE J/R | Truth | R cov | J cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sigma11 | 0.5595 | 2.3910 | 0.4598 | 999 | 0.6243 | 2.3666 | 0.4422 | 987 | +0.0648 | -0.0244 | 0.144 | 0.054 | 0.962 | 1.310 | y | y |
| Sigma12 | 0.1818 | 1.0155 | 0.2074 | 999 | 0.1954 | 0.9974 | 0.2051 | 987 | +0.0135 | -0.0180 | 0.066 | 0.087 | 0.989 | 0.540 | y | y |
| Sigma22 | 0.0905 | 0.6149 | 0.1345 | 999 | 0.0995 | 0.6085 | 0.1338 | 987 | +0.0090 | -0.0064 | 0.067 | 0.048 | 0.994 | 0.760 | n | n |
| Sigma13 | -0.6598 | 0.1545 | 0.2151 | 999 | -0.7393 | 0.1448 | 0.2145 | 987 | -0.0795 | -0.0097 | 0.370 | 0.045 | 0.997 | -0.450 | y | y |
| Sigma23 | -0.3533 | 0.0780 | 0.1118 | 999 | -0.3719 | 0.0807 | 0.1136 | 987 | -0.0187 | +0.0026 | 0.166 | 0.023 | 1.016 | -0.300 | y | y |
| Sigma33 | 0.1947 | 1.0498 | 0.2217 | 999 | 0.1756 | 1.1109 | 0.2419 | 987 | -0.0191 | +0.0611 | 0.082 | 0.263 | 1.091 | 0.850 | y | y |
| Sigma14 | 0.3277 | 1.5235 | 0.3058 | 999 | 0.3525 | 1.5305 | 0.3027 | 987 | +0.0249 | +0.0070 | 0.082 | 0.023 | 0.990 | 0.630 | y | y |
| Sigma24 | 0.1590 | 0.8081 | 0.1708 | 999 | 0.1566 | 0.8319 | 0.1743 | 987 | -0.0024 | +0.0238 | 0.014 | 0.138 | 1.021 | 0.420 | y | y |
| Sigma34 | -0.5569 | 0.1289 | 0.1771 | 999 | -0.5955 | 0.1254 | 0.1806 | 987 | -0.0386 | -0.0034 | 0.216 | 0.019 | 1.020 | -0.350 | y | y |
| Sigma44 | 0.3297 | 1.4476 | 0.2896 | 999 | 0.3322 | 1.4982 | 0.3048 | 987 | +0.0025 | +0.0506 | 0.008 | 0.170 | 1.053 | 0.790 | y | y |
| Sigma15 | 0.0355 | 0.7165 | 0.1848 | 999 | 0.0277 | 0.7471 | 0.1880 | 987 | -0.0079 | +0.0306 | 0.042 | 0.164 | 1.017 | 0.360 | y | y |
| Sigma25 | 0.0200 | 0.3958 | 0.0995 | 999 | 0.0098 | 0.4039 | 0.0985 | 987 | -0.0102 | +0.0081 | 0.103 | 0.082 | 0.991 | 0.240 | y | y |
| Sigma35 | -0.2619 | 0.0478 | 0.0810 | 999 | -0.2703 | 0.0516 | 0.0814 | 987 | -0.0084 | +0.0037 | 0.104 | 0.046 | 1.005 | -0.200 | y | y |
| Sigma45 | 0.0297 | 0.6049 | 0.1519 | 999 | 0.0239 | 0.6411 | 0.1561 | 987 | -0.0058 | +0.0361 | 0.038 | 0.235 | 1.028 | 0.280 | y | y |
| Sigma55 | 0.0599 | 0.5903 | 0.1396 | 999 | 0.0693 | 0.6133 | 0.1441 | 987 | +0.0094 | +0.0229 | 0.066 | 0.162 | 1.033 | 0.660 | n | n |
| comm1 | 0.3648 | 1.0000 | 0.1772 | 999 | 0.3968 | 1.0000 | 0.1659 | 987 | +0.0320 | +0.0000 | 0.186 | 0.000 | 0.937 | 0.618 | y | y |
| comm2 | 0.5588 | 1.0000 | 0.1324 | 999 | 0.5550 | 1.0000 | 0.1336 | 987 | -0.0039 | +0.0000 | 0.029 | 0.000 | 1.009 | 0.474 | n | n |
| comm3 | 0.0004 | 0.6525 | 0.1865 | 999 | 0.0006 | 0.6022 | 0.1752 | 987 | +0.0003 | -0.0503 | 0.001 | 0.278 | 0.940 | 0.294 | y | y |
| comm4 | 0.4432 | 1.0000 | 0.1626 | 999 | 0.4827 | 1.0000 | 0.1561 | 987 | +0.0395 | +0.0000 | 0.248 | 0.000 | 0.960 | 0.620 | y | y |
| comm5 | 0.0196 | 1.0000 | 0.2825 | 999 | 0.0144 | 1.0000 | 0.2672 | 987 | -0.0052 | +0.0000 | 0.019 | 0.000 | 0.946 | 0.242 | y | y |

#### Dataset 1104 (primary seed 2001)

| Cell | R lo | R hi | R SE | R n | J lo | J hi | J SE | J n | dLo | dHi | zLo | zHi | SE J/R | Truth | R cov | J cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sigma11 | 0.4280 | 2.1259 | 0.4292 | 999 | 0.4513 | 2.0877 | 0.4296 | 961 | +0.0233 | -0.0381 | 0.054 | 0.089 | 1.001 | 1.310 | y | y |
| Sigma12 | 0.2104 | 1.3047 | 0.2797 | 999 | 0.2501 | 1.3005 | 0.2753 | 961 | +0.0396 | -0.0042 | 0.143 | 0.015 | 0.984 | 0.540 | y | y |
| Sigma22 | 0.1390 | 1.1391 | 0.2603 | 999 | 0.1656 | 1.1761 | 0.2564 | 961 | +0.0266 | +0.0369 | 0.103 | 0.143 | 0.985 | 0.760 | y | y |
| Sigma13 | -0.3166 | 0.1120 | 0.1095 | 999 | -0.3062 | 0.1181 | 0.1100 | 961 | +0.0104 | +0.0061 | 0.094 | 0.056 | 1.004 | -0.450 | n | n |
| Sigma23 | -0.1965 | 0.0706 | 0.0664 | 999 | -0.2026 | 0.0755 | 0.0686 | 961 | -0.0061 | +0.0049 | 0.090 | 0.072 | 1.033 | -0.300 | n | n |
| Sigma33 | 0.0001 | 0.1410 | 0.0363 | 999 | 0.0001 | 0.1180 | 0.0352 | 961 | -0.0001 | -0.0230 | 0.002 | 0.642 | 0.970 | 0.850 | n | n |
| Sigma14 | 0.3335 | 1.6195 | 0.3293 | 999 | 0.3373 | 1.6057 | 0.3324 | 961 | +0.0037 | -0.0137 | 0.011 | 0.041 | 1.009 | 0.630 | y | y |
| Sigma24 | 0.1641 | 1.0511 | 0.2228 | 999 | 0.1948 | 1.0555 | 0.2254 | 961 | +0.0306 | +0.0044 | 0.137 | 0.020 | 1.012 | 0.420 | y | y |
| Sigma34 | -0.2554 | 0.0996 | 0.0893 | 999 | -0.2612 | 0.0963 | 0.0900 | 961 | -0.0059 | -0.0034 | 0.065 | 0.037 | 1.008 | -0.350 | n | n |
| Sigma44 | 0.3012 | 1.5814 | 0.3242 | 999 | 0.3040 | 1.5831 | 0.3321 | 961 | +0.0028 | +0.0017 | 0.009 | 0.005 | 1.025 | 0.790 | y | y |
| Sigma15 | 0.0995 | 1.3328 | 0.3135 | 999 | 0.0779 | 1.3034 | 0.3077 | 961 | -0.0216 | -0.0294 | 0.070 | 0.095 | 0.982 | 0.360 | y | y |
| Sigma25 | 0.0451 | 0.8177 | 0.2011 | 999 | 0.0431 | 0.8523 | 0.2035 | 961 | -0.0020 | +0.0346 | 0.010 | 0.171 | 1.012 | 0.240 | y | y |
| Sigma35 | -0.1994 | 0.0685 | 0.0674 | 999 | -0.2029 | 0.0663 | 0.0686 | 961 | -0.0035 | -0.0022 | 0.051 | 0.033 | 1.017 | -0.200 | n | y |
| Sigma45 | 0.0781 | 1.0600 | 0.2540 | 999 | 0.0794 | 1.1078 | 0.2625 | 961 | +0.0014 | +0.0478 | 0.005 | 0.185 | 1.034 | 0.280 | y | y |
| Sigma55 | 0.3511 | 1.7981 | 0.3698 | 999 | 0.3241 | 1.8102 | 0.3921 | 961 | -0.0270 | +0.0121 | 0.071 | 0.032 | 1.060 | 0.660 | y | y |
| comm1 | 0.7260 | 1.0000 | 0.0725 | 999 | 0.7225 | 1.0000 | 0.0817 | 961 | -0.0035 | +0.0000 | 0.045 | 0.000 | 1.128 | 0.618 | n | n |
| comm2 | 0.2762 | 1.0000 | 0.2095 | 999 | 0.3489 | 1.0000 | 0.1968 | 961 | +0.0727 | +0.0000 | 0.358 | 0.000 | 0.939 | 0.474 | y | y |
| comm3 | 0.0027 | 1.0000 | 0.3928 | 999 | 0.0036 | 1.0000 | 0.3902 | 961 | +0.0009 | +0.0000 | 0.002 | 0.000 | 0.993 | 0.294 | y | y |
| comm4 | 0.5734 | 1.0000 | 0.1311 | 999 | 0.5542 | 1.0000 | 0.1307 | 961 | -0.0193 | +0.0000 | 0.147 | 0.000 | 0.997 | 0.620 | y | y |
| comm5 | 0.0262 | 0.9094 | 0.2320 | 999 | 0.0267 | 0.9104 | 0.2282 | 961 | +0.0005 | +0.0011 | 0.002 | 0.005 | 0.983 | 0.242 | y | y |

#### Noise floor detail (seed 2001 vs 2002, same engine)

| Dataset | Engine | max z | cell | endpoint | failed 2001 | failed 2002 |
|---|---|---|---|---|---|---|
| 1101 | R | 0.770 | comm5 | hi | 0 | 0 |
| 1101 | Julia | 0.457 | comm5 | hi | 5 | 0 |
| 1102 | R | 0.448 | Sigma23 | lo | 0 | 0 |
| 1102 | Julia | 0.245 | Sigma14 | hi | 7 | 6 |
| 1103 | R | 0.499 | comm2 | lo | 0 | 1 |
| 1103 | Julia | 0.588 | comm3 | hi | 12 | 11 |
| 1104 | R | 0.440 | Sigma23 | lo | 0 | 4 |
| 1104 | Julia | 0.725 | Sigma33 | hi | 38 | 52 |
Descriptive truth containment (not used in the verdict; four datasets cannot estimate coverage): of 20 cells, R intervals contain the truth in 17, 20, 17, 14 cells for datasets 1101 to 1104, Julia in 16, 20, 17, 15.

## 6. What this does not show

- It does not show that the two engines agree. The pre-registered precondition for a conclusive comparison (a within-engine noise floor of at most 0.35) was not met.
- It does not show that they disagree. The cross-engine largest z values (0.33 to 0.91) are of the same size as the within-engine floors (0.25 to 0.77). If anything, the cross-engine differences are not clearly larger than the Monte Carlo noise, but that is an informal reading of a result the rules declare uninformative; applying rules 2 and 3 literally anyway, 1101 (one endpoint above 0.50) and 1104 (one endpoint above 0.50) would not reach Agree, and that cell in 1101 (comm5 upper, 0.818 for R and 0.650 for Julia) is exactly where R's own seed-to-seed floor is largest (0.77). Not a verdict.
- The Julia side loses replicates that R does not (0.5 to 3.8 percent on the primary seed, up to 5.2 percent on the repeat; R 0 to 0.4 percent), and the loss grows with the boundary dataset 1104. Each side's percentiles rest on its own survivors, so a failure-selection effect on the Julia intervals is not excluded. Julia's stricter refit convergence rule (Section 4 of the draft) is the likely cause; this was not tested.
- Four datasets, one tree size, rank 1, d = 5, shared residual, no missing cells, no REML, one Julia and one R version. Not a coverage, recovery or general parity claim.
- It does not move any row status. Whether to raise n_boot, relax or replace the noise-floor rule, use more datasets, or run the paired follow-up in Section 6 of the draft (same bootstrap datasets refitted in both engines, which removes Monte Carlo noise) is for Shinichi to decide under a new registration.

Sections 8 and 9 of the draft (pilot and limits) held: no timing surprise, no warning flood, and the failure and boundary issues were those listed there.
