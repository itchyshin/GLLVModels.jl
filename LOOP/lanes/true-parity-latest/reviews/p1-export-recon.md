# gllvmTMB Exports and S3 Methods Reconnaissance (commit 9539352f6)

## Summary

This document maps gllvmTMB exports and S3 methods added since v0.7.0, with R file locations, roxygen descriptions, test coverage, and capability status. The entries are organized by export/method name.

**Key distinction for slope/kernel_slope**: Both are formula-sugar desugar functions (brms-style formula helpers), NOT part of the Design 131 column-coefficient framework. The column-coef exports (column_coef, animal_coef, phylo_coef, kernel_coef, spatial_coef) are a separate system for response-column coefficients.

## Full Inventory

| Export/Method | R File:Line | Description | Tests Mentioned | Has Examples | Notes | Capability Status |
|---|---|---|---|---|---|---|
| animal_coef | R/column-coef-foundation.R:127 | Animal response-column coefficients | test-column-coef-animal-equivalence.R, test-column-coef-animal-parser.R, test-column-coef-animal-recovery.R (9 total) | Yes | Design 131: animal pedigree/relmat response-column coefficients with estimated/fixed rho mixture | scope-limited (column-coef framework) |
| bootstrap_temporal | R/temporal-bootstrap.R:21 | Parametric bootstrap for a temporal persistence parameter | test-temporal-program-bootstrap.R | No | Unreplicated Gaussian temporal_indep only; separate bounded contract | R-only bounded temporal helper |
| chibar2_pvalue | R/chibar.R:93 | Chi-bar-square p-value for a boundary likelihood-ratio test | test-select-lv-anova.R | Yes | Self-Liang chi-bar-square mixture for LRT boundary testing | scope-limited (part of select_lv / MS-01, MS-02) |
| column_coef | R/column-coef-foundation.R:32 | IID response-column coefficients | test-column-coef-animal-equivalence.R, test-column-coef-animal-parser.R, test-column-coef-engine-iid.R (20+ total) | Yes | Design 131: IID response-column slopes/intercepts with full or diagonal covariance | scope-limited (FG-20) |
| compare_temporal | R/temporal-selection.R:5 | Compare supplied temporal model candidates | test-temporal-program-selection.R | No | Unreplicated Gaussian temporal_indep only; model-selection helper | R-only temporal source-pair evidence |
| extract_latent_scores | R/extract-latent-scores.R:42 | Extract latent random-effect scores | test-extract-latent-scores.R | Yes | Per-unit latent variable estimates at unit or unit_obs level | implemented (point extraction) |
| extract_temporal | R/temporal.R:495 | Extract temporal covariance provider details | test-temporal-ar1-methods.R, test-temporal-program-animal-replicated.R (7 total) | No | Temporal state/parameter extraction from fitted models | R-only temporal workflow |
| forecast_temporal | R/temporal-forecast.R:35 | Forecast a native temporal covariance model | test-temporal-program-forecast.R | Yes | Unreplicated Gaussian temporal_indep only; separate bounded contract | R-only bounded temporal helper |
| kernel_coef | R/column-coef-foundation.R:171 | Dense-kernel response-column coefficients | test-column-coef-engine-iid.R, test-column-coef-foundation.R, test-column-coef-kernel-edge.R (8 total) | Yes | Design 131: labelled positive-definite kernel-sourced response-column coefficients | scope-limited (FG-20) |
| kernel_slope | R/brms-sugar.R:924 | Fixed-kernel response-column slopes | test-column-coef-kernel-equivalence.R, test-fixed-column-slope-family.R (4 total) | Yes | **Formula sugar (brms-style desugar)**; not Design 131. Fixed-effect kernel-coupled slope term with full or diagonal predictor covariance | scope-limited (FG-19, etc.) |
| ordinal_logit | R/families.R:886 | Ordinal-logit threshold family for the multivariate engine | test-enum-runtime-ids.R, test-ordinal-logit.R | Yes | Cumulative-logit ordinal response; distinct from R's missing-predictor cumulative_logit | scope-limited (FAM-24) |
| ordination_uncertainty | R/ordination-uncertainty.R:171 | Per-unit covariance of ordination (latent) scores | test-extract-latent-scores.R, test-ordination-uncertainty.R, test-temporal-ar1-methods.R | Yes | Unit-level latent-score covariance matrix (Wald or other estimator) | scope-limited |
| phylo_coef | R/column-coef-foundation.R:81 | Phylogenetic response-column coefficients | test-column-coef-engine-iid.R, test-column-coef-foundation.R, test-column-coef-phylo-estimated-rho.R (5 total) | Yes | Design 131: tree/vcv response-column coefficients with fixed/estimated rho phylogenetic mixture | scope-limited (column-coef framework) |
| profile_temporal | R/temporal-profile.R:14 | Profile a temporal persistence or decay parameter | test-temporal-program-profile.R | No | Unreplicated Gaussian temporal_indep only; separate bounded contract | R-only bounded temporal helper |
| select_lv | R/select-lv.R:156 | Select a latent-variable rank by information criterion | test-select-lv-anova.R, test-temporal-ar1-methods.R, test-temporal-sixth-source-engine.R | Yes | Rank-selection by AIC/BIC/AICc from a fitted sequence; paired with anova(test="chibar") | scope-limited (MS-01, MS-02) |
| slope | R/brms-sugar.R:880 | Ordinary response-column slopes | test-1188-trait-col-augmented-lhs-guard.R, test-animal-dep-slope-gaussian.R, test-animal-indep-slope-gaussian.R (50+ total) | Yes | **Formula sugar (brms-style desugar)**; not Design 131. Ordinary slope term with full or diagonal predictor covariance; Gaussian long-format only | scope-limited (FG-19, etc.) |
| spatial_coef | R/column-coef-foundation.R:222 | Spatial response-column coefficients | test-column-coef-foundation.R, test-column-coef-public-api.R, test-column-coef-spatial-edge.R (8 total) | Yes | Design 131: SPDE/Matérn-field response-column coefficients with labelled mesh | scope-limited (column-coef framework) |
| spatial_slope | R/brms-sugar.R:996 | Spatial response-column slopes | test-column-coef-spatial-equivalence.R, test-spatial-column-slope.R, test-spatial-indep-slope-gaussian.R | Yes | **Formula sugar (brms-style desugar)**; not Design 131. Spatial Matérn field slope term with normalized SPDE covariance | scope-limited (FG-19, etc.) |
| temporal_dep | R/temporal.R:65 | Temporal unstructured covariance provider | test-temporal-ar1-parser.R, test-temporal-program-animal-replicated.R, test-temporal-program-bootstrap.R (10 total) | No | AR1/OU process with unstructured per-trait covariance; native temporal source | R-only temporal covariance |
| temporal_indep | R/temporal.R:56 | Temporal independent covariance provider | test-temporal-ar1-parser.R, test-temporal-phylo-optimizer-qualification.R, test-temporal-program-animal-replicated.R (14 total) | No | AR1/OU process per-trait independent; native temporal source; most documented | R-only temporal source |
| temporal_latent | R/temporal.R:93 | Temporal covariance providers | test-temporal-ar1-methods.R, test-temporal-ar1-oracles.R, test-temporal-ar1-parser.R (6 total) | No | AR1/OU rank-one latent loading family; native temporal source | R-only temporal workflow |
| variance_lrt | R/chibar.R:144 | Boundary likelihood-ratio test for one or more variance components | test-select-lv-anova.R | Yes | Self-Liang LRT wrapper for variance boundaries; used by anova() | scope-limited (part of select_lv / MS-01, MS-02) |
| zi_binomial | R/families.R:523 | Zero-inflated binomial family | test-enum-runtime-ids.R, test-zi-families.R, test-zi-recovery.R | No | TRUE zero-inflation mixture (not hurdle); requires multi-trial data | scope-limited (zib / FAM-21, FAM-22, FAM-23) |
| zi_nbinom2 | R/families.R:517 | Zero-inflated NB2 family | test-enum-runtime-ids.R, test-predictive-diagnostics.R, test-zi-families.R, test-zi-recovery.R | No | TRUE zero-inflation mixture (not hurdle); per-trait NB2 dispersion reused | scope-limited (zinb / FAM-21, FAM-22, FAM-23) |
| zi_poisson | R/families.R:511 | Zero-inflated Poisson family | test-censored-poisson.R, test-enum-runtime-ids.R, test-predictive-diagnostics.R, test-zi-families.R | Yes | TRUE zero-inflation mixture (not hurdle); per-trait structural-zero probability (logit, intercept-only) | scope-limited (zip / FAM-21, FAM-22, FAM-23) |
| AIC.gllvmTMB_multi | R/aghq-report.R:213 | AIC extractor for model comparison (S3 method) | (no dedicated tests) | No | Wraps stats::AIC() with fit-compatibility checks; penalty k=2 default | (S3 method) |
| BIC.gllvmTMB_multi | R/aghq-report.R:222 | BIC extractor for model comparison (S3 method) | (no dedicated tests) | No | Wraps stats::BIC() with fit-compatibility checks | (S3 method) |
| anova.gllvmTMB_multi | R/aghq-report.R:456 | Likelihood-ratio comparison of nested gllvmTMB fits (S3 method) | (no dedicated tests) | No | Sequential rank/fixed-effect LRT via chi-bar-square (approx) or chi-square; rank-step caveat noted | (S3 method) |
| extract_latent_scores.default | R/extract-latent-scores.R:73 | Fallback extractor for latent scores (S3 method) | (no dedicated tests) | No | Default method; warns that class is unrecognized | (S3 method) |
| extract_latent_scores.gllvmTMB_multi | R/extract-latent-scores.R:47 | Latent score extraction for multi-fit object (S3 method) | (no dedicated tests) | No | Delegates to component fits' extract_latent_scores() methods | (S3 method) |
| extract_latent_scores.gllvmTMB_site_trait_sim | R/extract-latent-scores.R:62 | Latent score extraction for simulation (S3 method) | (no dedicated tests) | No | Extracts latent scores from simulated data object | (S3 method) |
| extract_latent_scores.gllvmTMB_va | R/extract-latent-scores.R:57 | Latent score extraction for VA fits (S3 method) | (no dedicated tests) | No | Extracts latent scores from variational-approximation fit | (S3 method) |
| print.anova.gllvmTMB_multi | R/aghq-report.R:582 | Printer for anova.gllvmTMB_multi result table (S3 method) | (no dedicated tests) | No | Formats LRT comparison output with per-row notes; default digits=4 | (S3 method) |
| print.gllvmTMB_ordination_uncertainty | R/ordination-uncertainty.R:303 | Printer for ordination_uncertainty result (S3 method) | (no dedicated tests) | No | Formats per-unit latent-score covariance estimates for display | (S3 method) |
| print.gllvmTMB_select_lv | R/select-lv.R:349 | Printer for select_lv result (S3 method) | (no dedicated tests) | No | Formats rank-selection output with IC values and recommendation | (S3 method) |
| update.gllvmTMB_multi | R/methods-gllvmTMB.R:14 | Refit a temporal latent model (S3 method) | (no dedicated tests) | No | Wrapper for refitting; temporal-latent specific use case | (S3 method) |

## Key Findings

### Slope and Kernel_Slope Classification

Both **slope** (R/brms-sugar.R:880) and **kernel_slope** (R/brms-sugar.R:924) are **formula-sugar desugar functions**, not part of the Design 131 column-coefficient framework. They return `invisible(NULL)` and carry roxygen `@return "A formula marker; never evaluated"`. This is fundamentally different from the column-coef exports (column_coef, animal_coef, phylo_coef, kernel_coef, spatial_coef), which are the Design 131 mechanism.

The capability ledger treats all five slope variants (slope, phylo_slope, animal_slope, kernel_slope, spatial_slope) as a single "Response-column slope family" row: scope-limited, Gaussian long-format only, predictor-only.

### Temporal Exports

The temporal family (temporal_indep, temporal_dep, temporal_latent, extract_temporal, forecast_temporal, bootstrap_temporal, profile_temporal, compare_temporal) are **R-only capabilities** with no Julia twin. The capability ledger maps them to unmapped-by-design register rows TEMP-06-01 through TEMP-06-05. Most temporal functions carry bounded contracts: forecast/bootstrap/profile/compare are unreplicated Gaussian temporal_indep only.

### Zero-Inflated Families

The zi_* functions (zi_poisson, zi_nbinom2, zi_binomial) are TRUE zero-inflation mixtures (distinct from hurdle/delta models) with per-trait, intercept-only structural-zero probability. They share the single capability row FAM-21/22/23 (scope-limited). Divergence note: gllvmTMB's zi_nbinom2 reuses ordinary per-trait NB2 dispersion; GLLVM.jl uses one scalar across species.

### Design 131 (Column-Coefficient Framework)

Five functions (column_coef, animal_coef, phylo_coef, kernel_coef, spatial_coef) implement response-column coefficients via Design 131. They are all formula markers like the slopes, but they build a distinct response-column coefficient system where both intercepts and slopes on a common predictor basis share covariance structure (full `|` or diagonal `||`). This is orthogonal to the slope system.

---

**Generated from**: gllvmTMB commit 9539352f6, docs/design/capability-status.md, and R source files  
**Date**: 2026-09-27
