# A0c input: new R exports and S3 methods at P1, with PROPOSED classes (2026-09-27)

Source: `diff` of `export(` and `S3method(` lines in gllvmTMB NAMESPACE, b4d5fee64 (0.7.0) vs 9539352f6 (P1). 25 exports added, 2 removed, 11 S3 methods added. Classes are proposals for Shinichi to sign in the A0c PR (D-295 row 3); an agent signs nothing.

| Export / method | R file at P1 | Proposed class | Arc | Note |
|---|---|---|---|---|
| temporal_dep, temporal_indep, temporal_latent | R/temporal-*.R (check) | twin | A6 | inside P1 (D-295) |
| extract_temporal, forecast_temporal, bootstrap_temporal, profile_temporal, compare_temporal | (check) | twin | A6 | R scope: Gaussian, rank 1, AR1/OU |
| ordinal_logit | (check) | twin | A2 small | |
| extract_latent_scores + S3 (default, gllvmTMB_multi, gllvmTMB_site_trait_sim, gllvmTMB_va) | (check) | twin | A2 small | the VA and sim methods may be excluded if Julia has no such class; recon decides |
| ordination_uncertainty + print method | (check) | twin | A2 medium | Wald form |
| select_lv + print method | (check) | twin, owned by the auto-d lane | #518 | Shinichi signs the row (D-295 row 6) |
| zi_binomial, zi_nbinom2, zi_poisson | (check) | semantic-divergence, then twin with R semantics | A2 medium | Julia two-part ZI stays a documented extra (D-295 row 5) |
| chibar2_pvalue, variance_lrt | R/chibar.R | semantic-divergence until checked | A2 small | Julia has both names in `src/boundary_inference.jl`; a name match never counts (row 5), so a twin test must show equal outputs |
| AIC, BIC, anova, update (gllvmTMB_multi) + print.anova | (check) | twin | A2 small | Julia has aic/bic; anova likely needs a surface |
| column_coef, animal_coef, phylo_coef, kernel_coef, spatial_coef | R/column-coef-foundation.R | outside-boundary (column grammar) | A7 at P2 | signed disposition row (D-295 boundary) |
| spatial_slope | (check) | outside-boundary (spatial) | A8 at P2 | |
| slope, kernel_slope | R/brms-sugar.R | outside-boundary (column grammar) | A7 at P2 | Checked 2026-09-27: roxygen defines them as response-column slopes, deviations in a response-column by predictor coefficient matrix B (Cov(vec(B')) = K kron Sigma), written as formula markers. R's capability-status groups them with phylo_slope, animal_slope and spatial_slope as the "Response-column slope family" (scope-limited, Gaussian long-format, predictor-only). The Haiku recon called them "formula sugar, not column-coef"; that is true of the syntax, not the model. Shinichi signs. |
| removed: .proportions_bootstrap_ci, .proportions_wald_ci | | retired | | any P0 row citing them becomes retired at P1, not stale |

Recon (A0r, Haiku) fills the "(check)" file cells and tests/examples per export.
