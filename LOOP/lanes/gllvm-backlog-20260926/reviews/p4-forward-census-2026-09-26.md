# P4: R exports with no Julia twin (census, 2026-09-26)

Source: `tools/parity_ledger.py` run against frozen gllvmTMB 0.7.0 (b4d5fee6) and GLLVModels.jl main at 2847b5dbf plus one docs commit (exports unchanged since). Raw output: `parity-ledger-forward-2026-09-26.txt`. FORWARD = 62, REVERSE unclassified = 0.

## How the 62 split

| Group | Count | Examples | Status in the ledger |
|---|---|---|---|
| Formula keywords for structures Julia does not have yet | 28 | phylo_*, spatial_*, animal_*, latent, traits, meta, meta_V, indep, scalar, spde, isdm_source(s) | mostly BLOCKED_NEEDS_JULIA_SURFACE; indep and scalar PARTIAL_PENDING_DECISION |
| Families R has and Julia intentionally does not | 13 | delta_* variants, *_mix, gengamma, truncated_nbinom1 | intentionally_excluded |
| Plots, screening and matrix helpers | 13 | plot_*, screen_gllvmTMB, screen_table, ridge_path, VP, block_V | intentionally_excluded |
| Standalone functions, small and well specified | 8 | extract_residual_split, pedigree_to_A, pedigree_to_Ainv_sparse, suggest_lambda_constraint(s), confirmatory_lambda, simulate_site_trait, gllvm_julia_setup | required_core or BLOCKED_NEEDS_JULIA_SURFACE |

Counts checked by hand against the raw list: 28 + 13 + 13 + 8 = 62.

## What could be built overnight, and why it did not merge

The eight standalone functions are the only items that fit a night. Each adds public API to GLLVModels.jl, which the overnight rule (D-290) sends to Shinichi. The formula keywords need the Julia-side structures (phylogenetic latent, spatial, animal model), which are multi-week builds.

Suggested order if Shinichi wants them: pedigree_to_A and pedigree_to_Ainv_sparse first (pure functions with an exact answer, testable against the frozen R build), then extract_residual_split (an extractor over existing fits), then the lambda-constraint helpers.
