# Review of PR #547, phylo_latent twin (head 3dd69810e), 2026-09-27: NON-BLOCKING

Fresh P1 install reproduced every receipt to 15 significant figures (tree nll 8.68192008542953, dense 8.68192011899238; log-dets; gradients). Hashes and generator verified; twin file 81/81 (test env); tutorial runs; scope clean; struct and extractor changes backward compatible.

F1 MEDIUM. A15 @test_broken rows can mask regressions: loose bounds instead (R grad at Julia theta <= 1e-3; Julia FD grad at R point <= 1e-2; converged = false recorded; beta abs gap <= 1e-4).
F2 LOW. A15 returns converged = false within about 2e-10 of the optimum: fit_precision_multivariate uses an absolute g_tol = 1e-5 on the FD gradient (src/precision_multivariate_fit.jl:296), below what FD delivers at objective ~7372. Shared with the bridge; owner is the convergence lane (#485). Options: one Newton polish with the existing FD Hessian (moves estimates ~1e-8, bridge receipts must be re-run) or a scale-aware rule. Interim: docstring note on g_tol. Promote A15 on objective and estimate receipts.
F3 DECISION. Ainv: R's keyword densifies (solve(Ainv), reorder, ridge); the sparse route is only the global phylo_vcv. Twin the keyword literally (vcv = inv(Ainv) before subsetting); sparse global route stays unadmitted.
F4 LOW. species_levels and Newick tree are R-faithful; built-in JSON reader adequate.
F7 INFO. Julia 4-8x slower (FD gradients, accepted in D-300).
