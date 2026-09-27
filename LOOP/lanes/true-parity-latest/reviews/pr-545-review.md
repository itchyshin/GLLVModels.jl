# Review of PR #545, phylo latent spec (head 92c11f2fb), 2026-09-27: NOT READY (fence ledger), READY after one revision

Verified: A14/A15 are bare Gaussian phylo_latent(species, d = K) (scoreboard lines 39-40); augmented precision (root dropped, tips last, unit-height scaling, ultrametric gate), 1e-8 dense ridge, H2 = 1 on a bare fit (extract-omega.R:520); Julia's PrecisionPhy / fit_precision_multivariate is R's phylo_rr block; Destination B receipts pair with 0.7.0 at 1.07e-14 / 6.4e-14 (precedent only: the carry rule bars them, cpp +866, fit-multi +2375 lines); two appended fields safe via the outer constructor; A15 FD gradient = 120 evaluations of an order-396 factorisation, well under 3 h.

F1 HIGH. STRUCT-PHY-TREE-PROPTO (phylo_scalar) is in A14's own ledger row but fenced unilaterally: make it a maintainer question (recommend fence; A14 then promoted with one planned case UNPAID).
F2 HIGH. Temporal-phylo files (8, 6, 4, 5 blocks) labelled "out of boundary"; temporal is inside P1 and #535 owns them (replicated one deferred there, others dev-only there). Relabel "owned by #535".
F3 MEDIUM. Non-Gaussian phylo_latent matrix cells are dev-only (heavy, not bare), not out of boundary; add maintainer question: is non-Gaussian phylo_latent inside P1 (recommend later slice via phylo_glm.jl).
F4 MEDIUM. rho != 1 refusal borrows an R sentence R does not emit for bare phylo_latent (R accepts rho, fit-multi.R:4742-4750): make it a Julia scope fence with its own tag plus a maintainer question.
F5 MEDIUM. Polytomies: R accepts; Julia PrecisionPhy(::AugmentedPhy) requires 2p - 2: admit via the raw-triplet constructor or refuse explicitly.
F6-F8 LOW. va-r3 file is R-internal; missing-predictor file is mi() grammar (zero phylo_latent calls); phylo slope fences cite D-297 and note heavy.
F9 counts change accordingly.
