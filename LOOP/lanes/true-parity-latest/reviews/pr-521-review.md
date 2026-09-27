# Independent review of PR #521 (head c645da0c0, base 1385b0490), 2026-09-27

Verdict: NON-BLOCKING. Reviewer: fresh Fable agent, own seeds 9101 to 9114, read-only on GitHub.

Whole fit (14 datasets, `fit_nb_gllvm_grouped`, Julia 1.10): 8 cells higher on the branch (by 0.09 to 34,151), 6 unchanged (|Δ loglik| < 5e-5), 0 lower. Every cell where main reported converged after 4 to 11 iterations was a false optimum (fitted r 2e-6 to 1e5 against a true r of 2 to 50); the branch recovers r in a plausible band. Total time main 332 s, branch 1393 s (4.2x); per evaluation about 1.3 to 1.7x.

Findings (all non-blocking):
1. Fix is real: 2400 random stress sites against an independent Newton reference: main 1524 off by more than 1e-6 (errors up to 3.7e21, never -Inf); branch 105 off, 8 -Inf (all in the |η| = 30 clamp regime); only 4 sites with the kernel lower than the reference (at most 0.28).
2. Author point (a), the two lower cells: artefact of comparing two different objectives; under the branch kernel the branch optimum is at least as high as main's point everywhere, and main's point is -Inf in 4 cells.
3. Author point (b), 1e-3 shortcut: the Fisher stage oscillates (fails 1287/2400 at maxiter 100) but a rejected step hands over to observed Newton, so no drift to -Inf (the mixed-bridge flaw is absent). Follow-up: under LogLink run observed Newton first; give the merit test relative slack (q1 >= q0 - 1e-12*abs(q0)).
4. Convergence rule certifies on step size, so at |η| beyond the clamp (μ >= 1e13) it can pass at non-stationary points (38 stress sites). Follow-up: add a gradient test. Clamp regime only.
5. Author point (c): the extra converged = false verdicts are genuine boundaries (profiles monotone in r, flat above 1e4); one real non-convergence (illcond_rhigh hit the 500-iteration cap after 495 s).
6. Author point (d): +70% per evaluation confirmed; total wall time on hard data is up to 4x or more because fits no longer stop early. Suggest CHANGELOG wording: "per-evaluation cost up about 1.5x; fits that previously stopped early now run to completion and can take several minutes at p = 12 to 15, n = 100 to 120".
7. β_init / Λ_init plumbing correct; defaults reproduce the old arithmetic. Notes: pack_lambda silently drops strict-upper entries (docstring sentence); Gamma and Tweedie grouped fitters lack the keywords.
8. Tests sound (hash-verified fixtures, relations not seed outcomes), 61/61 green. Before merge: register both new test files in test/runtests.jl and add a CHANGELOG entry.
9. The 20x Fisher retry for non-log links with hessian = :fisher is untested; document or cap.

Not covered: the author's own 48-cell panel; non-log links, mask, offset, fit_nb_gllvm_grouped_cov, CIs on the branch; `_grouped_laplace_mode` (getLV) still has the shortcut.

Scripts and logs: session scratchpad of the true-parity-latest lane (r521_* files, panel_*.log, stress_*.log).
