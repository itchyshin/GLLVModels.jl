# Review of PR #559, namespace rows at P1 (head ac78084cb), 2026-09-27: fine as a draft, BLOCKING for any EVIDENCED claim

1. HIGH. A Tier 0 pass asserts only isdefined(GLLVModels, Symbol(name)); notes are hand-typed. proportions and unpack_lambda are not exported; TwoLevelFit and OrdinalFit are types; 5 rows pass on them. The checker counts Tier 0 rows as bound, indistinguishable from numeric twins: D-295 row 5 defeated by content. Fix: measure isexported + callable; checker reads evidence_tier, reports bound_numeric vs bound_registration_only, C1_MET requires registration_only = 0 unless signed.
2. MEDIUM. Newly passing: deviance is a genuine semantic twin (-2 logLik); tidy, check_gllvmTMB, confint_inspect share the name but not R's contract: registration-only.
3. LOW. Receipts real and hash-true; no .unlazy references.
4. LOW. P0 default unchanged; the P0 Julia self-test failure predates the PR and is in no CI job.
5. OK. 17 not re-measured and 2 blocked are correctly unbound.
6. MEDIUM. The separate case-map-namespace.json is invisible to the default checker; fold into case-map.json after #533 (safer than a multi-file checker); record expected totals (107 rows, 90 required) in GATES.md.
Minor: eight *-JULIA-BRIDGE-COMPARE rows are circular (compatibility_adapter).
Also: #559 was closed when #539 merged and deleted its base branch; re-created on main as v2.
