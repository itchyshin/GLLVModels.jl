## 1. Goal

Complete the approved twin documentation remediation, correcting D011, D012, D036, and the paired R-side D010 bridge finding. Preserve the separate package branches and do not deploy or merge.

## 2. Implemented

- D011/D012: clarified that latent score SDs condition on fitted fixed parameters, stated the joint negative-log-likelihood Hessian precision convention, and distinguished TMB's default parameter-uncertainty propagation from a conditional comparison using `ignore.parm.uncertainty = TRUE`.
- D036: added `using Distributions: Poisson` to the documented Poisson cross-validation example.
- D010 was resolved in the paired gllvmTMB branch: non-finite Wald bounds now report unavailable with a diagnostic note, and the replacement example returns 14 finite bounds. Shinichi authorized completion as package maintainer.

## 3a. Decisions and Rejected Alternatives

- D010 was resolved on the companion R branch after maintainer authorization; its original source evidence remains preserved in the audit checkpoint.
- Kept the work within documentation and example imports; made no engine or user API change.
- Assumption: the cross-product work should be coordinated as one goal while retaining separate package PRs, as approved.

## 4. Files Touched

- `docs/src/se-profile-machinery.md`
- `src/cv.jl`
- `docs/dev-log/check-log.d/2026-10-26-gllvm-doc-remediation.md`
- `docs/dev-log/after-task/2026-10-26-gllvm-doc-remediation.md`

## 5. Checks Run

- Documenter/VitePress build with deployment disabled: exited 0; doctests, cross references, page rendering, and VitePress build passed.
- Generated HTML and intermediate Markdown contained the corrected uncertainty explanation and explicit import.
- Exact random-cell and site-block Poisson CV examples executed and returned finite metrics.
- `git diff --check` and prose style checks passed.

## 6. Tests of the Tests

No package behavior changed and no test files were added. The exact documented CV examples were executed, which verifies the import and call syntax for both advertised split routes. No negative-control test was appropriate for this documentation-only change.

## 7a. Issue Ledger

- Fixed: D011, D012, D036.
- Fixed: D010 on the companion gllvmTMB branch; exact negative-control and revised-example outcomes are recorded in the paired R after-task report.

## 8. Consistency Audit

Checked the API example in source and its generated reference page, plus the generated SE/profile machinery page and its cross references. The build rendered the docs successfully with deployment disabled.

## 9. What Did Not Go Smoothly

The closeout helper resolves its root to the brain repository rather than this package worktree, so the records were created directly. Julia docs dependencies required a scratch depot; deployment was explicitly disabled.

## 10. Known Residuals

- The brain-level closeout wrapper resolves its root to the brain repository and evaluates unrelated acceptance ledgers there. Its check fails on those unmet campaign gates; the direct after-task structure checker passes for this report.
- At 2026-10-27 01:31 UTC, GLLVModels.jl PR #1017 was open at `75fd1f80722cbefa26443e777e3bbb2eb80ff5f6`; Julia shards, twin tests, and Documenter had passed. That head's separate Documenter deploy job was cancelled.
- At 01:31 UTC, the paired gllvmTMB PR #1704 was open at `f16c4d0151e340d57c9da8daebd1b36c7cd326e5`; all four R-CMD-check shards were in progress.
- At 03:08 UTC, PR #1017 head `c0440d9bc3f19feff7141602cffb505f6d3f812b` had passed all four Julia shards, both twin tests, Documenter, and the latest PR wording guard. CI also published a PR preview; this was not a production site deployment.
- At 03:08 UTC, PR #1704 head `cb852bf7b9d34bcf08b59e87c75558c9a4e84153` had passed all four R-CMD-check shards. Both PRs were open and unmerged.
- The app refused to archive the superseded audit worktree because it is protected by a pinned task or workspace; its clean checkpoint and evidence remain preserved. The planning conversation is archived.
- No production deployment or public parity or calibration claim was made.

## 11. Team Learning

Examples that name a family marker should import it from the package that exports it; do not assume the model package re-exports every distribution type.

Memory receipt: used the repo instructions and routed GLLVM/gllvmTMB dossier notes; the route manifest reported no manifest. The twin ownership and public-claim boundaries shaped the wording. Golden Set: not checked because this slice changed documentation only and no known-mistake engine class was in scope.

## 12. Cross-Product Coverage

Covers: the named Julia documentation and Poisson CV example surfaces, plus the linked R-side D010 remediation.

Does NOT cover: Julia/R numerical parity, interval calibration/coverage, complete R package checks, deployment, or merge.
