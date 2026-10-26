## 1. Goal

Implement the approved GLLVM.jl side of the first-wave documentation audit, correcting D011, D012, and D036 while coordinating with the separate gllvmTMB changes. D010 requires its bridge/inference owner's approval and remains deferred.

## 2. Implemented

- D011/D012: clarified that latent score SDs condition on fitted fixed parameters, stated the joint negative-log-likelihood Hessian precision convention, and distinguished TMB's default parameter-uncertainty propagation from a conditional comparison using `ignore.parm.uncertainty = TRUE`.
- D036: added `using Distributions: Poisson` to the documented Poisson cross-validation example.

## 3a. Decisions and Rejected Alternatives

- D010 remains deferred because no bridge/inference owner approval was received.
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
- Deferred: D010 pending bridge/inference owner approval.

## 8. Consistency Audit

Checked the API example in source and its generated reference page, plus the generated SE/profile machinery page and its cross references. The build rendered the docs successfully with deployment disabled.

## 9. What Did Not Go Smoothly

The closeout helper resolves its root to the brain repository rather than this package worktree, so the records were created directly. Julia docs dependencies required a scratch depot; deployment was explicitly disabled.

## 10. Known Residuals

- D010 remains pending owner approval.
- The brain-level closeout wrapper resolves its root to the brain repository and evaluates unrelated acceptance ledgers there. Its check fails on those unmet campaign gates; the direct after-task structure checker passes for this report.
- No deployment or public parity or calibration claim was made.

## 11. Team Learning

Examples that name a family marker should import it from the package that exports it; do not assume the model package re-exports every distribution type.

Memory receipt: used the repo instructions and routed GLLVM/gllvmTMB dossier notes; the route manifest reported no manifest. The twin ownership and public-claim boundaries shaped the wording. Golden Set: not checked because this slice changed documentation only and no known-mistake engine class was in scope.

## 12. Cross-Product Coverage

Covers: the named Julia documentation and Poisson CV example surfaces.

Does NOT cover: D010, engine behavior, full R article-suite success, deployment, parity, or interval calibration.
