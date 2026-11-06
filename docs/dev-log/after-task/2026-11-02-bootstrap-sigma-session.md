# After-task: bootstrap_Sigma_r session, true-parity P1 lane (2026-11-02)

Lane: GLLVModels.jl true-parity P1 against pinned R gllvmTMB 9539352f (read-only reference). Resumed from `docs/dev-log/handover/2026-11-01-claude-handover.md` (handover commit 44a55f512, which is on Codex branches only, not on main). PR states below were read from `gh` on 2026-11-02, not copied from memory.

## 1. Goal

Build, wire, test and honestly document a Julia `bootstrap_Sigma_r` driver that follows the pinned R behaviour, while the parity row stays UNBOUND. Clear the review findings and test gaps found along the way. Do not touch `src/confint_derived_wald.jl` (hold on #1036), the PR #990 lane, or Codex branches.

## 2. Implemented

Merged to main by this session:

- #1054: the unwired `bootstrap_Sigma_r` driver.
- #1056: wired into the module, with `Logging` added as a test dependency.
- #1057 and #1058: the first R-versus-Julia evidence note, then a correction (boot_median, the SE gap, refit failures).
- #1059: hardening (validation, reporting, tests). For `has_diag` or `K_W > 0` fits the optimizer now runs with `iterations = 2000` and `f_tol = 1e-12`.
- #1060: the optimizer claim is scoped to `TwoLevelFit`, and an optimum note was added.
- #1061: N8 memory and wall-time design note.
- #1066: a test that pins the R_* clamp tolerance.
- #1067: four test gaps closed (M1, M4, M5, M6), each a mutation survivor.
- #1070: the flaky testset 25 gate was replaced with one that does not depend on the platform iteration path.

Open and mine, not merged: #1068 (worked example in the docs, `se-profile-machinery` added to the nav) and #1069 (Stage 1 phy tier for `PrecisionMultivariateFit`, structural evidence only).

Measured diagnostics, none of them a pass: ordinal Newton polish took Julia max|grad| from 9.4e-6 to 1.6e-6 and R from 4.3e-4 to 4.7e-7. A FAMILY-00 re-measure gave 31/31 pass, but only through a hand-edited copy of the oracle marker, and no receipt was written. A Stage 2 pre-run (the phy tier on `JointPhyloGroupedGaussianFit`) passed the 5% gate with worst case 2.5%.

## 3a. Decisions and Rejected Alternatives

User rulings of 2026-11-01 ("OK as you recommend"):

- The N9 signed-FAIL guard covers only ordinal, iSDM, fungi and spider. Beetle, fungi and urbanisation are guarded in `SIGNED_FAIL_ROWS`.
- Spider and beetle stay FAIL, with no refit.
- C2 rows stay UNBOUND, each with a decision packet.
- No FAMILY-00 oracle rebuild. Rejected: rebuilding it to turn the 31/31 diagnostic into a receipt.
- "Sign, wait for #1036": `src/confint_derived_wald.jl` is not touched.

Choices made by this session:

- Merge the driver unwired first and wire it second, so each review stayed small. Rejected: one large PR.
- Scope the optimizer finding to `TwoLevelFit` after the evidence showed that the fixture-c 0.4% gaps on `GllvmFit` are 500-iteration-cap failures, not worse optima.
- Hold `claude/sigint-twolevel` unpushed (see 7a).

Assumptions made without asking:

- "OK as you recommend" was read as covering every recommendation in the handover list. If a recommendation was meant to stay open, the matching ruling above is void.
- Ordinal Newton polish is diagnostic only. If it is wanted as evidence, it needs a receipt and a rule for the oracle.

## 4. Files Touched

This PR adds one file: `docs/dev-log/after-task/2026-11-02-bootstrap-sigma-session.md`. The session's earlier merged PRs touched `src/` bootstrap code, `test/test_bootstrap_sigma_r.jl`, `test/Project.toml` and `docs/dev-log` notes. They are listed in each PR, and this PR does not repeat them. Nothing under `src/`, tests, PR #990 or `src/confint_derived_wald.jl` is touched here.

## 5. Checks Run

- `gh pr view` on 18 PR numbers to settle the merged/open table above.
- `git fetch origin`, then a worktree from origin/main at b1eb85a46 (the #1070 merge).
- Per-PR test runs (full `Pkg.test()` or the bootstrap test file) happened before each merge and are recorded in those PRs. They were not re-run for this docs-only PR.
- `Rscript tools/check-after-task.R` from the brain, and `python3 tools/slop_check.py`, on this file (results in the PR body).

## 6. Tests of the Tests

The #1067 tests close four mutation survivors: the production line was broken on purpose and the new test was confirmed to fail (M1, M4, M5, M6). The #1066 clamp test pins the R_* tolerance, so changing the constant fails it. The #1070 gate was replaced because the old one passed or failed depending on the optimizer's iteration path, which is a test that does not test the claim.

Not shown: mutation evidence for #1068 and #1069. #1069 carries structural checks only.

## 7a. Issue Ledger

Fixed or merged: the PRs in section 2.

Open, not mine to merge:

- #1055 (N9 SIGNED_FAIL guard) is blocked behind the other lane's #1046, which refreshes stale zi receipt fingerprints. #1065 duplicates the #1046 fix.
- #1062, #1063 and #1064 are recovered drafts from an archived session.

Held locally, unpushed:

- `claude/sigint-twolevel`: an `InterruptException` at `twolevel.jl:229`. Held until the user rules whether the #1036 hold covers `twolevel.jl`. #1036 is open and is titled "confint family: rethrow InterruptException".
- `claude/c2-animal-latent-v3-20261101` and `claude/c2-post-coef-empty-bookkeeping-20261101`, held for the C2 option rulings.
- Unsigned gates drafts (no branch of that name was found when this report was written).
- `claude/bsig-stage2-jpg-20261102` (tip bfa00e7a2): the Stage 2 phy tier, pending verification.

## 8. Consistency Audit

After the optimizer finding, I checked the other fit types for the same class of problem. Result: the roughly 0.6 to 0.9% worse optimum is on `TwoLevelFit` only, and these are converged local optima. The `GllvmFit` fixture-c gaps of 0.4% are iteration-cap failures. They are different mechanisms and the notes now say so. I also swept comments for the stale optimizer claim (#1060). A reader could still take the iteration-cap change as a general fix, so the note scopes it to `has_diag` or `K_W > 0`.

## 9. What Did Not Go Smoothly

- The first evidence note (#1057) needed a correction (#1058) covering boot_median, the SE gap and refit failures.
- Testset 25 was flaky across platforms and needed its own PR (#1070).
- The 31/31 FAMILY-00 result came from a hand-edited oracle marker, so it cannot be used as evidence.
- Several drafts were recovered from an archived session (#1062 to #1064) and are unverified beyond what is stated in them.
- The goal-meter plugin (v0.6.3) needs `/reload-plugins` before its meter is right.

## 10. Known Residuals

- A `has_diag` B-tier matches R's unit Sigma only for one-row-per-cell data. Beyond that it is UNVERIFIED (#1064 is a draft).
- Phy-tier Stage 1 evidence is structural only, with no R comparison.
- Stage 2 passed the 5% gate (worst 2.5%), but the Hessian is non-PD in many converged replicates.
- The R comparison run (N1, about 1.5 to 2.5 h) has not been run. It needs approval of the criteria and permission to install the pinned gllvmTMB into a private R library on Totoro.
- The `bootstrap_Sigma_r` parity row stays UNBOUND. Nothing here is a parity claim.

## 11. Team Learning

- Check that a "worse optimum" is a real optimum before blaming the optimizer. Two different mechanisms looked the same until the iteration count was read.
- A test gate that depends on the optimizer's path is not a gate. Pin the quantity and leave the route free.
- A diagnostic through a hand-edited oracle marker is a diagnostic and cannot serve as a receipt.
- Questions for the user, each with a recommended answer: (1) merge #1046 so #1055 unblocks (recommended: yes); (2) does the #1036 hold cover `twolevel.jl`, so the SIGINT patch can ship; (3) approve the N1 R comparison and the private-library install on Totoro; (4) C2 rulings, with option A recommended for animal_latent and option 1 for POST-COEF-EMPTY.

## 12. Cross-Product Coverage

Cross-cutting items touched: the optimizer setting (`iterations = 2000`, `f_tol = 1e-12`) and the phy tier of the bootstrap.

- Optimizer setting covers: `TwoLevelFit` with `has_diag` or `K_W > 0`.
- Optimizer setting does NOT cover: `GllvmFit`, `PrecisionMultivariateFit`, `JointPhyloGroupedGaussianFit`, and fits with `has_diag` false and `K_W = 0`.
- Phy tier covers: `PrecisionMultivariateFit` (#1069, open) as structural evidence. The Stage 2 branch for `JointPhyloGroupedGaussianFit` is unpushed.
- Phy tier does NOT cover: any R comparison, replicated-row data, or the other fit types.
- The driver does NOT cover: public parity binding (row stays UNBOUND), the N8 memory design implementation, or the held SIGINT handling in `twolevel.jl`.
