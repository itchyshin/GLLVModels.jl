# After-task: issue sweep across GLLVModels.jl and gllvmTMB (2026-10-01)

Platform: Claude Code (Opus 5.5 orchestrator). Lane: `claude/issue-sweep-*` worktrees under
`~/local-scratch/lanes/`. Plan: `~/.claude/plans/can-you-start-an-inherited-horizon.md`.
Acceptance ledger: `.unlazy/issue-sweep-20261001/` (untracked, in the main checkout).

## 1. Goal

The goal was to fix the safe, concrete open issues in both repositories on small branches. Each fix
needed a test that fails on `origin/main` and passes after it. Issues already fixed on main were to be
closed with a comment. The true-parity lane, the NB lane, and the gllvmTMB engine stayed untouched.

## 2. Implemented

Nine draft PRs, none merged.

| PR | Issues fixed | Notes |
|---|---|---|
| GLLVModels.jl #667 | #134 #139 #141 #150 #151 #153 #159 #162 | input checks |
| GLLVModels.jl #668 | #138 #143 #145 #146 #147 #158 #161 | robustness; #158 and #161 are doc/comment fixes |
| GLLVModels.jl #669 | #536 #537 #538 | README and Quick start, with a test that runs the README code |
| GLLVModels.jl #670 | #137 | derived-profile constraint gate; #142 withdrawn after verification |
| GLLVModels.jl #671 | #574, #498 item 1 | damped ordinal mode search; binomial runaway-loading warning |
| gllvmTMB #1338 | #1335 #1333 | restart selection; default `n_init = 1` unchanged |
| gllvmTMB #1339 | #1149 | family-CDF branch tests |
| gllvmTMB #1340 | #1167 | one-iteration termination flag; #897 not fixed |
| gllvmTMB #1341 | #1326 #1320 | README |

### Second wave (approved 2026-10-02)

| PR | Issues fixed | Notes |
|---|---|---|
| GLLVModels.jl #677 | #575 | Tweedie series density about 390× faster; values unchanged (8.9e-16) |
| GLLVModels.jl #679 | part of #505 | `fit_phylo_gaussian` verdict only; scalar NB1 change dropped (no failing reproduction) |
| GLLVModels.jl #680 | #573 | ZIP/ZINB never end below their nested count fit |
| gllvmTMB #1342 | #1330, #1334 | R-side warnings only; root causes still open |

Closed with a verified comment: GLLVModels.jl #152, #129, #482, #499 and gllvmTMB #1163.

## 3a. Decisions and Rejected Alternatives

- gllvmTMB work in a Claude lane: Shinichi chose "Both repos". This is recorded as a dated
  exception to D-220 in the vault's `memory/DECISIONS.md` (commit `64e569e0`).
- #142 withdrawn from #670: The clamp covered only two new wrappers. The public
  `profile_ci_derived` still returns bounds above 1, and R's 0.999 ceiling can exclude the estimate.
  Floor and ceiling semantics need a maintainer decision.
- New exports rejected in #670: Exporting new public functions is an API decision, so the
  helpers were kept internal before being withdrawn.
- #897 not fixed: Reusing the binomial detector would flag 39.2% of healthy ordinal_probit fits
  (the evidence is in #1097).
- Parity pin loosened in #1338: It was 1e-10, Mac-only; it is now 1e-6. The repository had just
  replaced hard-coded cross-platform optima for the same reason.
- #476, #577, #133 and gllvmTMB #1080 left open: #476 belongs to the parity lane. For the other
  three the fix evidence was partial, or the thread shows open items.

## 4. Files Touched

GLLVModels.jl:
- `src/likelihood.jl`, `src/lowrank_cholesky.jl`
- the phylogenetic validators
- `src/bridge.jl`, `src/confint_derived.jl`, `src/confint_derived_wald.jl`
- `src/em_fa.jl`, `src/em_squarem.jl`
- `src/families/beta.jl`, `src/families/ordinal.jl`, `src/families/binomial.jl`
- `README.md`, `docs/src/tutorial.md`, `docs/make.jl`, `CHANGELOG.md`, `test/runtests.jl`
- five new test files

gllvmTMB:
- `R/fit-multi.R`, `R/diagnose.R`
- `README.md`, `NEWS.md`
- `tests/testthat/test-family-cdf-args-1080.R` and two new test files

No PR touches `docs/dev-log/core070/**`, `docs/dev-log/check-log.md`,
`src/families/grouped_dispersion.jl` or gllvmTMB `src/*.cpp`. The `scope_check.sh` gate checks this
on every slice.

## 5. Checks Run

- Targeted test files: Every slice's targeted file passes. They were re-run with
  `gate-check --reverify` after merging the latest `origin/main` into four of the Julia branches.
- Existing tests: The tests that exercise the touched functions were re-run with zero failures.
  The counts are in each PR body.
- Full suites: These run in CI on each PR: 8 Julia shards, and gllvmTMB R-CMD-check in 4 shards.
  CI was still pending when this was written. A local Julia full suite was stopped after 56 min,
  over the 30 to 60 min estimate, while it was testing a superseded commit.

## 6. Tests of the Tests

- Revert checks: For every code slice, the new test file was run against `origin/main` source:
  - JL-A, JL-B and JL-D were checked by a fresh Opus verifier.
  - The gllvmTMB slices were checked by a different Sonnet agent.
  - JL-E was checked by the orchestrator.

  Each issue's test fails or errors on main. The expected exceptions pass there: #158 (docstring
  only) and the parity or healthy-fit controls.
- Mutation check: #1149 is coverage-only. Changing NB1 `size = mu/phi` to `mu*phi` failed three
  new expectations.
- **The verifier found two weak spots, both repaired before PR:**
  - #141's NaN assertions passed on the old code. Underflow and negative-variance cases were added,
    and they fail on main.
  - #142 was partial, so it was withdrawn.

## 7a. Issue Ledger

Fixed (PR open):
- GLLVModels.jl: 134, 137, 138, 139, 141, 143, 145, 146, 147, 150, 151, 153, 158, 159, 161, 162, 498 (item 1), 536, 537, 538, 573, 574, 575; #505 in part (phylo Gaussian only)
- gllvmTMB: 1149, 1167, 1320, 1326, 1333, 1335; warnings only for 1330 and 1334 (issues stay open)

Closed as already fixed: GLLVModels.jl 129, 152, 482, 499; gllvmTMB 1163.

Needs the maintainer:
- GLLVModels.jl: 131, 135, 136, 140, 142, 149, 156, 157
- gllvmTMB: 897, 1330, 1331, 1334, 1020, 872

Carried over: #555 (NB postfit methods) waits for NB PR #662 to merge.

Deferred:
- NB lane: #615, #553, #554, #552, #477, #503
- parity lane: #476

## 8. Consistency Audit

- CHANGELOG conflicts: Four Julia branches conflicted with main's CHANGELOG. Both entries were
  kept, the merges are pushed, and the gates were re-verified.
- Shared files between open PRs: #667 and #670 both edit `src/confint_derived.jl`, in different
  functions. All five Julia PRs append one line to `test/runtests.jl`, which will need trivial
  rebases as they merge in turn.
- **Parity safety.**
  - #1338 leaves the default `n_init = 1` path byte-identical on macOS.
  - #671 changes the ordinal mode search only on steps that lowered the site log-posterior. The R
    fixture twin (`test_ordinal_logit_twin`) still passes 29/29. The parity owner should still look
    before merge.

## 9. What Did Not Go Smoothly

- The verification scout did not run anything: Its #152 "output" was invented, and for the other
  three issues it only saw that test files exist. The orchestrator re-ran the regression tests before
  closing anything.
- Orphaned processes: The scout left two full `Pkg.test` runs going (one ran 25 min at 99% CPU).
  They were found and killed.
- Two subagent reports did not arrive: One was a "placeholder" hand-back; one report was never
  delivered. In both cases the result was read from the worktree.
- Over budget on agents: The plan allowed six new agents; seven were spawned. After that, finished
  agents were reused.
- Malformed ledger edit: A perl one-liner broke a gate file; it was rewritten in Python.

## 10. Known Residuals

- Full-suite CI for all nine PRs was pending at close.
- #138: the masked `getLV` route was not exercised. Some `getLV` methods now abort `bridge_fit`
  instead of returning empty scores.
- #143: the test covers the new helper, not its one-line wiring.
- #146: `converged` and the iteration count describe the SQUAREM run even when the polished point is
  returned.
- #574: the issue's own bfi gap was not re-measured, because the data is not in the repo.
- #1167: the positive test mutates a real fit's `opt` and does not reproduce the 1e21 failure from data.
- #669: the Documenter site was not built locally.

## 11. Team Learning

- A Haiku verification scout reported "FIXED" from file presence. Close-with-comment needs a re-run
  by the orchestrator or a stronger verifier before anything is closed.
- A revert check run by a fresh verifier caught two weak tests that the builders had reported as
  "failed before: yes". Keep it as a standing gate.
- Agents that launch `Pkg.test` with `test_args` start the whole suite and can orphan it. Briefs
  should forbid full-suite runs explicitly and ask agents to kill what they start.

- CI caught a regression the targeted tests missed. The #153 binary-tree check went into the
  Newick parser that the phylo-latent route shares, and that route admits polytomies as R
  does. Fixed by moving the check into `augmented_phy` (`137602057`). A check placed in a
  shared helper hits every caller; grep the helper's callers before tightening it.

## 12. Cross-Product Coverage

The R and Julia twins were compared only where an issue needed it:
- #147 copies gllvmTMB's `tiny_y`.
- #137 copies R's `.fix_and_refit_constraint_tol`.
- #498 copies gllvmTMB's runaway-loading rule.

This arc does NOT cover a twin-wide check of the other robustness fixes (#134 to #162) against
gllvmTMB's behaviour, the bridge `engine = "julia"` gate, or the gllvmTMB C++ engine.

## Addendum, 2026-10-03: decision-list round

The maintainer asked for "do all, merge when green". That covered the held #671 and #680, #555, and the decision list. Three landing PRs carried the work:

| Landing | Contents |
|---|---|
| #1343 (gllvmTMB, `8010cfd4c`) | #1338 to #1342 |
| #683 (`fa692c476`) | #667 to #670, #672, #677 |
| #704 (`17c4e6a80`) | #696 to #700, #703, #671, #680 |

#679 merged on its own (`896a0a228`) after a Linux fix.

Every landing tree passed the parity lane's three checks. On the #704 tree, 64 receipts reproduce within 0.001 × tolerance, 297 assembled rows are current, and all negative controls pass.

### Decision-list outcomes

- #140 and #156 (#696): Non-converged bootstrap refits are dropped and counted. SD rows are reported on the raw scale.
- #142 (#697): `profile_ci_derived` gets an optional `bounds` keyword, and `lower <= estimate <= upper` always holds.
- #135 (#698): The W-tier covariance now matches C++. The identification caveat is filed as #702.
- #555 (#700): NB grouped postfit methods are added.
- #136 (#703): Docs only. The note took four review rounds, and each round measured and corrected an overclaim. Final wording: with `K_phy = 0`, the sign of each group of rows that `Σ_phy` links can flip on its own. For a tree-derived `Σ_phy` these groups always include the root's two daughter clades, because `sigma_phy_dense` drops the root edge.
- #131, not changed: On a matched fit, Julia's `communality()` already equals R to 5 decimal places. The real mismatch is `extract_communality(level = :unit)` on `has_diag` fits, filed as #701.
- #149, partial: n < p is pinned on the Laplace path, and the Gaussian guard is now an `ArgumentError`. The rank-condition guard and the Lognormal routing stay open.
- #157: Park note posted.
- gllvmTMB #1020, #1331 and #872: Held, as recommended.

### More GitHub keyword accidents

The fixes come first; the lessons follow under Team learning.

- "Does not fix #N" closed #N: This hit #142 and gllvmTMB #897; both are reopened.
- "Fixes #134, #139, ..." closed only the first issue: Fifteen issues were closed by hand with citing comments.
- The commit subject `fix(n<p): ... (refs #149)` closed #149: It is reopened.

### CI failures on #704, all fixed on the landing branch

- Documenter: `tools/check_reader_surface.py` rejects "issue #N" in rendered docstrings, and seven #135 docstrings said "(issue #135)". They now say "(#135)".
- Julia 1 (1.13) shard 3: A #142 test assumed the raw profile overshoots 1, which happens on macOS but not on Linux. The clamp is now asserted only when the raw bound overshoots; it stays pinned deterministically by unit tests.
- Julia 1 (1.13) shard 4: `using Logging` was not declared in `test/Project.toml`, and Julia 1.13 refuses undeclared standard libraries. The test now uses `Base.CoreLogging.Warn`, which is the same constant.

### Team learning (additions)

- Closing keywords: Write `Fixes #N` once per line, for fixed issues only. Never put "fix" next to an issue number that should stay open: not in prose, and not in a commit subject such as `fix(scope): ... #N`.
- Platform-dependent test numbers: Assert a property under the condition that produces it (`raw > 1 ⇒ clamped to 1`), never a number one optimiser path happens to give. CI caught this twice in the sweep, in #679 and in #142.
- Undeclared standard libraries: Julia 1.10 locally loads standard libraries a test project does not declare, but Julia 1.13 on CI does not. Scan new test files' `using` lines against `test/Project.toml` before pushing.
- Docs written from inference: These need measurement as well as review. Each #136 round that "fixed" the wording from inference introduced a new overclaim; the rounds that measured the claim (block flips, `sigma_phy_dense` on a tree with a root edge) converged.
