# After-task: GLLVModels reader PR triage

## 1. Goal

Classify the stacked reader-documentation PRs (#429, #428, #439, #437, #433, plus PROTECTED #430) as OWED, DONE,
RETRACTED, or PROTECTED with checkable evidence, and measure exactly how #428 conflicts with current main, so the
maintainer has one decision packet instead of five ambiguous PR pages. This is handover step 1 in
`docs/dev-log/handover/2026-09-21-claude-handover.md`.

## 2. Implemented

Wrote `docs/dev-log/reader-pr-triage-2026-09-21.md`: a table classifying #429, #428, #439, #437, #433, and #430, each
row carrying its title, branch, head SHA, draft state, mergeable state, CI checks, files touched, overlaps with the
other rows, a classification, an Evidence cell naming the command that backs it, and a recommended action. It adds a
"Conflict probe" section for #428 with the raw `git merge-tree` output and a per-file table showing which later main
PR each of the 7 conflicting files traces to, and a "Recommended next" section ordering the maintainer's open
decisions.

The headline finding: PR #429 was never merged into `main`. `gh api repos/itchyshin/GLLVModels.jl/pulls/429` shows its
base branch was `codex/reader-text-recovery-20260919`, which is PR #428's own head branch, not `main`. Its squash
commit exists only on that one branch; `git merge-base --is-ancestor` against `origin/main` returns false, and a file
#429 added (`docs/src/what-can-i-fit-today.md`) does not exist on `origin/main` at all.

The file was then revised once, after an independent Fable (Rose + Pat) review found 0 BLOCKING items and confirmed
the headline reproducible, applying six GLLVModels-specific findings: relabelled #429 from "DONE" to "OWED (carried
inside #428; not on main)"; narrowed the "split the clean files" recommendation to exactly 13 files, holding back 4
that need a named owner; added each row's head SHA; pointed the "correct the record" step at the one handover file
that is actually wrong; added a short "Your clicks" box with a drafted default at the top of "Recommended next"; and
noted that this triage's and the sibling DRModels triage's commits carry different `Co-Authored-By` model trailers by
design.

PR #444's body was appended twice (fetch current body, then `gh pr edit --body-file`, never overwritten) with dated
summaries of each pass. The PR stayed a draft throughout; nothing was merged, rebased, or force-pushed.

## 3a. Decisions and Rejected Alternatives

No merge was attempted on any PR: #428 is `CONFLICTING` and #439/#437/#433 are drafts, so none meets the handover's
own merge gate of "current, clean, settled-green, explicit ownership."

Used `git merge-tree --write-tree origin/main origin/codex/reader-text-recovery-20260919` to measure #428's conflict,
instead of an actual rebase or a merge attempt. `merge-tree` writes tree objects and reports conflicts without
creating a branch, a worktree, or moving any ref, which was the only way to get an exact conflict count and file list
while honoring "never rebase, never check out, never push any PR branch."

Rejected one of the handover's three suggested paths for #428: opening a fresh lane "on top of #429." This is not
well defined here, because #429 has no existence outside #428's branch; closing #428 would discard #429 with it.

Relabelled #429 from an initial "DONE-but-mislabelled" to "OWED (carried inside #428; not on main)" after review
pointed out that "DONE" is one of the four permitted classification tags and means nothing is owed, which contradicted
the row's own evidence that the content is not on main.

Added a per-row head SHA column rather than relying on the single main-SHA stamp at the top of the file, since a PR
branch's head can move independently of main and the original stamp could not show that.

## 4. Files Touched

`docs/dev-log/reader-pr-triage-2026-09-21.md` (created, then revised in a second commit); this after-task report;
the check-log entry appended to `docs/dev-log/check-log.md`. No file under `src/`, `test/`, or `docs/src/` was
touched, and #430's branch was never touched.

## 5. Checks Run

- `gh pr list -R itchyshin/GLLVModels.jl --state open --json number,title,headRefName,isDraft,mergeable,files` and
  `gh pr view <n> --json ...` for #429, #428, #439, #437, #433, #430, and the other open PRs whose files intersect
  `docs/src/` or `docs/make.jl` (#411, #410, #409, #399), to gather live PR state and status checks.
- `gh api repos/itchyshin/GLLVModels.jl/pulls/429 --jq '{merged,merge_commit_sha,base:.base.ref,state}'` returned
  `base: codex/reader-text-recovery-20260919`, the fact behind the headline finding.
- `git merge-tree --write-tree origin/main origin/codex/reader-text-recovery-20260919` exited 1 with 7
  `CONFLICT (content)` lines: `docs/src/choose-r-julia-bridge.md`, `docs/src/index.md`, `docs/src/quickstart.md`,
  `docs/src/response-families.md`, `docs/src/roadmap.md`, `docs/src/vignettes/community-abundance.md`,
  `docs/src/vignettes/phylogenetic-gllvm.md`. No branch or worktree was created.
- `git log --oneline origin/main -8 -- <file>` per conflicted file, to attribute each conflict to the most recent
  main-side commit; 5 files trace to `794f2ddbf` (#431), 2 to `326ba020c` (#434).
- `git merge-base --is-ancestor 51f63db5dd5481162d3d966637be854a57e26778 origin/main` returned false (exit 1).
- `git diff --check` ran clean (exit 0) before both the first triage commit and the revision commit.
- `~/shinichi-brain/tools/lane_lease.sh --claim/--release GLLVM.jl` for the triage path, then re-claimed with the
  after-task and check-log paths added for this report; both claims were GRANTED and both releases confirmed.
- `git for-each-ref` snapshots before and after each push, diffed with plain `diff`: only the handover branch's local
  and remote-tracking refs moved each time, plus one fetch-updated `refs/remotes/origin/gh-pages` from unrelated CI
  deploy activity picked up by an intervening `git fetch origin`.
- `python3 ~/shinichi-brain/tools/slop_check.py docs/dev-log/after-task/2026-09-21-reader-pr-triage.md` on this
  report; see the commit for its final clean result.

## 6. Tests of the Tests

Not applicable, and here is why: this is a documentation triage with no code, no numerical claim, and no behavior
change, so there is no executable assertion for a test to exercise. The closest equivalent was re-running every cited
command myself before writing the table cell it backs, rather than restating a remembered result, and having an
independent reviewer (the Fable pass) re-derive the same facts from the same primary sources without reading this
file's prose first. That review returned 0 BLOCKING items and reproduced the headline finding and all four spot-check
claims it tried.

## 7a. Issue Ledger

No new issue opened. This is a docs-only triage on the existing draft PR #444; it found a stale claim in a handover
document, not a code defect that needs a GitHub issue.

## 8. Consistency Audit

`docs/dev-log/handover/2026-09-21-claude-handover.md` states, in its PR table, "#429 ... merged at `51f63db5`; use as
base, do not recreate," and, in its Current state and boundaries section, "**Working:** #429's landing work is on
main." Both lines are wrong: #429's squash commit is not an ancestor of `origin/main`, and its base branch was #428's
own branch, not main. `git grep -n "#429"` on `origin/main` was run against `AGENTS.md` and
`docs/dev-log/coordination-board.md`; neither mentions #429 at all, and `HANDOVER.md` does not exist in this
repository, so the one handover file above is the only place carrying the wrong claim. The triage file's
"Recommended next" step 1 names that exact file and those exact lines, rather than a vague "update the board."

## 9. What Did Not Go Smoothly

Understanding why #428 showed `CONFLICTING` took more digging than expected. The obvious read (main already contains
#429, and #428 is a stale rebuild sitting on top of it) was wrong. Confirming the real mechanism, that PR #429's base
branch was #428's own branch rather than main, needed a REST API call (`gh api .../pulls/429`) because the GraphQL
`mergeCommit` field alone, as returned by `gh pr view --json`, does not show a PR's base branch and so does not make
this visible on its own.

## 10. Known Residuals

Handover steps 2 to 5 (question-first pages, a plain limit beside each action, the jargon sweep, the render-and-read
pass, and Rose's claim-and-boundary pass) remain undone. They are deferred until the maintainer picks #428's path,
since all five would edit the same `docs/src/` pages #428 currently holds contested. Two decisions are still owed
from the maintainer: which path to take for #428 (split the 13 clean files then reconcile the 7 contested pages by
hand, or a full owner rebase-and-reconcile of all 24 files in one pass), and whether the reader lane may edit
`src/*.jl` docstrings at all, which covers #439, #437, #433, and #428's own `src/formula.jl` change.

## 11. Team Learning

A PR's `merge_commit_sha` (or GraphQL `mergeCommit.oid`) field being present, together with `state: MERGED`, does not
mean that commit is reachable from the repository's default branch. A PR can be merged into another open PR's branch
instead of into `main`; only checking `base.ref` (via the REST API, since the GraphQL PR view used here does not
surface it) or running `git merge-base --is-ancestor` against the actual default branch catches this. Separately,
`git merge-tree --write-tree` is a safe way to measure a PR's real, current conflict against main without creating a
branch, a worktree, or touching any ref, which matters whenever a triage task needs a quantitative conflict count
without performing the rebase itself.

## 12. Cross-Product Coverage

This triage covers the classification of the five stacked reader PRs, the #428 conflict count with per-file
attribution, and the two ownership questions the maintainer needs to answer next. It does not cover any
documentation-page edit, any merge, any rebase, the Frozen R 0.7.0 oracle gap (recorded as advisory and unchanged),
GLLVModels' numerical parity, PROTECTED PR #430, or the sibling DRModels triage, which is a separate lane on PR #801.
