# Reader PR triage, 2026-09-21

Branch: `claude/gllvmodels-reader-arc-handover-20260921` (PR #444, draft). main = `402f2f8e3` at 2026-09-21 17:18:51 UTC.
Written by: Claude (S3 stacked-reader-PR triage auditor). No merges, no rebases, no pushes to any other branch.

## Headline finding, read this before the table

**PR #429 is not on `main`.** `gh api repos/itchyshin/GLLVModels.jl/pulls/429 --jq '{merged,merge_commit_sha,base:.base.ref,state}'`
returns `"base":"codex/reader-text-recovery-20260919"`, i.e. PR #429's base branch was **PR #428's own head branch**, not
`main`. Its squash commit `51f63db5dd5481162d3d966637be854a57e26778` is confirmed present only on
`origin/codex/reader-text-recovery-20260919` (`git branch -a --contains 51f63db5d...` returns exactly that one ref) and
`git merge-base --is-ancestor 51f63db5d... origin/main` returns `NO`. Direct proof: `docs/src/what-can-i-fit-today.md`
(a file #429 added) exists on `origin/codex/reader-text-recovery-20260919` but `git show origin/main:docs/src/what-can-i-fit-today.md`
fails with "path does not exist in 'origin/main'".

So #429 was squash-merged **into #428's branch**, not into main. The handover's line "#429 merged at `51f63db5`; use
as base, do not recreate" is true only in the narrow sense that #429's content is folded inside #428 and should not be
rebuilt from scratch; it is false if read as "#429's content is already on main and #428 is the leftover diff on top of
it." #428's branch now bundles #429's full landing rewrite plus #428's own reader-routes work, and none of it is on
main. #429's fate is therefore entirely tied to #428's fate; there is no independent "#429 on main" to build a fresh
lane from.

Meanwhile `main` kept moving on its own line: `#431` (`794f2ddbf`, 2026-09-20 15:15:40 -0600, a 61/59-line wording pass
touching `choose-r-julia-bridge.md`, `index.md`, `quickstart.md`, `response-families.md`, `roadmap.md`,
`studentt-parity.md`) and `#434` (`326ba020c`, 2026-09-20 16:57:48 -0600, "recover runnable community and phylogenetic
tutorials", touching the two vignette pages) both landed directly on main after #428's branch last touched those files
(#428's newest commit, the #429 squash, is timestamped 11:55:04 -0600, over three hours before #431). That is the
entire mechanism behind #428's `CONFLICTING` status: two independent wording passes on the same sentences, not a stale
merge marker.

## Table

| PR | Title | Branch | Draft | Mergeable | Checks | Files (reader-facing) | Overlaps | Classification | Evidence | Recommended action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #429 | docs: explain GLLVM before the first model | `codex/gllvmodels-reader-first-landing-20260920` | no (MERGED) | n/a, merged into #428's branch not main | Documenter SUCCESS at merge | 24 files, 1245+/853-: landing, quickstart, diagnostics, route guard, new `what-can-i-fit-today.md`, new `tools/check_reader_surface.py`, new `tools/tests/test_reader_surface.py`, `src/formula.jl` | Same 7 files that later collide inside #428 vs main, plus 17 that do not | DONE-but-mislabelled: merged into #428, not main; not independently landed | `gh api .../pulls/429` base=`codex/reader-text-recovery-20260919`; `git merge-base --is-ancestor 51f63db5d origin/main` = NO; `git show origin/main:docs/src/what-can-i-fit-today.md` fails | Correct the board/handover record: stop treating #429 as "on main, use as base". Its only path to main is through #428. |
| #428 | docs: rebuild reader-first learning routes | `codex/reader-text-recovery-20260919` | no | **CONFLICTING** (mergeStateStatus DIRTY) | 8/8 Julia shards SUCCESS, Documenter SUCCESS, Frozen R 0.7.0 family smoke FAILURE (advisory, expected per handover, not a repair signal) | 24 files, 1245+/853-. Reader-facing `docs/src/`: 17, of which 7 conflict (`choose-r-julia-bridge.md`, `index.md`, `quickstart.md`, `response-families.md`, `roadmap.md`, `vignettes/community-abundance.md`, `vignettes/phylogenetic-gllvm.md`) and 10 do not (`confidence-intervals.md`, `covariance-correlation.md`, `diagnostics.md`, `model.md`, `postfit-extractors.md`, `structured-dependence.md`, `structured-term-fitting.md`, `tutorial.md`, `what-can-i-fit-today.md`, plus `docs/make.jl`). Non-docs: `.github/workflows/Documenter.yml`, `.gitignore`, `docs/deploy.jl`, one after-task file, `src/formula.jl`, two new `tools/` files. | Conflicts exactly with #431 (5 files) and #434 (2 files), both already merged straight to main | **OWED, CONFLICTING (real, measured)** | `git merge-tree --write-tree origin/main origin/codex/reader-text-recovery-20260919` exits 1, 7 `CONFLICT (content)` lines (see Conflict probe below) | See "Recommended next" below: split off the 17 clean files first, then reconcile the 7 contested pages deliberately; do not force-merge, do not discard (that would also discard #429). |
| #439 | docs: describe bootstrap scope in reader language | `docs/bootstrap-sigma-reader-wording` | yes | MERGEABLE (mergeStateStatus UNSTABLE, draft-only, not a real conflict) | 8/8 Julia shards SUCCESS, Documenter SUCCESS, Frozen R advisory FAILURE | 1 file: `src/confint_derived_wald.jl`, docstring-only wording swap, 3+/4- | None with #428, #437, #433, or #429 (touches no `docs/src/` page at all) | OWED, source-edit ownership transfer needed | `gh pr diff 439` shows a pure prose edit inside an existing docstring, no logic change | Independent of #428's conflict. Needs an explicit owner sign-off for a docstring wording edit inside `src/` (handover: "source edits need owner transfer"), then can come out of draft and merge on green without waiting on #428. |
| #437 | docs: clarify derived loading interval scope | `codex/derived-ci-reader-cleanup` | yes | MERGEABLE (UNSTABLE, draft-only) | 8/8 Julia shards SUCCESS, Documenter SUCCESS, Frozen R advisory FAILURE | 2 files: `docs/src/derived-confidence-intervals.md` (not touched by #428) and `src/confint_derived.jl` docstring, matched wording, 21+/16- | None with #428's 7 conflicts; the one doc page it touches is outside #428's file list entirely | OWED, source-edit ownership transfer needed for the `.jl` docstring half | `gh pr diff 437` shows the `.md` page and the `.jl` docstring carrying the same corrected wording about R's confirmatory vs GLLVModels.jl's exploratory estimand | Independent of #428. Needs the same ownership sign-off as #439 for its `src/` half, then mergeable on its own schedule. |
| #433 | docs: clarify public post-fit docstrings | `codex/gllvmodels-public-docstrings-20260920` | yes | MERGEABLE (UNSTABLE, draft-only) | 8/8 Julia shards SUCCESS, Documenter SUCCESS, Frozen R advisory FAILURE | 2 files, both `src/`: `extractors.jl`, `postfit_tables.jl`, docstring-only, 17+/26- | None with #428 (zero `docs/src/` files touched) | OWED, explicitly named by the handover as needing owner transfer (100% `src/`, no paired doc page) | `gh pr diff 433` shows docstring-only edits, no `docs/src/` file in the PR's file list | Needs an explicit ownership decision (reader-docs lane vs engine/`src/` owner) before integration; no file overlap with #428, so it does not have to wait on #428's resolution. |
| #430 | speed78: phylo EM p^2.4 to p^0.5, grouped-route CHOLMOD reuse and confined warm start | `claude/lane-speed78-20260919` | yes | MERGEABLE | not checked (PROTECTED, out of scope) | 27 files, 7313+/98-. One incidental `docs/src/low-level-reference.md` line (registers one new internal function in the reference page) | No overlap with #428's 7 conflicted files | **PROTECTED**, not touched | Handover: "PROTECTED: #430 engine lane. Do not change likelihood code, API contracts, numerical claims, R oracle, or generated assets." | Leave untouched. Noted only because its one `docs/src/` line matched the gather step's `docs/src/` filter; it does not intersect #428's conflict set. |

Not part of the reader stack, noted only because the gather step's `docs/make.jl` filter matched them: `#411`, `#410`,
`#409`, `#399` (all `DRAFT: waits for paste ...` harness scaffolds from unrelated compute/engine lanes). Each shows the
identical single-line pre-rename diff on `docs/make.jl` (`github.com/itchyshin/GLLVM.jl` to `github.com/itchyshin/GLLVModels.jl`),
inherited from branching before the `#423` rename landed on main. They carry no reader-facing content and are out of
scope for this triage.

Stamp for every row above: `main=402f2f8e3 @ 2026-09-21 17:18:51 UTC`.

## Conflict probe: #428

```
$ git merge-tree --write-tree origin/main origin/codex/reader-text-recovery-20260919
(exit code 1)

Auto-merging docs/src/choose-r-julia-bridge.md
CONFLICT (content): Merge conflict in docs/src/choose-r-julia-bridge.md
Auto-merging docs/src/index.md
CONFLICT (content): Merge conflict in docs/src/index.md
Auto-merging docs/src/quickstart.md
CONFLICT (content): Merge conflict in docs/src/quickstart.md
Auto-merging docs/src/response-families.md
CONFLICT (content): Merge conflict in docs/src/response-families.md
Auto-merging docs/src/roadmap.md
CONFLICT (content): Merge conflict in docs/src/roadmap.md
Auto-merging docs/src/vignettes/community-abundance.md
CONFLICT (content): Merge conflict in docs/src/vignettes/community-abundance.md
Auto-merging docs/src/vignettes/phylogenetic-gllvm.md
CONFLICT (content): Merge conflict in docs/src/vignettes/phylogenetic-gllvm.md
```

No branch or worktree was created; `--write-tree` only wrote tree objects and reported conflicts. 7 of #428's 24 files
conflict; the other 17 (including the three brand-new files `what-can-i-fit-today.md`,
`tools/check_reader_surface.py`, `tools/tests/test_reader_surface.py`) apply cleanly against current main.

Per-file attribution, `git log --oneline origin/main -8 -- <file>`, latest main-side commit per conflicted file:

| File | Latest commit on main | PR |
| --- | --- | --- |
| `docs/src/choose-r-julia-bridge.md` | `794f2ddbf` docs: clarify reader-facing model routes | #431 |
| `docs/src/index.md` | `794f2ddbf` docs: clarify reader-facing model routes | #431 |
| `docs/src/quickstart.md` | `794f2ddbf` docs: clarify reader-facing model routes | #431 |
| `docs/src/response-families.md` | `794f2ddbf` docs: clarify reader-facing model routes | #431 |
| `docs/src/roadmap.md` | `794f2ddbf` docs: clarify reader-facing model routes | #431 |
| `docs/src/vignettes/community-abundance.md` | `326ba020c` docs: recover runnable community and phylogenetic tutorials | #434 |
| `docs/src/vignettes/phylogenetic-gllvm.md` | `326ba020c` docs: recover runnable community and phylogenetic tutorials | #434 |

None of the 7 conflicts trace to the #429 squash commit itself (that commit is not on main at all, see the headline
finding above); all seven trace cleanly to two later, independently merged main PRs. #428's branch has not been
rebased since before either of them landed.

The advisory Frozen R 0.7.0 family-smoke failure appears on #428 (and on #439/#437/#433, which inherit it from the
current CI baseline). Per the handover this is a known, pre-existing oracle gap and is not evidence against any of
this wording work; it is recorded here for completeness, not as a reason to weaken any page's claims.

## Recommended next

1. Correct the record first. Update the board/handover language so #429 is described as "squashed into #428's branch,
   not on main" rather than "merged, use as base." Anyone opening a fresh lane "on top of #429" right now would be
   building on nothing, since #429 has no existence outside #428.
2. Get the maintainer's decision on #428's path. The realistic options, in order of risk:
   - **Split (lowest risk, recommended first step).** Land the 17 non-conflicting files as their own PR (this banks
     #429's three new files and #428's non-contested rewrites, all already CI-green) and shrinks #428 to exactly the 7
     contested pages.
   - **Owner reconciliation of the remaining 7 (recommended second step, not a mechanical rebase).** #431 and #434 made
     independent, already-merged wording fixes to the same sentences #428 rewrites more broadly. A plain `git rebase`
     would produce syntactically resolved but not necessarily correct prose; the owner (Codex, on
     `codex/reader-text-recovery-20260919`) needs to read both versions of each of the 7 pages and decide which wording
     wins per paragraph, then re-run the reader-surface checks.
   - **Close in favour of a fresh lane is not well-defined here**, because #429's content is not separately recoverable
     from main; closing #428 outright would discard #429 too. Not recommended.
3. In parallel, with no dependency on #428's outcome: get an ownership decision for #439, #437, and #433's `src/*.jl`
   docstring edits (none of the three overlaps any of #428's 7 conflicted files or #429's content). Once an owner signs
   off on editing reader-facing docstrings inside engine-owned `src/` files, each can come out of draft and merge on
   green independently, on its own schedule.
4. Only after #428's fate is settled should the remaining handover steps (question-first pages, a plain limit beside
   each action, the jargon sweep, the render-and-read pass, Rose's claim-and-boundary pass) resume, since they would
   edit the same `docs/src/` files #428 currently holds contested.

## Not done here

Handover steps 2 to 5 (public pages starting from the data and scientific question with a runnable Gaussian example;
a plain limit beside each action; the jargon and internal-process sweep; the render-and-read novice pass and Rose's
claim-and-boundary pass) are deferred. They are not attempted in this triage because they would write prose into the
same `docs/src/` pages that #428 holds in `CONFLICTING` state, and this session's mandate is triage only: classify,
measure the conflict, and hand the maintainer a decision, not edit any documentation page. No merge was possible today
either way: #428 is `CONFLICTING` against current main and #439/#437/#433 are still drafts.
