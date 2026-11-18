# P1 candidate gap to origin/main

Date: 2026-10-21

Worktree: `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`

Branch: `codex/p1-isdm-current-main-20261019`

## Method

Git commands only were used: `rev-parse`, `merge-base`, and `log`. No fetch was run because the `origin/main` ref was already present in this worktree. No source files and no receipts were edited. This note is the only file written.

## Commit positions

Candidate HEAD: `5cfc6754212588222779ccfd66ad7000375cf8d5`

`origin/main`: `1d8fc465d71ecfcf3d33e4381ee0ff64a65ec482`

Merge base: `882292ac6539d00bc0a08f5d0d06dc9f471251b3`

Ahead count: 18 commits in HEAD that are not in `origin/main`.

Behind count: 34 commits in `origin/main` that are not in HEAD.

The command used for the counts was `git rev-list --left-right --count HEAD...origin/main`, which returned `18 34`.

## Relation

HEAD and `origin/main` have diverged. HEAD is not simply ahead of `origin/main`, and it is not simply behind it. Both sides have commits that the other side does not have, counted from the merge base above.

These counts describe the commit graph only. They do not describe uncommitted files in the worktree.

## Gate state

The parity goal needs C0, C1, C2, C3, C4, C5, C6, C7, C8, and X2 MET on `origin/main`.

On the candidate at `5cfc6754212588222779ccfd66ad7000375cf8d5`, C0, C1, C5, C6, C7, and C8 are MET. C2, C3, C4, and X2 are not MET on that candidate.

That candidate state is not a state of `origin/main`. This note does not claim that `origin/main` is MET for any gate.

A merge is not authorized. No merge, push, commit, checkout, or staging was done for this note.
