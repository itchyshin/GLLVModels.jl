# GOAL — <one sentence: what "done" looks like>

**IMMUTABLE for this run.** Re-read this file at the top of EVERY arc, before anything else.

## Definition of done
- [ ] <the observable end state, not the activity>

## Invariants (never violate, even to finish faster)
- Never push, merge, or publish — those are HUMAN GATES. Land work on this branch only.
- Verification means reading the LOG and inspecting the ARTEFACT, never the exit code.
- A narrow or negative search is not proof. "No X exists" usually means the query missed X.
- Destructive or irreversible ⇒ STOP and surface, even if it feels urgent.
- Query the second brain first (`search_notes` with `project: "shinichi-brain"` first; add `search_all_projects: true` for other repos' docs; name no other project). In a code repo also run `python3 ~/shinichi-brain/tools/route.py <repo>` and read the LOAD-FIRST block. This worktree is one branch, not the memory.

## Pre-authorisation (copied from approved ultra-plan)
- Routine scoped edits, local commands, tests, builds, checkpoints, local commits, and listed checks: CONTINUE.
- Optional remote authority: <none | push named branch | create named draft PR; never merge or release>.
- Must stop: merge/release/public message or claim; credentials/security changes; destructive work outside this worktree; new compute/cost beyond the estimate; scope-changing evidence.

## Git and GitHub transport
- Use this repository's existing `origin` SSH remote for all Git operations (`git fetch`, `pull`, and an explicitly authorised `push`). Do not change remotes or keys.
- Do not open a browser or run `gh auth login`, device login, or token setup. GitHub API work (Actions dispatch, PR creation/merge, issue/comment writes) is separate from SSH and needs explicit task authority plus an already-working API credential. If it is not already available, report the limitation; do not request or start a login as a workaround.

## Out of scope (the fence — do NOT drift here)
- <...>
