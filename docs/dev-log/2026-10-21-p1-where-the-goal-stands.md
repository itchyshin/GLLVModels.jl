# Where the P1 parity goal stands

Date: 2026-10-21. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`.

The P1 parity goal is still open. This note records the measured state of the candidate. It does not claim the goal is met, and nothing here was merged.

## Where the candidate sits

HEAD is `5cfc6754212588222779ccfd66ad7000375cf8d5`. It has diverged from `origin/main`. The merge-base is `882292ac6539d00bc0a08f5d0d06dc9f471251b3`. HEAD is 18 commits ahead and 34 commits behind. Those positions are in `docs/dev-log/2026-10-21-p1-origin-main-gap.md`. No merge was done.

On this candidate, C0, C1, C5, C6, C7, and C8 are MET. C2 is 290/297. C3 is 6/8. C4 is 4/8. X2 is 304/317. The gap note records that MET list, and records that C2, C3, C4, and X2 are not MET on the candidate. The four fractions are in the morning record `docs/dev-log/recovery-checkpoints/2026-10-20-0508-cursor-morning.md`, written at this same HEAD. Those MET marks belong to the candidate. The closing paragraph records the boxes on `origin/main`.

The scoreboard on this candidate has 317 rows. 304 are bound and carry pass receipts. 13 are open, and those 13 are the closed list. See `docs/dev-log/2026-10-21-p1-scoreboard-mismatch.md`.

The scoreboard and gate file on `origin/main` at `1d8fc465d` were read. The measured boxes are in the closing paragraph.

## What is still open

Family receipt JSON files leave two rows open: `FAMILY-00-IDENTITY` and `ORDINAL-LOGIT-RSZ`. See `docs/dev-log/2026-10-21-p1-family-json-census.md`.

The identity checker reached the oracle library and refused. `CORE070_SOURCE_PIN.toml` lacks `gllvmTMB_version`. No pass receipt was written. See `docs/dev-log/2026-10-20-p1-overnight-identity-df8.md`.

The ordinal gap on traits (t01, t15) stays 0.000129891. The second read found no new bug (`docs/dev-log/2026-10-21-p1-ordinal-opus.md`). Spider found no new bug, and beetle stays blocked because R records `pdHess` as false (`docs/dev-log/2026-10-21-p1-spider-gpt.md`). Fungi found no new bug (`docs/dev-log/2026-10-21-p1-fungi-sonnet.md`). The iSDM headline read found no proven bug (`docs/dev-log/2026-10-21-p1-isdm-fable.md`).

The F-TEMPORAL-1FO tick stands (`docs/dev-log/2026-10-21-p1-temporal-tick-verdict.md`). Four earlier F boxes cite passing tests: F-ISDM-1FO-PREDICT, F-ORDINAL-LOGIT-1FO, F-PHYLO-LATENT-1FO, and F-ZI-1FO (`docs/dev-log/2026-10-21-p1-f-ticks-check.md`).

Nine C6 holds stay held (`docs/dev-log/2026-10-21-p1-c6-holds.md`).

No behaviour row outside the thirteen is open (`docs/dev-log/2026-10-21-p1-behaviour-open.md`).

Receipts, `GATES.md`, and source were left unchanged. Julia was not started. Nothing was committed, pushed, merged, or staged.

On `origin/main` at `1d8fc465d`, only C7 is ticked. C0, C1, C2, C3, C4, C5, C6, C8, and X2 are unticked with evidence pending. All five F boxes are unticked. No negative-control result is present for that commit. The candidate remains ahead 18 and behind 34. Nothing was merged. The parity goal remains open.
