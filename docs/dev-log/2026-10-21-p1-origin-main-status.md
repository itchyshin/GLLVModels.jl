# P1 status on origin/main (read 2026-10-21)

Verdict: NOT_MET_ON_ORIGIN_MAIN. The parity goal is unmet on this ref.

## Read

`origin/main` was already local: `1d8fc465d71ecfcf3d33e4381ee0ff64a65ec482`, subject `Merge pull request #868 from itchyshin/cursor/false-converged-754-753-781-78dc`, committed 2026-10-21 06:42:17 +0800. No fetch, no checkout, no Julia, no checker run.

Both files exist and were read with `git show origin/main:path`:

- `docs/dev-log/core070/true-parity-latest/GATES.md` (1238 lines)
- `docs/dev-log/core070/true-parity-latest/scoreboard.md` (458 lines)

Last commit touching either file: `b8983cc6e0657c025e87223a388d81e04cfd45d0`, 2026-10-17 23:16:42 -0600, subject `Integrate FAMILY-11 boundary receipt checkpoint`.

Worktree HEAD is `5cfc6754212588222779ccfd66ad7000375cf8d5` on `codex/p1-isdm-current-main-20261019`. Merge-base with `origin/main` is `882292ac6539d00bc0a08f5d0d06dc9f471251b3`. `git rev-list --left-right --count origin/main...HEAD` prints `34 18` (34 commits on origin/main that this HEAD lacks, 18 commits on this HEAD that origin/main lacks).

## C0-C8 and X2 lines in GATES.md

Each box below is the checkbox line. The evidence line is the `EVIDENCE:` line under that box. `EXPECT:` names a `*_MET` token for every clause. A recorded `*_MET` result is present only for C7.

| Clause | Line | Box | Evidence line |
|---|---|---|---|
| C0 | 1032 | `[ ]` | 1047: `EVIDENCE: pending.` The same paragraph records a 2026-09-27 run on branch `claude/true-parity-p1-pin` that printed `C0_NOT_MET` with `default_pin=P0`. |
| C1 | 1056 | `[ ]` | 1063: `EVIDENCE: pending (P1 case-map not yet populated; A0c/A0d)` |
| C2 | 1065 | `[ ]` | 1072: `EVIDENCE: pending` |
| C3 | 1074 | `[ ]` | 1078: `EVIDENCE: pending` |
| C4 | 1080 | `[ ]` | 1086: `EVIDENCE: pending` |
| C5 | 1088 | `[ ]` | 1094: `EVIDENCE: pending` |
| C6 | 1096 | `[ ]` | 1106: `EVIDENCE: pending` |
| C7 | 1108 | `[x]` | 1113-1115: verified 2026-09-27 against `origin/main` (`1385b0490`); the evidence text includes `C7_MET`. |
| C8 | 1117 | `[ ]` | 1125: `EVIDENCE: pending (A0c classifies the +25 exports / +11 S3 methods since 0.7.0; Shinichi signs in that PR, Packet 1 row 3).` |
| X2 | 1131 | `[ ]` | 1137: `EVIDENCE: pending` |

The scoreboard has no C0, C1, C2, C3, C4, C5, C6, C7, C8, or X2 verdict line and no clause fraction. It names C1 and C3 inside the header and inside some row ids.

## Scoreboard tally that is printed

Line 41, family `` `all` ``:

| EVIDENCED | EVIDENCED-BEHAVIOURAL | DISPOSITION-SIGNED | REGISTRATION-ONLY | PARTIAL | NEEDS-SURFACE | NON-DISCRIMINATING | FAIL | NOT-MEASURED | total |
|---|---|---|---|---|---|---|---|---|---|
| 206 | 57 | 35 | 3 | 3 | 5 | 1 | 6 | 1 | 317 |

X2's checkbox text says every scoreboard row is `EVIDENCED`, `EVIDENCED-BEHAVIOURAL`, or `DISPOSITION-SIGNED`. The printed `all` row has non-zero counts in the other six status columns. This note does not convert that row into a pass fraction. The file does not print one.

## F-capability boxes

Present in GATES.md under the heading `F-gates (per-capability, informational until the scoreboard exists)`, lines 1139-1153. All five boxes are unticked:

- line 1144: `[ ]` F-ISDM-1FO-PREDICT
- line 1145: `[ ]` F-TEMPORAL-1FO
- line 1146: `[ ]` F-PHYLO-LATENT-1FO
- line 1148: `[ ]` F-ORDINAL-LOGIT-1FO
- line 1149: `[ ]` F-ZI-1FO

Line 1153: `EVIDENCE: pending`. The scoreboard file is present on this commit. The five boxes stay unticked.

## Negative-control result

Absent for `origin/main` at `1d8fc465d`.

GATES.md describes checker negative controls (fixture names and `tools/test_true_parity_check.mjs`) and keeps older check notes, including the 2026-09-27 `C0_NOT_MET` on another branch and the C7 `C7_MET` line. The scoreboard has no negative-control result line.
