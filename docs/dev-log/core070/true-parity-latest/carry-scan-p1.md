# P1 carry-rule scan (CARRY_VERIFY)

Generated 2026-09-27T18:56:00Z against `origin/main`, P0 = `b4d5fee64def88bc768dda1f1f77c29b295edd86`, P1 = `9539352f66f2db2cc26b1c393e67212a359b60c9`. Source: `docs/dev-log/core070/required-source-case-map.json` (769 rows).

## Counts by status

| Status | Count |
| --- | --- |
| DANGLING | 278 |
| OUT_OF_SCOPE_NOT_REQUIRED | 272 |
| NOT_BOUND_AT_P0 | 189 |
| PARTIAL_STALE_AT_P1 | 21 |
| NO_R_PINS | 5 |
| RETIRED | 4 |

## Counts by status x family (source_id group)

| Family | DANGLING | NOT_BOUND_AT_P0 | NO_R_PINS | OUT_OF_SCOPE_NOT_REQUIRED | PARTIAL_STALE_AT_P1 | RETIRED |
| --- | --- | --- | --- | --- | --- | --- |
| aghq | 7 | 0 | 0 | 18 | 14 | 0 |
| covariance | 10 | 22 | 0 | 56 | 7 | 0 |
| data | 28 | 0 | 0 | 28 | 0 | 0 |
| family | 16 | 1 | 5 | 47 | 0 | 0 |
| fit-input | 6 | 5 | 0 | 3 | 0 | 0 |
| inference | 63 | 1 | 0 | 34 | 0 | 0 |
| isdm | 20 | 0 | 0 | 17 | 0 | 0 |
| masks-known | 9 | 0 | 0 | 8 | 0 | 0 |
| namespace | 69 | 91 | 0 | 53 | 0 | 2 |
| postfit | 34 | 64 | 0 | 0 | 0 | 2 |
| postfit-policy | 16 | 5 | 0 | 8 | 0 | 0 |

## 5 largest PARTIAL_STALE_AT_P1 groups (by family)

- **aghq**: 14 rows
- **covariance**: 7 rows

## DANGLING rows (receipts that do not resolve on origin/main)

278 rows. The 20 `isdm/*` rows in this list cite receipts under `.unlazy/core070-aghq/...` (via `evidence.receipts` directly, or via their case plan's trace through `docs/dev-log/core070/isdm-admission-evidence.json`'s own `runs[].receipt`, e.g. `.unlazy/core070-aghq/isdm-admission/attempt2/receipt.json`); none of that resolves on `origin/main`. The iSDM port (spec PR #525) re-measures these natively; no receipt is invented here.

isdm/* DANGLING rows (20): isdm/ISDM-ALIASED, isdm/ISDM-ALIGN, isdm/ISDM-COUNT, isdm/ISDM-EXTRA-SOURCE, isdm/ISDM-LEGACY, isdm/ISDM-MASKED-ARM, isdm/ISDM-MASKED-COLUMNS, isdm/ISDM-MISSING-IN-TRAIT, isdm/ISDM-MISSING-SOURCE, isdm/ISDM-MIXED, isdm/ISDM-NO-OFFSET, isdm/ISDM-NO-TRAITS, isdm/ISDM-SUPPORT, isdm/ISDM-THREE, isdm/ISDM-UNBALANCED, isdm/ISDM-WITHIN-TRAIT-ADMIT, isdm/ISDM-WRAPPER-LAW, isdm/ISDM-WRONG-ID, isdm/ISDM-WRONG-LINK, isdm/ISDM-ZERO-ORDINARY

## RETIRED rows

- `namespace/export/.proportions_bootstrap_ci`
- `namespace/export/.proportions_wald_ci`
- `postfit/POSTFIT-SURFACE-.proportions_bootstrap_ci`
- `postfit/POSTFIT-SURFACE-.proportions_wald_ci`

## CARRIED rows (added to the P1 case-map)

None. Every required, previously-bound P0 row that pins a gllvmTMB source file pins at least one file that changed between P0 and P1 (62 of 117 R/src files changed; the two most-shared files, `R/fit-multi.R` and `R/gllvmTMB.R`, both changed) -- matching the contract-level preliminary scan's finding that all 29 R-pinning contracts touch a changed file. Everything that could carry did not; this is the honest count, not an empty search.

## NOT_BOUND_AT_P0 / OUT_OF_SCOPE_NOT_REQUIRED

NOT_BOUND_AT_P0 (required rows already BLOCKED_*/PARTIAL_* at P0 -- never counted, so the carry question does not apply): 189. OUT_OF_SCOPE_NOT_REQUIRED (P0 `rejected`/`intentionally_excluded` rows, never part of C1/C8 accounting): 272.

## What this scan does not cover

- No R or Julia fitting: everything above is a file-hash comparison, not a re-run of any model, control, or paired-control case.
- `NO_R_PINS` rows are not evaluated for staleness at all -- their P0 evidence never cited a gllvmTMB source file, so byte-identity has nothing to check; they need their own re-measurement plan, not a carry decision.
- Name-twin detection is out of scope (GATES.md gap 6): a row can be `CARRIED` here and still be wrong if it was misclassified at P0.
- `PARTIAL_STALE_AT_P1` rows are listed, not re-measured; WS0d/arc A3 owns re-running them at P1.

