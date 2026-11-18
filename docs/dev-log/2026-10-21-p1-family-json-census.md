# P1 family receipt JSON census, 2026-10-21

Read-only census of every JSON receipt under
`docs/dev-log/core070/true-parity-latest/receipts/family/`
in worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`.
No receipt, GATES.md, or source file was read for mutation and none was changed.
No Julia was started. This note is the only file written.

## What was counted

58 JSON files in the tree. 36 of them carry a verdict field:

- `campaign/`: 5 receipts (BINOMIAL-LOGIT-RSZ, GAUSSIAN-IDENTITY-RSZ, NB2-LOG-RSZ, ORDINAL-LOGIT-RSZ, POISSON-LOG-RSZ).
- `cases/`: 30 receipts (CORE070-FAMILY-*, CORE070-A6-*, NATIVE-*, BETA-ALIAS).
- `first-seven-boundary/`: 1 receipt (CORE070-FAMILY-11-LOG-PUBLIC-R-BRIDGE).

The other 22 files carry no verdict: 14 `runparity-*/run-commit.json` commit stubs, plus the
`a6-p1`, `bridge-p1`, and `family-links-p1` support files. The support files that do carry a
status field read PASS: `family-links-p1/receipt.json` (status PASS, 2 positive cases plus 4
negative controls all behaving), `family-links-p1/julia-results.json` (status PASS), and the
four case ids inside `bridge-p1/results.json` (families 02, 05, 07, beta alias) each have
their own PASS case receipt under `cases/`.

30 of the 36 verdict-bearing receipts read PASS. The 6 that do not:

| Receipt | Verdict | Row status quoted from the receipt |
| --- | --- | --- |
| `campaign/ORDINAL-LOGIT-RSZ.json` | FAIL | "does not bind: LLt: abs difference 0.000129891 > tolerance 0.0001" |
| `cases/CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL.json` | FAIL | no row_status field; cell_status "failed", run_status "failed" |
| `cases/CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE.json` | FAIL | no row_status field; cell_status "failed", run_status "failed" |
| `cases/CORE070-FAMILY-00-IDENTITY-PUBLIC-R-BRIDGE.json` | NOT_EXECUTED | no row_status field; deliberate skip, not in the frozen three-case bridge sub-contract (02, 05, 07 only) |
| `cases/CORE070-FAMILY-11-LOG-PUBLIC-R-BRIDGE.json` | NOT_EXECUTED | no row_status field; same deliberate-skip reason as above |
| `first-seven-boundary/CORE070-FAMILY-11-LOG-PUBLIC-R-BRIDGE.json` | R_BOUNDARY_UNCHANGED | no row_status field; boundary capture, both pinned bridge routes refuse out-of-scope families at GJL-GATE-FAMILY |

## Mapping the non-PASS receipts to rows

Two of the six sit on the known open rows:

- `family/ORDINAL-LOGIT-RSZ`: campaign receipt FAIL. The scoreboard lists this row as FAIL,
  evidence tier `numeric_fail`. The failing leg in the receipt's pass rule is
  `every_quantity_within_tolerance: false`; the row status names the LLt product as the
  quantity that missed, at 0.000129891 against the 1e-4 tolerance. Every other leg (both
  engines converged, same data bytes, deparse-identical P1 entry points, R 0.7.1 from the
  pinned library) is true.
- `family/FAMILY-00-IDENTITY`: native-model and formula-interface case receipts FAIL (30
  passed, 1 failed assertion each, shared fixture-group counts), and the bridge case is
  NOT_EXECUTED. Scoreboard lists the row PARTIAL, evidence tier
  `partial_case_not_executed`.

The other four non-PASS receipts do not add an open row:

- The three FAMILY-00 receipts above all belong to the FAMILY-00-IDENTITY row, so they
  introduce nothing new.
- The two FAMILY-11 receipts (NOT_EXECUTED case receipt, R_BOUNDARY_UNCHANGED boundary
  receipt) belong to `family/FAMILY-11-LOG`. The scoreboard records that row as EVIDENCED,
  evidence tier `numeric`, bound on its native (NATIVE-12-TRUNCATED-NB2) and
  formula-interface case receipts, both PASS. GATES.md section N1 rules the bridge case a
  non-binding boundary context: a `not_executed` + `NOT_EXECUTED` receipt is refused as
  binding evidence by design, and the `R_BOUNDARY_UNCHANGED` probe records the pinned R
  bridge refusing unsupported families before any Julia call. Its verdict is not a pass
  value by design, so it is recorded, not compared. These two receipts are therefore a
  recorded disposition on a row the scoreboard already marks EVIDENCED.

## Census verdict

ONLY_TWO_OPEN. The open family rows are exactly the two known ones:

1. `FAMILY-00-IDENTITY`
2. `ORDINAL-LOGIT-RSZ`

The only non-PASS receipts outside those two rows are the FAMILY-11-LOG bridge pair named
above (verdict NOT_EXECUTED, and verdict R_BOUNDARY_UNCHANGED, both with no row_status
field, both quoted in the table). They do not open a row because family/FAMILY-11-LOG is
recorded as EVIDENCED in the scoreboard and the N1 rule sets the bridge case aside as
non-binding boundary context. If the parent wants to treat a NOT_EXECUTED bridge case as
open regardless of the N1 disposition, that choice would go beyond what these receipts
themselves record.