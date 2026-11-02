# P1 five-row iSDM receipt candidate

Status: committed candidate 7670e106b passes the exact seven-row checkpoint;
full regression, completion review and landing remain pending.

## 1. Goal

Close exactly ISDM-COUNT, ISDM-EXTRA-SOURCE, ISDM-MISSING-IN-TRAIT,
ISDM-MISSING-SOURCE and ISDM-WRAPPER-LAW with retained public-call evidence.

## 2. Implemented

The iSDM map applies the behavior overlay only to five approved source IDs at
the measured R-side tier, current P1 reference and exact executable case IDs.
Five sidecars bind the inherited capture; the residual receipt is unchanged.
The checkpoint reader accepts nested paired-receipt hash groups while checking
every supplied group, signed ruling, exact row/case identity, P1 oracle source
and build, and bridge execution context. Its canonical caller supplies row IDs.

## 3a. Decisions and Rejected Alternatives

Reused helper input from 1a230eb51b4d1845d5dc7938890c909aac05f85d and the narrow
pattern reviewed at 40f403b6475bab47d29f6c8154dec7bf9d0bf5ae. The worker report
mistyped the former suffix; its transfer manifest records the correct source.
Excluded broad historical batch-writing changes. No new fit or R rerun.

## 4. Files Touched

The iSDM tool and focused tests, behavior tests, checkpoint reader and controls;
five sidecars and paired receipts; iSDM map, behavior table, assembled map and
scoreboard; this report and check log. No engine, Julia test, version, tolerance
or dependency declaration changed.

## 5. Checks Run

Seven runnable leaf gates passed parent re-verification: raw derivation;
36/36 behavior controls; iSDM ownership (20 receipts, 20 rows); behavior
regeneration (32 receipts, 24 classes, 63 entries); assembly (317 rows);
three focused scope tests; checkpoint provenance controls. Assembly changed
exactly five rows versus a444181ae, leaving the other 312 and C6 unchanged.
Committed-source checkpoint exited zero in 6.26 seconds: X2 304/317, C2 290/297.
Complete Node and Python controls then passed in 19.87 and 7.60 seconds on
Totoro. The valid-output smoke passed in 24.76 seconds. Julia/core, Aqua/JET
and documentation results remain pending; no pass is inferred from launch.

## 6. Tests of the Tests

Focused controls reject stale references, substituted cases, out-of-scope
sources and numeric rows. The six-source assertion failed before admission.
A valid nested-receipt regression failed before the reader repair and passed
afterward. Eleven added negative controls reject invalid nested provenance,
hashes, identities and execution evidence. Failed logs remain retained.

## 7a. Issue Ledger

No issue action was needed for this approved slice. No merge or classification
signature was inferred from implementation authority.

## 8. Consistency Audit

Owning maps, receipt derivation and assembled outputs agree. The bounded
checkpoint reviewer and separate five-row reviewer found no blocker.
Public APIs and numerical claims are unchanged. Whole-P1 review remains open.

## 9. What Did Not Go Smoothly

The checkpoint reader rejected valid nested hashes; the repair preserves its
existing refusal predicates. Three configuration changes are excluded and
preserved: .claude/settings.json, .cursor/hooks.json and .mcp.json. Their hashes
are retained in the local exclusion record. Initial report writing was denied
by the sandbox because this managed worktree lies outside the writable roots.

## 10. Known Residuals

Main 8ca3eb76e was freshly measured at 298/317. Local 304/317 belongs to
7670e106b, based on the older d9afa090c main. The four intervening main commits
include oracle-version and phylogenetic convergence changes; landing validation
must account for them. Thirteen scoreboard rows, five F gaps, regressions,
completion review and landing remain open. Inherited environment snapshots
are supplemental; canonical sidecars do not independently validate the lock.
Four checkpoint corruption controls reject invalid evidence. The original
expectation that label corruption would change summary counts was wrong:
the final exact-receipt check rejects it, with the frozen predicate separately
confirming the mismatch. The failed expectation is retained. Direct helper
calls without a row ID are less strict than the canonical caller,
which always supplies exact IDs. No full-P1 claim follows from this slice.

## 11. Team Learning

Require exact row IDs before regeneration, and test a valid evidence format
alongside its refusal cases.

## 12. Cross-Product Coverage

Covers five public-door behavior receipts. This does NOT cover numerical fit
parity, missing-data fits, aggregation, REML, penalty changes or main closure. Benchmarks: N/A; no hot-path change. JET, Allocs and Aqua were not run.
Rose verdict: PENDING full completion panel, regressions and landing.
