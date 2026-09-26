# Rose audit: gate-tier scoreboard (on `main`) + true-parity decision packet (PR #497)

Scope: `docs/dev-log/core070/true-parity-gate-tier-scoreboard-2026-09-25.md` as it stands on
`origin/main` (merged via #487, not itself under review here — audited as context/evidence for
PR #497 and PR #496), and the decision packet in PR #497. Default posture: NOT READY until
checked; findings below are what changed that.

## Scoreboard: structural checks

- **Row count and category breakdown match the doc's own headers.** `## A: ... (15 rows)`,
  `## B: ... (4 rows)`, `## C: ... (5 rows)`, `## D: ... (8 rows)` — counting actual
  `| A1 ... | A2 ... |` etc. table rows gives 15+4+5+8 = **32**, matching both the section
  headers and the "Count summary" table's `**Total** | **32**`.
- **Count-summary table is internally consistent with the row-level statuses.** An independent
  per-row tally of the `Status` column (`awk -F'|'` on every `| X# ...` row) gives **19
  PARTIAL, 13 OPEN, 0 EVIDENCED, 0 DISPOSITION-SIGNED** — exactly matching the "Count summary"
  table's own numbers and the doc's prose ("Zero rows are EVIDENCED and zero are
  DISPOSITION-SIGNED, so zero of 32 rows are promoted").
- **D8's live tool-run claim reproduces.** D8 says `python3 tools/parity_ledger.py` on `main`
  reports `FORWARD=62 REVERSE=91`. Ran it on `origin/main` (current tip, `d89179d41`): exact
  match, `FORWARD=62 REVERSE=91`. This is also the number PR #496 turns into `REVERSE=0` via
  classification — the scoreboard and PR #496 are consistent with each other.

## Spot-checked rows (11 of 32 — every PARTIAL that cites a PASS receipt, plus 6 more)

All five PARTIAL rows whose receipt text contains the word "PASS" were checked (task
instruction: check every one of these), plus six additional rows chosen for citation density
(quoted files, JSON fields, `gh` states):

| Row | Claim checked | Result |
|---|---|---|
| A2 | 4 named `t4-p6-gaussian-*-receipt-2026-09-05.json` files exist, all `result: PASS`, `claim_boundary` field reads the quoted "NOT true-parity or gate-tier promotion" | Matches exactly, all 4 files, all fields |
| A5 | Same pattern, `t4-p6-poisson-*` | Matches exactly, all 4 files |
| A7 | `second-order-matched-pilot-batch1-20260905.md` row `binomial_logit \| **pass** \| 4.2e-8 \| 8.1e-8` | Matches exactly (se_max_rel 4.2e-8 quoted correctly) |
| A9 | `theta-map-disposition-2026-09-05.md` exists; PR #483 "open, needs maintainer review"; issues #480/#482 | File exists. Issues #480 (closed) and #482 exist. **PR #483 is now MERGED**, not open (`gh pr view 483` → `state: MERGED`) — see Finding 1 below |
| A11 | `t4-p6-nb2-*-receipt` (4 files, PASS); PR #478 "merged 2026-09-25"; issue #477 | All 4 receipts PASS. PR #478's merge commit is `d9bc77412` itself — the scoreboard's own measurement ref, so this citation is exactly self-consistent. Issue #477 exists |
| A12 | `latent-bare-model-evidence.json` four-route agreement; `.unlazy/` git-ignored per `.gitignore:19` | JSON status `COV_ORD_LATENT_BARE_THREE_ROUTE_PASS`; `.gitignore:19` is literally `.unlazy/` |
| A14 | `required-source-case-map.json` row `covariance/COV-PHYLO-LATENT`: zero executable cases, 3 named planned cases all `PREPARED_REFERENCE_NUMERICS_UNPAID` | Matches exactly, including all 3 case IDs and the status string |
| B1 | `docs/src/grouped-models.md` quote "agreement with R and recovery of known simulated parameters have not yet been established" | Present verbatim (line 4–5, wrapped) |
| C1 | `acc-bridge-urbanisation-receipt-2026-09-05.json`: PASS, logLik agreement ~1.6e-7, `max_gradient` not surfaced | `status: PASS`, `loglik_abs_diff: 1.628e-07`, `max_gradient: "NA"` — matches exactly |
| D6 | `fit-input-contract.md` quotes "prepared-input evidence, not fitted parity" and "...UNRESOLVED/UNPAID" | Both present verbatim |
| D8 | Live tool run | Reproduced exactly (see above) |

**11/11 rows check out against their cited receipts, with one staleness caveat (A9, below).**
No fabricated citation of the kind found in PR #496 (the wrong-issue-number problem) appears
anywhere in the scoreboard.

## Findings

1. **Should-fix — A9's PR-state citation is now stale.** A9 reads: *"the default Beta grouped
   fitter this receipt used is under active correction: PR #483 (open, needs maintainer
   review)."* That was accurate when the scoreboard was measured, at `origin/main @
   d9bc77412` (2026-09-25) — `git log d9bc77412..d4da31544` shows PR #483's merge commit
   (`72cc7458c`) landing 12 commits *after* the scoreboard's own ref, so at measurement time it
   really was still open. But `gh pr view 483` now reports `state: MERGED`. The doc is honest
   about its own ref (this is not a false claim, it's dated evidence), but the scoreboard has
   been sitting on `main` since #487 merged and is now 24 commits stale. A9 (and by extension
   any row whose disposition depends on "is PR #483 in yet") should be re-measured before it's
   read as current.
2. **General staleness, not specific to one row.** The scoreboard's own measurement commit
   `d9bc77412` sits 24 commits behind current `origin/main` (`git log d9bc77412..origin/main
   | wc -l` = 24), spanning #481, #483, #492, #494, #488, #495, #487 itself and others. I did
   not find a second row (beyond A9) whose specific prose is contradicted by one of those 24
   commits — the intervening work (Gamma/Beta mode-search fixes, a docstring reword, the
   cross-audit-triage and capability-status docs merges) doesn't obviously touch the other 31
   rows' cited evidence — but this was not exhaustively re-checked row-by-row against every one
   of the 24 commits, only against the specific files/receipts each row names. A full refresh
   (re-running whatever produced this table against current `main`) would be the clean fix
   rather than patching row text piecemeal.

## Relation to PR #497
PR #497's items 35–36 (the #129/#131 live divergences) are pinned to the same stale ref
(`d9bc77412`) as this scoreboard, for the same underlying reason: PR #495 (the source of both
the scoreboard and the packet's new items) was triaged against that commit. This is one
staleness issue with two symptoms, not two independent problems — see PR #497's review for the
packet-side detail.

## Bottom line
The scoreboard is internally consistent (row counts, category totals, and the PARTIAL/OPEN
tally all agree with each other) and every spot-checked receipt resolves to exactly what its
row claims — a materially cleaner audit than PR #496's reverse-gap table, which had one
fabricated issue citation. The one substantive finding is staleness: the scoreboard is pinned
honestly to `d9bc77412` but that ref is now 24 commits behind `main`, and at least one row (A9)
names a PR state that has since changed. Not a citation violation, but worth a refresh before
the maintainer treats any "still open" / "not yet merged" language in it as current.
