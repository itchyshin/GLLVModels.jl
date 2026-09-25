# Handover: true-parity-finish lane (2026-09-25)

Stopped at the maintainer's request. True parity is NOT reached and was never claimed. Base: `main` at
d9bc77412 (CI green). Lane kit: `LOOP/lanes/true-parity-finish-20260925/` on branch
`claude/lane-true-parity-finish-20260925` (pushed).

## Where parity stands

0 of 32 gate-tier rows are evidenced: 19 partial, 13 open (scoreboard, #487). The decisions only the
maintainer can make are in `docs/dev-log/owed/2026-09-25-true-parity-decision-packet.md` on the lane
branch, each with a reply to paste. Start with B-06 (which R version), the NATIVE-06/10/12 contract
revision, and the T9 ledger sign-off.

## PRs opened this session (none merged by the lane)

| PR | What | State |
|---|---|---|
| #492 | Docstring that Julia 1.10 parsed as a link; the 1.10 docs build failed | merge-when-green gate running (maintainer asked to merge) |
| #487 | Gate-tier scoreboard (T9 draft) | draft |
| #488 | capability-status join fixes; CLOSURE is now an honest FAIL with 2 proposed dispositions | draft |
| #489 | Parity page and stale-document status blocks | draft |
| #490 | R-library guard for every tool; 2 receipts with unknown provenance | draft |
| #491 | Realistic-size Binomial-logit cell on Totoro: passes | draft; review: the simulated intercept was changed after a separated first run whose log was not kept |
| #493 | Truncated-NB2 second-order cell now pairs per-trait; Wald check FAIL to PASS | draft; review: it adds `src/` Wald intervals for the per-trait fitter |
| #494 | Covariate kernel fix (Fixes #486); Binomial 2 of 8 fits now honestly not converged | draft |
| #495 | Cross-audit triage: 9 closed with receipts, 9 live divergences, 11 need a decision | draft |
| #481, #483 | Gamma and Beta fixes from 2026-09-24 | open, waiting for review |

## Issues filed

#484 (two-part inner search), #485 (converged on a zero-length step), #486 (covariate kernel).
Receipts: `docs/dev-log/core070/class-audit-20260924/` on the lane branch.

## CARRIED-OVER (stopped mid-slice; nothing reviewed)

| Branch | State | Resume |
|---|---|---|
| `claude/twopart-mode-search-484` | test committed (fails on main); kernel fix is a WIP commit, not verified | re-run `test/test_twopart_mode_search.jl` and the two-part test files, then the fail-without-fix checks; open a PR "Fixes #484" |
| `claude/reverse-gap-classes-20260925` | `tools/parity_ledger.py` classes as a WIP commit; the doc is not written | re-run the tool, confirm unclassified REVERSE = 0, write `docs/dev-log/core070/reverse-gap-classes-2026-09-25.md` with the D8 text |
| #485 central verdict draft | not started (the branch has no commits) | brief in the lane `arcs.md`, row S8c |
| Sibling screen | not run | row S7: mixed, beta-binomial, NB1, GP1, COM-Poisson, ordered beta, Student-t, ordinal |

## Not done

- No adversarial review of #488 to #495 yet, and no Rose audit of the scoreboard or packet.
- The packet does not yet list the 11 NEEDS-DECISION issues from #495, or the live divergences #129 and #131, which gate-tier extractors depend on.
- Beta realistic-size cell waits for #483.
- No Totoro campaigns over 30 minutes were started.
