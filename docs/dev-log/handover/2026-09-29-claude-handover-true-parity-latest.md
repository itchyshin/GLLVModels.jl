# Handover: true-parity-latest lane, 2026-09-29

Goal: GLLVModels.jl at true parity with gllvmTMB main pinned at P1 = 9539352f6 (0.7.1); boundary D-295 (temporal and phylo latent inside; column grammar and spatial outside). Status: PAUSED at the maintainer's decisions. Every PR this lane had ready is merged; what remains needs a signature, a ruling, or new comparison work. Not done: only C7 is MET on origin/main.

## Where truth lives

- Kit `LOOP/lanes/true-parity-latest/` in the lane worktree `~/local-scratch/lanes/GLLVM.jl-true-parity-latest` (branch `claude/lane-true-parity-latest`): `GOAL.md`, `checkpoint.md`, `morning-report-2026-09-28.md` (per-PR entries and the full landed list), `packet-2-draft.md` (open decisions with replies to paste), `reviews/`, `scripts/` (refresh, cascade and merge-train scripts), `train-logs/`.
- Ledger and checker on main: `docs/dev-log/core070/true-parity-latest/GATES.md`, `tools/true_parity_check.mjs`, `tools/true_parity_assemble.py`, the assembled `scoreboard.md`, `case-map-assembled.json` and `reverse-gap.json` (#589).

## Landed since the 2026-09-28 handover (#592)

- A3 re-measurement stack, complete: #567 covariance, #569 postfit, #571 inference, #579 data and fit-input, #584 family (6e45f26b9), #586 aghq (f66cfab65), #587 isdm (307ca141a), #589 ledger assembly (24c9aaa1d).
- Engine and twins: #543 and #563 temporal slices 1 and 2, #546 iSDM build, #558 ISDM-PSI, #547 phylo latent, #556 anova twin, #557 zero-inflated twins, #561 checker hardening, #593 grouping-level receipts, #612 iSDM psi4 flake, #576 two-level ICC and #581 truncated NB2 guard (through #606).
- NB2 formula cell (Core070 FAMILY-05 formula interface): #625 records R's gradient instead of gating on it; #631 (auto d lane) fixes the fixture slice that made the file error before any fit; #632 accepts converged or boundary dispersion for the native, wide and long Julia fits, with R's convergence, the native gradient, the same-point delta and logLik agreement still gated. Run locally at the frozen oracle: 19 of 19.

## Gates on origin/main at d80822dc8

Measured with `node tools/true_parity_check.mjs <clause>` at 24c9aaa1d; #631 and #632 touch only parity tests, not the ledger.

| Clause | Result | Blocked on |
|---|---|---|
| C0 | NOT MET | default pin is P0; the switch to P1 is the maintainer's call |
| C1, C8 | MEASUREMENT FAILED | `case-map.json` is not on main; it arrives with #533 (signature) |
| C2, X2 | NOT MET, 52 of 297 rows done | #533, then the per-family rulings in Packet 2 |
| C3, C4, C5 | NOT MET, empty selection | no realistic-size, real-data or grouping-level rows yet; C4 and C5 also need the id-prefix ruling (the checker selects `RD-` and `GRP-`, case-map ids carry a family prefix) |
| C6 | NOT MET | 372 Julia-only exports await a decision each |
| C7 | MET | |

Negative controls (`node tools/test_true_parity_check.mjs`): all pass. Node must be on PATH, or every control fails falsely.

## Open, for the maintainer

1. #533 case-map classes: signature. #534 (carry-rule scan) is stacked on it.
2. Packet 2 rulings: C0 pin flip; nobs convention; the behavioural tier for routing and error-class rows; bridge case-id splits; the C4/C5 id matching; the ten feasible aghq Julia twins; the #546 mapping for isdm.
3. Campaigns for C3 to C5 (realistic size, real data, grouping levels). Totoro is back; each needs a time estimate first, and approval if over 3 hours.
4. Follow-ups found on the way: the fragile Woodbury solve remains in likelihood.jl, profile.jl, reml.jl and the sparse phylo path; `_mixed_laplace_mode` shows the same stall pattern #612 fixed; the Frozen R advisory job can end failed while reporting 0 failures (seen on #587).

## Lessons

- Stacked PRs: when the merge train deletes a merged branch, GitHub closes a PR stacked two levels up instead of retargeting it (#589 closed when #586's branch was deleted; restored the branch, reopened, retargeted to main). Retarget the next PR to main before merging its base.
- Trains time out on runner queues, not failures: check the heads are unchanged and restart. Every PR adds a check-log entry, so refresh after each merge.
- A test that slices another test file's source breaks when that file changes shape (#608 to #631).

## Resume

Read `GOAL.md`, `checkpoint.md` and `ultra-plan.md` in the kit, then the morning report, and continue from the maintainer's replies to the open items above.
