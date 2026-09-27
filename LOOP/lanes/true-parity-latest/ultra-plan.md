# Plan: true parity of GLLVModels.jl with the latest gllvmTMB (pinned, re-pinned at milestones)

```
🎯 GOAL
Solo platform: Claude Code (a new session opened in the GLLVM.jl folder; Opus 5.5 orchestrating;
  Fable for load-bearing review). This session only finishes today's queue and hands over the kit.
Deliverable: GLLVModels.jl at true parity with gllvmTMB pinned at P1 = 9539352f6 (0.7.1 candidate),
  inside a claim boundary you sign: every P1 capability in the boundary has a scoreboard row with a
  receipt made at P1 (or a carried receipt whose source files are byte-identical at P1); everything
  outside it closes by a signed disposition. Then re-pin to P2 and repeat for what is new.
HEADLINE: integrated SDM through R's public door (gllvmTMB(..., family = isdm_sources(...))), non-spatial,
  Laplace, first order plus predict, scoped exactly as R's own ledger claims it.
IN PARALLEL: additive re-pin (WS0) · small twins · silent-failure backlog · decision packets
DEFER: spatial iSDM (after WS8); anything you place outside the P1 boundary; releases, tags, Project.toml;
  select_lv (auto-d lane builds it; you sign its row).
DISCIPLINE: verify = twin test against R at P1, red then green, Fable review, whole-fit check · compute =
  Mac 4 cores per lane; Totoro <=30 min once the socket is back; DRAC only with an estimate first ·
  closure = tracked ledger --reverify all met + handover on main
```

## Context

Shinichi, 2026-09-27: parity with the latest gllvmTMB, "including integrated JSDM ... and also temporal stuff". His answers today: pin now to 9539352f6 and re-pin at milestones; integrated SDM first; port R's semantics and keep Julia's own designs as documented extras. This supersedes T2 (stay frozen at 0.7.0).

Measured today: the 0.7.0 ledger reads 1 of 10 gates (C7); 0 of 32 gate-tier rows done; 191 of 497 required rows unsigned at 0.7.0, and about 211 once the 20 isdm rows whose receipts never existed are corrected (finding below). Since 0.7.0 gllvmTMB went from 160 to 183 exports (+25, 2 removed) and 54 to 65 S3 methods; 62 R/src files changed and 12 were added (`R/fit-multi.R` 9620 to 11521 lines; `src/gllvmTMB.cpp` +824). `spatial_dep`, `meta`, `multinomial`, `phylo_latent` and `isdm_source` are 0.7.0-era gaps, not new ones.

A Fable review of the first draft (2026-09-27) found it not ready; its ten findings are folded in below.

## Prior-work sweep receipt

- **Repo git state** → `git worktree list`, open PRs, `lane_preflight.sh GLLVM.jl` → 10 other Claude lanes; my work: #512 and #516 (open drafts, CI running, #516 approved), #514 (open, rework paused, holds a lease on `src/families/mixed.jl`), issue #515 (builder paused, lease on `src/families/beta_binomial.jl`); resume notes at `~/.claude/plans/hidden-meandering-newell-agent-a9af17eb0b86e402c.md` and `...-agent-ae8bcbcfc2a90df28.md` → resume, do not rebuild.
- **Twin repo** → NAMESPACE diff b4d5fee64..9539352f6 (inventory agent, checked by the reviewer) → counts above; gllvmTMB #1236 and #1283 open drafts, CONFLICTING; #1238 iJSDM forensic follow-up; R's `docs/design/capability-status.md` rates ISDM-01 to 03 point-fit-recovery with no interval claim.
- **Brain** → `search_notes` "gllvmTMB integrated species distribution model isdm temporal AR1 new structures since 0.7.0" (all projects) and "... re-freeze oracle" (vault); `grep -in "true parity" memory/DECISIONS.md` → dr30 (iSDM design research), dr46 (temporal: R's first slice is a Gaussian rank-one latent score), D-220, D-290, D-292; 23 parity decisions recorded; four Packet-1 items already answered on 2026-09-24 in `LOOP/lanes/true-parity-20260924/GOAL.md` → reuse.
- **Verdict** → re-point the existing machinery at P1 additively, repair its two holes (receipts not resolved, ledger not tracked), then build the gap.

## Lanes and working folder

| Lane | Owns | Rule |
|---|---|---|
| NB per-species (assigned by Shinichi) | `grouped_dispersion.jl` NB2 kernel | stay out; review its PR when asked |
| Gaussian intercepts (#519) | `gaussian_intercept.jl`, confint entry points, `cv.jl`, `formula.jl` Normal branch, `postfit.jl` | stay out; shared appends only |
| auto-d (#518, gllvmTMB #1324) | `select_lv`, `model_selection.jl`, `fit_gllvm.jl` K/d, `binomial.jl` ridge | message before any Binomial adapter |

Migration, no handover needed: the NB per-species and Gaussian lanes finish where they are; auto-d may move itself in place at a pause; this session stays for today's queue; the programme starts in a new session opened in the GLLVM.jl folder, from this plan and the lane kit, with a paste-ready prompt. Rules: absolute paths; code changes only in worktrees under `~/local-scratch/`; briefs tell agents to read the target repo's `AGENTS.md`; leases on exact files with an own `LANE_ID`; no agent messages another.

## Destination (the stopping condition)

At pin P, with every gate read from a TRACKED file on origin/main:
- **C0** the P oracle exists alongside P0 (additive), `DEFAULT_R_REF`, `CAPABILITY_LEDGER_REF` and a required (non-advisory) CI job for the P twin tests point at P.
- **Receipt carry rule:** a P0 receipt counts at P only if every file in its `source_pins` is byte-identical at P; otherwise its row becomes `PARTIAL_STALE_AT_P1` until re-measured or signed.
- **C1** every required row at P is bound to a receipt that resolves on origin/main (for `required_core`, the receipt shows a Julia call) or is maintainer-signed.
- **C2 to C5** as before, over a scoreboard whose rows include every P1 capability inside the claim boundary (isdm 1FO plus predict, temporal 1FO, ordinal_logit 1FO, zi 1FO, and so on); row count read from the file.
- **C6** reverse-gap list with a written decision per item, including the Julia-only extras (SourceCovariance, two-part ZI).
- **C7** met.
- **C8** every R export at P is twinned (case-map row with a Julia receipt) or signed; name matches alone never count (`SEMANTIC_DIVERGENCE` set keeps `zi_*` and similar in FORWARD until signed).

## Workstreams

| WS | Slice | Member · model · effort | Time | Output | Dep |
|---|---|---|---|---|---|
| 0 | Additive re-pin: P1 oracle beside P0 (pin appears in about 555 tracked files; many tools refuse a wrong ref), `CAPABILITY_LEDGER_REF` pinned, required P1 CI job, case-map rows for the +25 exports and +11 S3 methods (classification proposed, you sign), stale-row scan under the carry rule, the 20 isdm rows reclassified, `check.mjs` committed as `tools/true_parity_check.mjs` with dynamic row count, receipt resolution and negative controls; ledger and receipts tracked under `docs/dev-log/core070/` | Julia engineer · Sonnet · high; review Fable · high | 4 to 5 days | PRs + tracked ledger + stale-row count | — |
| 0r | Recon: per new export, the R file, tests and examples | Scout · Haiku · low | 0.5 day | `reviews/p1-export-recon.md` | 0 |
| 1a | iSDM spec from R's public door at P1: `R/fit-multi.R` (contract and long-table engine), `R/isdm-*.R`, `R/offset.R`, `tests/testthat/test-isdm-public-door.R`, `test-isdm-predict.R`, #1238, dr30; Julia side `src/families/mixed.jl`, `src/spde_latent.jl`. iSDM needs per-row family within a trait (a new axis) | Gauss · Fable · high | 2 to 3 days | `docs/design/isdm-port-spec.md` | 0, your iSDM-scope decision |
| 1b | iSDM kernel and fitter, non-spatial, Laplace, in new files (not `mixed.jl`) | Sonnet · high; Fable for the coupling if hard | 10 to 12 days | PRs | 1a, #514 merged |
| 1c | iSDM twin tests (`test/parity/`), first order plus predict, whole-fit checks | Sonnet · high; review Fable | 3 days | receipts | 1b |
| 1d | iSDM `engine = "julia"` route in gllvmTMB (D-292) | Sonnet · high | 2 days | gllvmTMB PR | 1c only |
| 2 | Small twins one per PR: `ordinal_logit`, `extract_latent_scores`; medium: R-semantics `zi_*` (Julia two-part stays as extra), `meta`/`meta_V`, multinomial with latent variables, Wald `ordination_uncertainty` | Sonnet · high; review Fable | 1 day small, 2 to 3 days medium | PRs | 0 |
| 3 | Re-measure stale rows at P1 (count from WS0), T9 bindings, D3 to D5 and D8, non-Gaussian grouping pairing | Sonnet · medium | sized by the stale count | receipts | 0, 5 |
| 4a | gllvmTMB #1236 bridge expansion rebase and finish; #1283 recorder | Sonnet · high; review Fable | 3 days | gllvmTMB PRs | — |
| 4b | Real-data workflows C1 to C5 | Sonnet · high | 1 to 2 days each | ACC receipts | 4a |
| 5 | Silent-failure backlog: resume #514 and #515, #504 adapters, #505, Tweedie grouped | Sonnet · high; review Fable | about 4 days | PRs | — |
| 6 | Temporal at R's scope (Gaussian, rank-1, AR1/OU; extract, forecast, bootstrap, profile, compare) | Fable spec, Sonnet build | 2 to 3 weeks | PRs | 1c |
| 7 | Column-coefficient grammar at R's scope (Gaussian point model, no intervals; 109 R files, 25,724 lines) | Fable spec, Sonnet build | 4 to 6 weeks | PRs | 6, boundary decision |
| 8 | `spatial_dep` and `spatial_*` on the existing SPDE stack | Fable spec, Sonnet build | 3 to 4 weeks | PRs | boundary decision |
| 9 | Phylo latent A14/A15, ordination A12/A13 | Fable spec, Sonnet build | 3 to 4 weeks | PRs | 3 |
| D | Decision packets with drafted replies | Ada · Opus | 0.5 day each | packets | — |
| V | Mechanical verify per milestone | Haiku · low | minutes | ledger output | milestones |
| R | Melissa plan-vs-actual per milestone | Sonnet · medium | 0.5 day | `docs/dev-log/plan-actual/` | milestones |

At most 3 agents live, counting reviewers; the orchestrator schedules at most two builds at once.

## What needs Shinichi

| When | Decision | His time |
|---|---|---|
| Day 1 | Reopen the Totoro socket | 1 min |
| Day 2 | **Packet 1 (13 items).** Kept from the open list: T11 collisions, T12 unit_obs, T14 NB2 boundary, T15 fixture seeds, B-04 what "pair" means for grouping, T5, ledger-gap ranks. New: the receipt-carry rule; the iSDM twin target and scope (public door, non-spatial, Laplace, first order plus predict); classification of the +25 exports and +11 S3 methods; ledgers tracked in git; the semantic-divergence rule for name twins; who signs the `select_lv` row. (Already answered 2026-09-24, not re-asked: S4 probe, D3 Stage 1, Totoro #323 Track A, delta dispersion A.) | 45 min |
| Day 2 | **The P1 claim boundary:** which of temporal, column grammar, spatial and phylo latent are inside P1, and which close by signed disposition | 15 min |
| Week 2 | Packet 2: the 47 pending-decision rows and 22 spec-defect rows (mostly print and format reclassifications) | 45 min |
| Week 5 | Packet 3: the 91 reverse-gap classes and the Julia-only extras | 60 min |
| Daily | "merge #A, #B when green" | 2 min |

## Estimate

- To the HEADLINE (iSDM through R's door, with twin tests and the bridge route): about 5 weeks (WS0 1 week, iSDM spec and build 3 weeks, tests and bridge 1 week), with small twins and the backlog alongside.
- Full P1 parity with everything inside the boundary: **18 to 22 weeks** at 2 builds at a time. If you place column grammar and spatial outside P1 (signed dispositions), about **10 to 12 weeks**. The measured pace of the 0.7.0 programme (0 of 32 rows in a month) is the reason for the wide range; the largest builds carry 1.5x overrun risk.
- Your decisions gate the C1 and C6 clauses entirely; build time is the rest.

First two weeks:

| Day | Work |
|---|---|
| 1 | This session: message the three lanes; merge #512 and #516 when green; resume #514 and #515; create the lane kit (`lane_launch.sh "…/GLLVM.jl" true-parity-latest`) with this plan and a start prompt. You open the new session in the GLLVM.jl folder |
| 2 | New session: WS0 starts (additive pin, tracked ledger); Packet 1 and the boundary question drafted |
| 3 to 5 | WS0 continues (stale-row scan, case-map rows, check tool with controls); `ordinal_logit`; #1236 rebase |
| 6 | WS0 review; iSDM spec (1a) starts once the scope decision is in; `extract_latent_scores` |
| 7 to 8 | iSDM spec review; #514 merged; `zi_*` R-semantics twin |
| 9 to 10 | iSDM kernel (1b) starts in new files; stale-row re-measurements begin (Totoro) |

## Acceptance ledger (tracked: `docs/dev-log/core070/true-parity-latest/GATES.md`, oracle `tools/true_parity_check.mjs`)

C0 to C8 as defined above; F-gates per capability inside the boundary (a scoreboard row with a resolving P1 receipt); X2 all scoreboard rows done; M1 joint note (manual). Negative controls before any baseline: a dangling receipt fails C1; a capability in C8 with no scoreboard row fails C2; a name-only match fails C8; a stale carried receipt fails its row.

## Pre-authorisation (after G0)

PRE-AUTHORISED: worktrees under `~/local-scratch/`; local Julia and R tests; local commits; pushing branches and opening draft PRs in GLLVModels.jl and gllvmTMB; Mac at 4 cores per lane; Totoro <=30 min on <=4 cores once the socket is back; kohaku <=8 vCPU for short checks; new public API that mirrors an R export at P1 under R's name, merged only on your word.
MUST STOP: every merge (your word, batchable daily); releases, tags, `Project.toml`; public parity claims; the shared `_laplace_mode`; DRAC jobs (estimate first); any run over 3 hours; another lane's files; case-map classifications (you sign).

## Verification

Per PR: twin test against R at P1, red then green, on Julia 1.10 and 1.13; Fable review with its own probe and a whole-fit check; CI green, with P1 twin tests in the required job. Per milestone: `node tools/true_parity_check.mjs` modes plus the negative controls, a Haiku mechanical check, Melissa's note, and a handover on main.

## Lessons carried in

Entering plan mode pauses running builders: park or finish them first. Never block an unattended loop on a question. No closing keywords in titles. One Julia process per agent. Whole fits, not only sites. Receipts and ledgers under gitignored paths get lost: track them.
