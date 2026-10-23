```
🎯 GOAL
Solo platform: Claude Code (this session, Opus 5.5; read from the runtime)
Lane: GLLVModels.jl true parity (Claude). Other lanes, not touched: Cursor lease src/grouped_nongaussian_fit.jl ·
  Codex docs #433/#437/#439 · Claude reader #444 · Cursor speed/latte.
Deliverable: all four paste gates of the 2026-09-24 handover EXECUTED, each to its runbook's stop line:
  #409 S4 probe receipt · #410 Totoro #323 Track A receipt · #399 Delta dispersion A merged + D1 remeasured ·
  #411 Stage 1 harness merged + Stage 1 slice PR. Programme docs state the live truth. After-task + handover.
HEADLINE: the four DRAFTs rebased onto the renamed package (conflict fix + GLLVM→GLLVModels port), CI-green,
  and merged in order. Everything else hangs on that.
IN PARALLEL: 4 rebase/port builders (one per DRAFT, disjoint worktrees) ∥ docs-truth PR.
DEFER: Project.toml bump; tag/release; Stage 1 heavy grid on Totoro (needs its own D-139 ack); any gllvmTMB
  write (D-220); foreign lanes; RETRACTED #363/#314/#384/#391; D-280 trial (deferred by Shinichi).
DISCIPLINE: verify = unlazy `--reverify` per leaf + Rose on every claim + a D-43 panel at the final claim ·
  compute = Mac (OPENBLAS/JULIA threads 1, ≤4 cores/lane), Totoro only for #410 after its D-139 pre-run ·
  closure = every leaf gate met or ABANDONED with a reason; handover written.
```

## Context

Shinichi confirmed the handover is `GLLVM.jl/docs/dev-log/handover/2026-09-24-claude-handover.md`: GLLVModels.jl ↔ gllvmTMB true parity. The handover's no-paste path was a docs refresh. In this session Shinichi **gave all four paste strings** and approved **pre-rebasing all four DRAFTs**, so the OWED work is now the full paste path (handover step 4) for all four gates.

## WHAT SHINICHI TOLD US (verbatim selections, 2026-09-24)

- Pastes: `S4 probe yes`, `G0 Stage 1`, `ack Totoro D-139 #323 Track A`, and, after a plain-language explanation of the trade-off, `accept delta dispersion A`.
- Pre-rebase: "Yes, all four as DRAFT": rebase, port and push all four now, and merge each only under its paste.
- D-280: "Defer to next clean plan". This session is not a clean arm, so the trial fires on the next fresh ultra-plan. Record this in D-280.

## Evidence base (Phases 0–0.6; details measured this session)

- **Preflight (Shannon):** FOREIGN LANE ACTIVE (codex, cursor) plus one Claude lane (#444). Lease `cursor:GLLVM.jl src/grouped_nongaussian_fit.jl`. No other Claude process is in GLLVM.jl. None of the four DRAFTs touches the leased file.
- **Sweep receipt:**
  - repo: `git fetch`, `merge-tree` ×4, `branch_drift_check.sh`. Main is `6ba1770ab`: #464 plus **#463, which reverted #453; latte is back to default OFF**.
  - twin: gllvmTMB `1d7e68da1`; #1283 DRAFT @ `97214679c`, with a local worktree at `/private/tmp/destination-b-s4-phylo-dep-formula-20260910`; frozen oracle `b4d5fee64` (worktree `GLLVM.jl/.worktrees/gllvmtmb-b5-frozen-20260909`).
  - brain: `search_notes` ×2 plus greps over AGENT_LOG, DECISIONS, OPEN_QUESTIONS and journal. D-224 says *"critical path waits for the owner's paste unlocks"*; D-220 and D-269 also apply. Nothing to reuse.
  - roster: `MODEL-ROUTING.md:85` current as of 09-24.
  - **Verdict: build the gap.**
- **Load-bearing finding:** all four DRAFTs branch from `7a6fe4962`, **before the #423 rename** (`69a69b0a0`). Their 09-17 "MERGEABLE + Julia green" is void.
  - Old-name code: #399 `GLLVM._family_ci`, `GLLVM.rr_theta_len`; **#411 `using GLLVM`**, which fails at load.
  - Conflicts: `docs/make.jl` in all four is mechanical (DocumenterVitepress). #399 `test/test_second_order_delta_followup.jl` and #411 `test/parity/loading_profile_confirmatory_substrate.jl` are semantic (substrate moved to `src/loading_profile_confirmatory_internal.jl`).
- **CI:** about 75 min per PR (8 Julia shards). Frozen-R smoke is advisory and fails; that is expected. `main` has no branch protection.
- **Route check:** the destination is writable, no slice is "TBD", and every output is a path or PR. The route is knowable.

## DECISIONS LOCKED

1. All four gates execute. Merge order: **#409 → #410 → #399 → #411-harness → Stage 1 slice PR**. The two harness-only PRs go first because they collide least. #399 merges only after its default flip and D1 remeasure. #411's Stage 1 slice is a new PR after the harness merges.
2. Each DRAFT rebase includes the rename port in the same commit series. No `GLLVM.` or `using GLLVM` may remain in added `.jl` code.
3. D1 tolerances are never widened. No `Project.toml` bump.
4. After each merge: re-run `merge-tree` on the remaining DRAFTs. Rebase again if needed. Watch `main` CI; red `main` stops further merges.

## Slice table

| # | Slice | Member | Model · effort · dispatch | Time | Owns (worktree / paths) | Dep |
|---|---|---|---|---|---|---|
| S0 | Coordination: worktrees, leases, ledger `.unlazy/true-parity-20260924/`, docs-truth PR (board, paste packet, handover tip `6ba1770ab`, #463, pastes recorded 09-24, void pre-rename green, AGENTS.md snapshot) | Ada (this session) | Opus 5.5 · high · parent | 40 min | `~/local-scratch/lanes/GLLVM.jl-true-parity-20260924` (`claude/true-parity-docs-20260924`): the 4 docs paths + check-log | none |
| B399 | Rebase #399 on main; resolve both conflicts; port to `GLLVModels`; local targeted test; force-push (lease-guarded); stays DRAFT | Noether-builder | Sonnet · high · claude/model-param | 45 min | `…/GLLVM.jl-tp-399-20260924` (`feat/delta-dispersion-a-scaffold-20260916`) | S0 |
| B409 | Same for #409 (make.jl only) | builder | Sonnet · medium · claude/model-param | 30 min | `…/GLLVM.jl-tp-409-20260924` | S0 |
| B410 | Same for #410 (make.jl only) | builder | Sonnet · medium · claude/model-param | 30 min | `…/GLLVM.jl-tp-410-20260924` | S0 |
| B411 | Same for #411 (make.jl + semantic substrate move + `using GLLVM`) | Noether-builder | Sonnet · high · claude/model-param (promote to Opus if the substrate merge needs a design call) | 60 min | `…/GLLVM.jl-tp-411-20260924` | S0 |
| V1 | Mechanical verify of B*: post-rename base, zero old-name code, MERGEABLE, CI rollup | scout | Haiku · low · claude/model-param | 10 min | read-only | B* + CI |
| X409 | Merge #409 on green, then run the S4 probe vs #1283 `97214679c` locally. **State an estimate first**; >30 min → pre-run + ask. Write the classified receipt. Update DestB/GOAL docs. | reuse B409 builder + Ada | Sonnet · high | 1–2 h | #409 paths + receipt dir | V1 |
| X410 | Merge #410 on green; `--dry-run`; **D-139 pre-run on Totoro** (one cell, non-empty gradient_max) → **PAUSE for Shinichi** → full Track A (90–150 min, single process, OPENBLAS=1) → receipt NATIVE-06/10/12 | Ada (ssh via existing Totoro socket) | Opus 5.5 parent | 20 min + pause + 2.5 h | #410 paths + Totoro `~/…` run dir | X409 merged |
| X399 | On #399: append ACCEPTED (A) to `docs/dev-log/decisions/2026-09-15-delta-dispersion-alignment-pending.md`; flip the delta fitter/`fit_gllvm` default to `:species` with `:shared` opt-in; postfit vector σ/α; docs/tutorial/README per design rule 3; D1 remeasure via `tools/core070_second_order/smoke_delta_*_eoo.jl` with `GLLVM_PARITY_R_LIBS` → frozen oracle lib (build it into a scratch lib from `b4d5fee64` if absent); ready + merge on green | reuse B399 builder | Sonnet · high (promote to Opus if the postfit vector-σ path needs engine judgment) | 2–3 h + CI | #399 worktree | X410 merged |
| X411a | Merge the #411 harness on green | Ada | — | 10 min + CI | — | X399 merged |
| X411b | Stage 1 slice, new PR `claude/d3-stage1-slice-20260924` per `docs/dev-log/plans/2026-09-16-*stage1*` runbook: `lambda_constraint` pin on Gaussian + ordinary latent fitters; public confirmatory export with docstring; pinned-cell tests; one R-aligned cell receipt; ledger rebind; docs/tutorial/README in the same PR; merge on green | reuse B411 builder | Sonnet · high → Opus if the export signature is not fixed by the runbook (then PAUSE: API) | 3–4 h + CI | new worktree `…/GLLVM.jl-tp-stage1-20260924` | X411a |
| V2 | Mechanical re-verify of every leaf (`gate-check --reverify`) | scout (reuse V1) | Haiku · low | 15 min | read-only | all X |
| D43 | Completion panel on the claim "four paste gates executed to their stop lines" | 2 fresh reviewers + 1 fresh verifier (Rose lens) | 2× Sonnet · high, 1× Opus 5.5 · high | 30 min | read-only | V2 |
| Z | Close: board final, after-task report, handover (`handover_gate.sh` first), check-log; brain AGENT_LOG + journal + D-280 deferral line; slop_check on shipped prose | Rose (this session) | Opus 5.5 parent | 45 min | S0 worktree + vault (local commit) | D43 |
| M | Melissa reconcile plan vs actual → `docs/dev-log/plan-actual/2026-09-24-true-parity-four-gates.md` | Melissa | Sonnet · medium | 15 min | that file | Z |

PARALLEL: {B399, B409, B410, B411} after S0. SEQUENTIAL: V1 ← B*; X409 → X410-merge → X399 → X411a → X411b (merge order). The Totoro full run overlaps X399/X411 work once approved.
FAN-OUT BUDGET (checkpoint G0-20260924): new production children 5/6 (4 builders + 1 Haiku scout, all reused for later slices) · ceiling 0 (promotion only on the named triggers) · plus the plan-review child below. The D-43 panel is separate by rule.
SCOUT SUITABILITY: yes. V1/V2 are bounded, read-only and mechanical.
CONTEXT BRAKE / LANE: the run spans hours. Execution runs as **`/arc-loop`** with this plan as the on-disk goal. At the first compaction, freeze scope to the current merge. At the second, write the handover and start a fresh task.
ESTIMATE: about **12–16 h wall-clock** across 2–3 sessions (mostly CI waits and the Totoro run). About 4 h of agent-active work. A handover is likely after X399.

## Acceptance ledger (written in S0 before any dispatch; `.unlazy/` is git-ignored here, confirmed by `git check-ignore`)

`.unlazy/true-parity-20260924/GATES.md` holds the scope and `OWNS:` globs above. It has one `gates/leaf-<id>.md` per slice. Every CHECK prints a success-only token. Core gates:

- **leaf-B{399,409,410,411}**:
  - G1 post-rename base: `git merge-base --is-ancestor 69a69b0a0 origin/<branch> && echo POST_RENAME`
  - G2 zero old-module code: node script counting `(using|import) GLLVM\b|\bGLLVM\.[A-Za-z_]` in added `.jl` lines of `git diff origin/main...origin/<branch>`, prints `OLDNAME=0`
  - G3 `gh pr view N --json mergeable,isDraft` → `MERGEABLE true`
  - G4 local: the branch's changed test files run under `julia --project=.` with exit 0 → `LOCAL_TESTS_PASS`
  - G5 CI: every `Julia*` check `SUCCESS` → `JULIA_GREEN`
- **leaf-X409**: G1 #409 `MERGED`. G2 the estimate is written in the receipt before the run. G3 the receipt file exists with non-empty pass/fail/classification counts. G4 `git -C <gllvmTMB worktrees> status --porcelain` is empty, so gllvmTMB is untouched.
- **leaf-X410**: G1 `MERGED`. G2 dry-run exit 0. G3 the pre-run cell writes a finite gradient_max. **G4 MANUAL: Shinichi approves the full run after seeing the pre-run.** G5 the receipt has finite gradient_max for NATIVE-06/10/12. G6 no Totoro process from this lane remains (`pgrep -u snakagaw -f runparity` is empty).
- **leaf-X399**: G1 the ACCEPTED (A) block is present. G2 a test asserts the default `disp_group == :species`, and the `:shared` opt-in test passes. G3 `git diff` of the PR shows **no rtol/atol change**. G4 the D1 receipt is written with its verdict. **FAIL → PAUSE, no merge**. G5 `JULIA_GREEN`. G6 `MERGED`.
- **leaf-X411**: G1 harness `MERGED`. G2 the new export has a docstring (`Docs.hasdoc` check). G3 pinned-cell tests pass locally and in CI. G4 docs, tutorial and README are touched in the same PR. G5 `MERGED`.
- **leaf-S0/Z**:
  - G1 the docs on main cite `6ba1770ab` or later.
  - G2 no undated "MERGEABLE + Julia green" claim remains.
  - G3 all four paste strings are recorded with 2026-09-24.
  - G4 `slop_check.py` shows no ❌ on the after-task report and handover.
  - G5 `agent_mention_check.py --text` is clean on every PR body.
  - G6 `handover_gate.sh` states are declared.
  - G7 the D-280 deferral line is in the vault DECISIONS.

Checks are read at G0 and approved once (`gate-check --approve`). A new or changed check is a new decision.

## Pre-authorisation envelope

```
PRE-AUTHORISED AFTER G0: worktrees under ~/local-scratch/lanes/; lane leases; scoped edits on the 4 DRAFT branches,
  the docs branch and the Stage 1 slice branch; local Julia tests (threads 1); a scratch R lib with frozen
  gllvmTMB b4d5fee64; local commits; the gate-check commands above; vault AGENT_LOG/journal/DECISIONS appends.
REMOTE AUTHORITY (from the four pastes + "Yes, all four as DRAFT"): push --force-with-lease the 4 DRAFT branches;
  push new branches; open PRs; mark ready and merge #409, #410, #399, #411 and the Stage 1 slice PR and the docs PR
  when Julia CI is green (merge-when-green skill); ssh to Totoro via the existing socket for the #410 pre-run.
MUST STOP: Totoro full run (after pre-run, D-139); D1 still FAIL after alignment; any tolerance change; red main CI
  after a merge; an API choice the runbook does not fix; any gllvmTMB write; Project.toml bump/tag/release;
  touching a foreign-lane file; a new compute cost beyond the stated estimates.
```

## Members plan-review (Rose + Noether, Sonnet high): RUN WITH CHANGES. All applied below.

Ada re-measured the load-bearing items herself.

1. **BLOCKING, confirmed:** #409, #410 and #411 each touch both `test/runtests.jl` and `docs/dev-log/check-log.md`, and #399 touches one of them. Sequential merges collide there, not only in `docs/make.jl`. `runtests.jl` is **position-sharded**, so insertion order moves shard membership.
   → Every rebase and re-rebase follows the **`ci-shard-suite` skill's** conflict procedure: parse-check `runtests.jl`, count includes (no duplicates, none lost), and keep new files appended at the end. Leaf gates B*/G4 gain: "include count = main count + this PR's new files" (`INCLUDES_OK`), and run the shard(s) that contain the PR's files, not just the files. `check-log.md` conflicts keep both dated sections.
2. **BLOCKING, resolved by Shinichi:** the Totoro runbook names "Codex (or maintainer)" as Track A executor. Shinichi chose **"Claude runs it here"**. X410 runs from this lane through the existing Totoro socket, with one Julia process and `OPENBLAS_NUM_THREADS=1`. The executor deviation is recorded in the receipt and the after-task report. The full run still waits for his OK after the pre-run (D-139).
3. **BLOCKING:** the Verification §3 fences now include **"Delta A accepted ≠ D1 pass"**, and the D-43 panel scores closeout wording against all four fences.
4. SHOULD-FIX, applied: G2 old-name gate counts **code only**, skipping comments and string literals. Prose `GLLVM.jl` repo-name mentions become a separate non-blocking rename checklist item. The GitHub repo is `itchyshin/GLLVModels.jl`.
5. SHOULD-FIX, applied: the #1283 worktree `/private/tmp/destination-b-…` is **gone** ("prunable"). X409 recreates it from the live ref `codex/destination-b-s4-phylo-dep-formula-20260910` = `97214679c`, read-only.
6. SHOULD-FIX, applied: aggregate compute cap. At most **2 Julia processes at once** across the four builders (precompile and tests staggered), each with `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. The shared Mac also hosts the foreign lanes.
7. SHOULD-FIX, applied: X411b gate G6 **"no `loading_profile` deprecation-shim removal in this PR"** (runbook fence).
8. NIT, applied: G3 reads `mergeable == "MERGEABLE"` (string enum).

## Verification (end to end)

1. `node ~/.claude/skills/unlazy/scripts/gate-check.mjs --reverify` on every leaf exits 0, or the leaf lists its ABANDON reasons.
2. `gh pr view` for #399/#409/#410/#411 and the Stage 1 slice shows MERGED, and `main` CI is green at the final tip.
3. The receipts for S4, Track A, and D1 exist on main, and Rose checks each claim's wording against its receipt with all four fences: **S4 probe ≠ parity · Track A ≠ programme complete · Stage 1 ≠ full R grid · Delta A accepted ≠ D1 pass**.
4. The D-43 panel returns no BLOCKING finding.
5. Melissa's plan-actual file lists every deviation.
