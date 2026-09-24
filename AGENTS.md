# GLLVModels.jl Agent Instructions

`GLLVModels.jl` is a Julia implementation of the Gaussian + phylogenetic
Generalised Linear Latent Variable Model, built as a Julia companion to R's
`gllvmTMB` engine (Core070 ledger accounting ≠ true parity; see
`docs/src/gllvmtmb-parity.md`).

## Project identity

- Status: v0.3.0 development / integration tree — broad Gaussian, non-Gaussian,
  missing-data, structural-dependence, and bridge capability is present, but
  release/tag signoff remains gated by the issue ledger, R bridge parity, docs,
  and Rose audit.
- Headline result: median 265.1× (range 161–698×) on the published Gaussian
  closed-form profile grid over R/`gllvmTMB`, with log-likelihoods and point
  estimates agreeing to at least six significant digits on that grid (worst case
  `|Δ logLik| = 2.343e-07`, not ~2.2e-16 machine precision).
- Phylogenetic representations: sparse (CHOLMOD), contrasts, edge-incidence;
  all return identical log-likelihoods to ≤1e-13 on cross-check cells (internal
  equivalence, not an R↔Julia parity claim).
- Next milestone: finish-gap hardening for the R-Julia twin: reconcile #95/#94,
  keep runtime fixes (#91/#92/#96 and gradient defaults) accuracy-anchored, wire
  verified O(p) sparse-phylo gradients, validate the R bridge, and keep the
  public capability matrix honest.
- Reference design: `~/.claude/plans/users-z3437171-downloads-gllvm-jl-rocke-mighty-sundae.md`
  is the canonical roadmap.

## Design rules

1. Do not add a new response family without an ADEMP simulation-recovery
   test that exercises the new likelihood.
2. Do not add an exported function without a docstring (DocStringExtensions
   conventions).
3. Do not change a user-facing API without updating tutorials, reference
   docs, README, and tests **in the same PR**.
4. Do not change likelihood parameterisation without updating the
   corresponding math doc under `docs/dev-log/decisions/`.
5. Do not silently widen test tolerances. If a test breaks, fix the cause,
   not the tolerance.
6. Keep commits surgical: one concern per commit. Engineering changes,
   cosmetic renames, and chores stay in separate commits.
7. Every meaningful change updates `docs/dev-log/check-log.md`.
8. Every completed task closes with an after-task report under
   `docs/dev-log/after-task/YYYY-MM-DD-*.md` (Definition of Done below).
9. If code is ported from `gllvmTMB`, the comparison bench repo, or any
   external source, document provenance in `docs/dev-log/decisions/` or
   `inst/COPYRIGHTS` before treating the change as complete.

## Convention-change cascade

If a syntax change, argument rename, formula-grammar change, or extractor
rename ships, the same PR must atomically update: (a) docstrings, (b)
Documenter tutorials and reference pages, (c) tests, (d) README.md,
(e) any roadmap or status table referencing the old name. Partial cascades
are blockers at the Rose pre-publish gate.

## Standard commands

```sh
julia --project=. -e 'using Pkg; Pkg.test()' # full suite incl. Aqua/JET (what CI runs)
julia --project=. test/runtests.jl          # quick core suite (skips quality tools)
julia --project=docs docs/make.jl           # local Documenter build
gh run list --limit 3                       # confirm CI state
git status; git rev-parse --short HEAD      # evidence-first state check
```

## Standing review roles

These names are shorthand for recurring review perspectives. They do not run
continuously; the orchestrator launches them only for bounded tasks. Use the
canonical names in status updates; do not rename them.

| Name | Role | Owns / leads |
| --- | --- | --- |
| **Ada** | Orchestrator and maintainer voice | Phase planning, after-task review, final consistency audit |
| **Boole** | Julia formula and macro grammar | `StatsModels.jl` integration, `@formula`, user-facing API syntax |
| **Gauss** | Julia numerical engine | CHOLMOD, SparseArrays, ForwardDiff stability, Takahashi selected-inverse, `src/likelihood*.jl`, `src/sparse_phy*.jl` |
| **Noether** | Symbolic ↔ Julia API ↔ math kernel consistency | Closed-form Gaussian and (Phase 3) Laplace cross-layer correctness |
| **Darwin** | Ecology / evolution audience | Tutorial framing, applied use cases, phylogenetic signal interpretation |
| **Florence** | Scientific figure editor | CairoMakie.jl figures, Confidence Eye contract port |
| **Fisher** | Statistical inference | Profile / Wald / bootstrap CIs, identifiability, validation against R |
| **Pat** | Applied PhD-student tester | Quickstart readability, error messages, tutorial accessibility |
| **Jason** | Julia ecosystem scout | MixedModels.jl, Turing.jl, Phylo.jl, Distributions.jl — idioms and packages to reuse |
| **Curie** | Simulation and recovery testing | ADEMP recovery tests via `Test.jl` + `StableRNGs`, edge and malformed cases |
| **Emmy** | Julia package architecture | Multiple dispatch, types, exports, `Project.toml`, Aqua.jl, JET.jl |
| **Grace** | Julia CI and reproducibility | GitHub Actions matrix, Documenter.jl deploy, Pkg.jl registry hygiene |
| **Rose** ★ | Systems auditor | Pre-publish gate, claim-vs-evidence audit, README/CLAUDE.md drift detection — **the most important guardrail** |
| **Shannon** | Cross-team coordination | Branches, PRs, after-task coverage, file-overlap and lane checks |
| **Karpinski** | Julia-specialist | Type stability, dispatch, performance, sparse linalg, AD backends; `@code_warntype`, JET, Allocs.jl |
| **Hopper** | R↔Julia translator | API equivalence, idiom mapping, `gllvmTMB`↔`GLLVM.jl` parity tests via RCall.jl |

Full responsibility detail lives in §2 of the reference plan.

## Phase state snapshot

- **True-parity closeout (2026-09-24).** All four paste gates executed; #409/#410/#399/#411 merged. S4 waits on a recorder fix (gllvmTMB#1283); Track A and D1 receipts on main. `Project.toml` stays **`0.3.0`**. **START HERE:** `docs/dev-log/handover/2026-09-24-claude-handover-closeout.md`.

- **True-parity execution (2026-09-24, later).** `origin/main` @ **`6ba1770ab`** (#463 reverted #453; latte default OFF). Shinichi pasted all four gate strings; DRAFT **#399/#409/#410/#411** are being rebased onto the renamed package and executed in order #409, #410, #399, #411 harness, Stage 1 slice. Lane kit: `LOOP/lanes/true-parity-20260924/`. Receipts land on the board as each gate closes. `Project.toml` stays **`0.3.0`**. **START HERE:** `LOOP/lanes/true-parity-20260924/checkpoint.md`, then `docs/dev-log/2026-09-14-true-parity-pending-board.md`.

- **True-parity continuation (2026-09-24).** `origin/main` @ **`1703c54b4`**
  (post-rename **GLLVModels** #423/#425; speed/latte #453 on main; programme goal
  **not** complete). Paste-gated DRAFT harnesses **#399/#409/#410/#411** still
  **DRAFT until Shinichi paste**; **#409/#410/#411** were **CONFLICTING** with
  `main` at handoff (rebase after paste). **#357** bridge receipts **MERGED**
  (`5ee6dc596`; do not revert). `Project.toml` stays **`0.3.0`**. Multi-lane:
  see `docs/dev-log/coordination-board.md`; do not orphan reader (**#444**) or
  latte (**#463**) lanes. Board:
  `docs/dev-log/2026-09-14-true-parity-pending-board.md`. Paste packet:
  `docs/dev-log/owed/2026-09-16-post-402-paste-packet.md`. **START HERE:**
  `docs/dev-log/handover/2026-09-24-claude-handover.md`.

- **True-parity continuation (2026-09-17).** `origin/main` @ `e590eb9ec` after
  docs tip chain through **#418** (paste-ready status for four DRAFT harness
  heads). **#357** bridge logLik receipts **MERGED** (#357, `5ee6dc596`; do not
  revert). DRAFT **#399/#409/#410/#411** remain **DRAFT until paste** (do not
  merge without Shinichi paste). Still paste-gated: S4, D3 Stage 1, Totoro T4,
  Delta dispersion A; `Project.toml` stays `0.3.0`. Goal **not** complete.
  Board: `docs/dev-log/2026-09-14-true-parity-pending-board.md`. Paste packet:
  `docs/dev-log/owed/2026-09-16-post-402-paste-packet.md`. Handover:
  `docs/dev-log/handover/2026-09-15-mac-studio-true-parity-handover.md`.

- **HANDED OVER to a single Cursor lane holding BOTH repos (2026-09-04).** GLLVM.jl `main`
  `4d57cb29` (CI 8/8 shards + Documenter green, 0 open PRs); gllvmTMB `main` `72ed68b5`
  (R-CMD-check green). Both lanes reported ready, nothing in flight. The move follows the
  H² twin pattern (one orchestrator chat, multi-root workspace). Two invariants get WEAKER
  under one lane and need a mechanical guard, not prose: the gllvmTMB read-only-reference
  boundary, and the frozen oracle `b4d5fee6` — see the handover §3/§4 and the verified
  `parity_ledger.R` ref asymmetry (vault `D-220` caveat 1). Confirmed directly by Shinichi
  to both lanes as **`D-220`**, which also rules that campaigns move with the lane but keep
  running on Totoro (ControlMaster socket, 150-core cap, D-139 estimates). Six maintainer
  decisions still open.
  **START HERE: docs/dev-log/handover/2026-09-04-cursor-handover.md**
  (sibling lane's own handover: gllvmTMB `docs/dev-log/handover/2026-09-03-claude-handover.md`)

- **Core 0.7.0 + AGHQ: MERGED to `main` (2026-09-03) as `2524b787`.**
  PR #277 merged with a merge commit (history preserved); lane branch
  `codex/core070-aghq-20260830` deleted — resume commands naming it in older
  handovers and slice notes are historical, and everything they point at is in
  `main`. CI green on `main`: 8/8 Julia shards (1.10 + 1.x) and Documenter; the
  frozen-R family smoke stays advisory (`continue-on-error`, rebuilt-oracle
  finding). gllvmTMB DRAFT PR #1236 (R bridge, green) is unchanged. Ledger 505
  required = **292 bound + 213 dispositioned**, 0 free (Rose-audited over 3 rounds; required set honestly
  resized from 533 by maintainer-approved bulk triage). ~100 new surfaces
  across 8 receipted conversion waves; all 12 maintainer decisions executed —
  incl. the CLOGLOG parity defect resolved as OURS (Fisher-vs-observed
  curvature; identity 2.099 -> 7.4e-12) and Student-t paired green at pinned
  df. Suite green at the merge: 13327 pass / 0 fail / 0 error. Compute: 10,000-fit
  DRAC recovery campaign; Julia-only Wald interval *coverage* in the arcG
  diagnostic grid ran 0.932–0.958 across 20 cells — undercoverage evidence, not
  a calibrated-coverage or R-parity certificate (gllvmTMB 0.7.1 withdrew its
  old total-variance “0.94 floor” claim; see `gllvmtmb-071-gap-sheet.md`).
  CARRIED-OVER: six maintainer decisions in
  docs/dev-log/core070/maintainer-decision-set-2026-09-03.md (only D3,
  `loading_profile` estimand scope, blocks a ledger row); 2 invalid
  realistic-size pairs needing an R-driver re-run with the seed in the
  filename; the Julia-vs-R wall-time gap on the largest realistic cell
  (2h36m vs 498s — a performance track, not parity); phylo S3/S4 pending the
  Q1-Q4 confirmation; 8 prior-cycle acceptance ledgers unmet.
  START HERE: docs/dev-log/core070/maintainer-decision-set-2026-09-03.md

- **Full-parity campaign day (2026-08-28).** PR #273 (12 commits,
  branch `claude/lane-beyond-20260824`): the Fisher-vs-observed curvature
  fault class CLOSED (census `KNOWN_OPEN` empty; Tweedie+probit flipped,
  GP-1 Fisher by decision, cloglog intrinsic); AGHQ unparked, Slice 0 done;
  delta `predictor = :shared` twin-identity mode; grouped structs record
  `hessian`; JuliaCall/Totoro embedding segfault root-caused + fixed
  (libunwind LD_PRELOAD). Maintainer decision batch re-scoped fences:
  `docs/dev-log/decisions/2026-08-28-arc-decision-batch.md`. Suites 6955 and
  6997 pass / 0 fail / 4 broken (Totoro). Session handed to Cursor.
  **START HERE:** `docs/dev-log/handover/2026-08-28-cursor-handover.md`.
- **Multinomial name-clash vs Distributions (2026-08-18).** PR #257
  after `origin/main` @ `13ccb7d5` (#259). Bare `Multinomial` in
  `test/test_multinomial.jl` was `UndefVarError` once `runtests.jl`
  had `using Distributions`. Public API stays `Multinomial`; tests
  qualify `GLLVM.Multinomial`. Mac-light **41/41**. After-task:
  `docs/dev-log/after-task/2026-08-18-multinomial-name-clash.md`.
- **Multinomial P1 engine (2026-08-18).** Branch
  `cursor/lane-parity-beyond-20260818` (PR #257). Identity ACCEPTED
  `docs/dev-log/decisions/2026-08-18-multinomial-identity.md`. v1 FE
  softmax only: marker `Multinomial`, `η₁≡0`, pack `(K−1)(1+p)`, one
  softmax per observation, no TMB pseudo-rows, no LV. Focused
  `test/test_multinomial.jl` **41/41** after the name-clash qualify.
  Ledger stays `missing`. No Δ.
  After-task: `docs/dev-log/after-task/2026-08-18-multinomial-engine.md`.
  Rose: Julia focused claim only ≠ twin Δ ≠ ledger promote ≠ LV.
- **ZINB+X confint under X (2026-08-14).** Branch
  `feat/zinb-x-confint-20260814` from `origin/main` @ `d589bd40` (#203).
  `confint(ZINBCovFit)` packs `[βz; γz_free; βc; γc_free; pack(Λc); log r]`;
  `_bridge_ci_guard_zinb_x` gone; `_BRIDGE_NO_CI_X_FAMILIES` empty;
  `ci_x_*` true for `zinb`. After-task:
  `docs/dev-log/after-task/2026-08-14-zinb-x-confint.md`. Rose: Julia CI
  claim only ≠ twin Δ ≠ ADEMP.
- **ZINB+X engine Arc 0 MERGED #203 (2026-08-14).** `main` @ `d589bd40`.
  `fit_zinb_gllvm_cov` / `ZINBCovFit` packing
  `[βz; γz; βc; γc; pack(Λc); log r]`, `Λ_z=0`, **shared scalar `r`**.
  After-task: `docs/dev-log/after-task/2026-08-14-zinb-x-engine.md`.
  Identity lock: `docs/dev-log/decisions/2026-08-13-zinb-x-identity.md`.
- **ZINB+X Identity Arc 0 ACCEPTED (2026-08-13, docs-only).** MERGED #202
  @ `daf95da6`. Decision:
  `docs/dev-log/decisions/2026-08-13-zinb-x-identity.md`.
- **ZIP+X confint under X MERGED #201 (2026-08-13).** `main` @ `8abdd751`.
  `confint(ZIPCovFit)` + bridge CI lift (FD Hessian). After-task:
  `docs/dev-log/after-task/2026-08-13-zip-x-confint.md`. Rose: Julia CI
  claim only ≠ twin Δ ≠ ADEMP ≠ ZINB+X engine.
- **ZIP+X engine Arc 0 MERGED #200 (2026-08-09).** `main` @ `5d570b11`.
  `fit_zip_gllvm_cov` / `ZIPCovFit` + identity/FD + bridge/`@formula`.
  After-task: `docs/dev-log/after-task/2026-08-09-zip-x-engine.md`.
- **Capacity S3 ZIP+X Identity MERGED #198 (2026-08-09).** `main` @
  `6f9050e5`. Decision ACCEPTED:
  `docs/dev-log/decisions/2026-08-09-zip-x-identity.md`.
- **Capacity S2 BetaBinomial grouped CI MERGED #197 (2026-08-09).** `main` @
  `9c2b18d6`. After-task:
  `docs/dev-log/after-task/2026-08-09-betabinomial-grouped-ci.md`.
- **Capacity S1 Species-XB Binomial MERGED #196 (2026-08-08).** `main` @
  `6aa8e0cb`. Live Δ abs ≈ **1.322e-9** (seed=49, rtol 1e-6). Gaussian
  skipped. After-task:
  `docs/dev-log/after-task/2026-08-08-species-xb-binomial.md`.
- **Post-#192 capacity programme ultra-plan MERGED #194 (2026-08-07).**
  `main` @ `49056186` — G0 LOCKED (Binomial required / Gaussian optional;
  ZIP+X Identity docs-only; packaging A; merge-on-green). Plan:
  `docs/dev-log/plans/2026-08-07-post-bb-x-capacity-programme.md`.
- **Post-#192 board/handover hygiene MERGED #193 (2026-08-07).** `main` @
  `2f07ad37`. Handover:
  `docs/dev-log/handover/2026-08-07-cursor-handover-post-bb-x.md`.
- **BetaBinomial+X engine Arc 1+2 MERGED #192 (2026-08-07).** `main` @
  `f56befc1` — `fit_beta_binomial_gllvm_grouped(_cov)` + bridge/`@formula`;
  light RCall Δ abs ≈1.50e-8 (seed=49). After-task:
  `docs/dev-log/after-task/2026-08-05-betabinomial-x-engine-arc12.md`. Rose
  fence: ≠ full family parity ≠ ADEMP ≠ CI for BB grouped.
- **Post-NB1 closeout programme DONE (2026-08-05).** Packaging A: #187
  hygiene + #190 Species-XB + BetaBinomial+X Identity (this PR). After-task:
  `docs/dev-log/after-task/2026-08-05-post-nb1-closeout-programme.md`.
- **Species-XB Arc 0 MERGED #190 (2026-08-05).** `main` @ `a8d19579` —
  Poisson `(0+trait):x` Δ≈4.20e-9. After-task:
  `docs/dev-log/after-task/2026-08-04-species-xb-light-rcall.md`.
- **Post-NB1 hygiene MERGED #187 (2026-08-05).** `main` @ `f230b372` —
  live Δ paste + Distributions + capabilities golden (#189 superseded).
- **NB1+X Arc 1+2 MERGED #186 (2026-08-05).** `main` @ `a100cc63` —
  `fit_nb1_gllvm_grouped_cov` + bridge/`@formula`; live Δ abs ≈1.53e-9
  (seed=48). After-task:
  `docs/dev-log/after-task/2026-08-05-nb1-x-engine-arc12.md`. Rose fence:
  ≠ full family parity.
- **NB1+X identity Arc 0 MERGED #185 (2026-08-05).** `main` @ `210de76d` —
  ACCEPTED decision
  `docs/dev-log/decisions/2026-08-05-nb1-x-dispersion-identity.md`.
- **Board / snapshot hygiene Arc 0 MERGED #183/#184 (2026-08-05).** Docs-only
  post-#181 pointer truth + merged remote-head GC. `main` @ `13d97b13`.
  After-task: `docs/dev-log/after-task/2026-08-05-board-hygiene.md`.
- **Ordinal+X light RCall Arc 2 MERGED #181 (2026-08-04).** `main` @
  `a92c5040` — `:ordinal` in X helper + light cell Δ≈5.38e-9. After-task:
  `docs/dev-log/after-task/2026-08-03-ordinal-x-arc2-parity.md`. Rose fence:
  ≠ full family parity.
- **Ordinal+X engine Arc 1 MERGED #180 (2026-08-04).** `main` @ `e4c20195` —
  `fit_ordinal_gllvm_pertrait_cov` + bridge/`@formula`. After-task:
  `docs/dev-log/after-task/2026-08-03-ordinal-x-engine.md`.
- **Ordinal+X identity Arc 0 MERGED #179 (2026-08-03).** ACCEPTED decision
  `docs/dev-log/decisions/2026-08-03-ordinal-x-cutpoint-identity.md` on `main`
  @ `0630f8e4`.
- **Gamma+X Arc 1–2 MERGED #178 (2026-08-03).** Identity + grouped_cov engine +
  OH default + Gamma+X light RCall cell on `main` @ `5f027f19`.
- **NB2/Beta+X Arc 2 MERGED #177; Windows NA #176 MERGED.**
- **Windows row-effect NA budget MERGED (2026-08-03).** PR #176 → `main`
  @ `0e241215`.
- **NB2/Beta+X engine Arc 1 MERGED (2026-08-02).** PR #175 → `main`.
- **NB2/Beta+X identity design Arc 0 MERGED (2026-08-02).** PR #174 → `main`.
  Decision: `docs/dev-log/decisions/2026-08-02-nb2-beta-x-dispersion-identity.md`.
- **Grouped dispersion one-group bug MERGED (2026-08-02).** PR #172 → `main`.
- **MC Julia capability-status MERGED (2026-08-02).** PR #173 → `main`.
- **X/covariate light logLik cohort 1 MERGED (2026-08-02).** PR #170 → `main`.
  Rose fence: not full family parity.
- **Default-route NB2/Beta per-trait φ MERGED (2026-08-02).** PR #169 → `main`.
- **Catch-up light gllvmTMB logLik oracles DONE (2026-08-01).** Named-route
  **63/63**. Rose fence: **not** full family parity.
- **Codex restart handoff for Phylo Model A redesign (2026-06-30).** PR #127
  closed/parked. Do not orphan.
  `docs/dev-log/handover/2026-06-30-codex-handover.md`.
- **Phase 0 — Team and memory scaffolding (complete, 2026-05-30).** PR #1.
- **Phase 1.1 — O(p) node-frame gradient (complete, 2026-05-30).** PR #2.
- **Phase 1.0 — RCall.jl parity suite (live on catch-up branch, 2026-08-01).**
  Light logLik oracles green on named routes; still not “full family parity.”

- **Core 0.7.0 + AGHQ M2 ledger — COMPLETE (2026-09-01, lane codex/core070-aghq-20260830).**
  533/533 required manifest rows accounted: 222 bound on verified Totoro batch
  receipts + 311 receipted dispositions (142 needs-Julia-surface, 44 spec-defect,
  125 pending maintainer decision incl. the confirmed nobs and binomial-cloglog
  cross-engine defects). Rose audit round-1 blockers fixed. Unpushed; see
  docs/dev-log/after-task/2026-09-01-m2-ledger-complete.md.

- **Core 0.7.0 + AGHQ lane (2026-09-02, Claude).** First uncancelled CI run on
  `codex/core070-aghq-20260830` (33622687447): both Julia jobs red for test-side
  reasons (deprecation-shim wording vs `@test_deprecated`; a 1e8-offset fixture at
  rtol 1e-9) plus one open NB2 Wald NaN on 1.12.7 — fixes applied, one push approved.
  ZI-trio ADEMP campaign done on Totoro (6000 fits). True-parity decision map at
  `docs/dev-log/core070/true-parity-decision-map.md`; maintainer: one-directional
  claim, oracle stays 0.7.0, second-order scope SE + vcov block + Wald endpoints.

Update this snapshot after every after-task report.

## Pre-publish gate

Before any user-facing change reaches `main`, Rose runs a narrow audit:
README, CLAUDE.md, AGENTS.md, docs, and CHANGELOG are scanned against the
engine for stale claims, broken refs, and unsupported assertions. The
`rose-pre-publish-audit` skill drives this. It is mandatory before any
release/tag, registry action, or public capability promotion.

## Merge authority

- **Self-merge (low risk):** documentation, after-task reports, audits,
  test additions that don't widen tolerances, recovery checkpoints.
- **Maintainer approval required (high risk):** any API change, formula
  grammar change, likelihood parameterisation change, version bumps,
  `.codex/agents/*` or `.agents/skills/*` edits, AGENTS.md or CLAUDE.md
  edits beyond Phase-state snapshot updates.

## Definition of Done

A task is done only when **all** of these are present:

1. Implementation in `src/`.
2. Tests in `test/` exercising the change, passing under `Pkg.test()` (full,
   incl. quality tools) and `julia --project=. test/runtests.jl` (core).
3. Docstrings on every new exported symbol.
4. Worked example or reference entry in `docs/` if the change is
   user-facing.
5. Updated `docs/dev-log/check-log.md` entry.
6. After-task report at `docs/dev-log/after-task/YYYY-MM-DD-*.md`.
7. Rose audit verdict — explicit OK or list of remaining blockers.

## Routines and robustness

| Routine | Form |
| --- | --- |
| Evidence-first rehydration | `git status` + `git rev-parse --short HEAD` + `gh run list --limit 3` before assuming repo state |
| Pre-edit lane check | `gh pr list` + `git log --all --oneline --since="6 hours ago"` before editing AGENTS.md, CLAUDE.md, or shared design docs |
| Named-perspective reporting | Status reports speak as Ada and name which perspectives reviewed |
| Recovery checkpoint | `julia --project=. tools/julia-checkpoint.jl --goal "..." --next "..."` writes a snapshot under `docs/dev-log/recovery-checkpoints/` |
| Local checks before push | `Pkg.test()` clean (full, incl. Aqua/JET); local Documenter build clean if docs touched |
| Verify CI green | `gh run view` after every push, before claiming green |
| Confidence Eye contract | Pale CI region + darker outline + darker center mark + hollow point-estimate circle; Florence owns the CairoMakie.jl port |
| Cross-project learning | Routinely scan the sister/twin projects — **gllvmTMB** (our R twin; pkgdown + repo), **drmTMB** + **DRM.jl** (the DRM family) — at session start, before docs/API slices, and at phase boundaries. Port good ideas *and improve them*; share ours back. Log in the cross-pollination tracking issue (#13). |

## Hard boundaries

- **No engine surgery on R's `gllvmTMB`** from this repo. That R package
  is a read-only reference.
- **No push without an explicit instruction** from the maintainer. Always
  commit locally first; ask before pushing.
- **Never `git add -A`** or `git add .`. Stage by name only — disjoint
  agents may be editing in parallel.
- **Test commands:** `Pkg.test()` is the full suite (incl. Aqua/JET; what CI
  runs); `julia --project=. test/runtests.jl` is the quick core run. The old
  `can not merge projects` breakage is resolved — verified macOS + CI.
- The benchmark / comparison repo (`gllvmTMB-julia-bench/`) stays local
  and is intentionally separate from this repo.
- **Private-provenance rule.** One collaborator's name (recorded only in
  the maintainer's private notes) must never appear in any public artifact:
  READMEs, docstrings, tutorials, commit messages, CHANGELOG, Documenter
  pages, or this file. The edge-incidence representation cites Bolker's
  `phylog.rmd` only. `.gitignore` carries the guard patterns; consult the
  private notes for the specific name and patterns — do not transcribe
  them here.

## Engine Quality Battery

Every algorithm shipped in Phase 1 (and every family in Phase 3) passes
**Workflow Q** — seven parallel checks before merge:

1. **FD verification** — numerical-gradient check ≤ 1e-6.
2. **Cross-check** — vs reference implementation in repo ≤ 1e-8.
3. **R-parity** — vs `gllvmTMB` via RCall.jl ≤ 1e-6, gated by
   `ENV["GLLVM_PARITY_TESTS"] == "1"` (off by default in CI).
4. **JET.jl pass** — zero type instability in the hot path.
5. **Allocs.jl pass** — zero allocation in the inner loop.
6. **Aqua.jl pass** — project hygiene clean.
7. **Multi-shape** — balanced + caterpillar trees; p ∈ {100, 1000, 10000}.

Failure on any check kicks the algorithm back to draft. After the battery
passes, Florence renders a speedup plot vs `gllvmTMB`, an after-task audit
is written, and Rose signs off. Workflow Q is the keystone routine —
defined in §7 of the reference plan.

## Cursor Cloud specific instructions

This is a pure Julia library — there is **no service, server, or UI to run**.
"Running the app" means loading the package and fitting a model, e.g.
`using GLLVModels; fit_gaussian_gllvm(y; K = 2)` (see the README Quick start).

- **Julia toolchain.** Julia (1.10 LTS, the CI primary) is installed via
  `juliaup` at `~/.juliaup/bin`. It is on `PATH` only through `~/.bashrc` /
  `~/.profile`, so non-login shells may not see `julia`; use the explicit
  `~/.juliaup/bin/julia` (or `source ~/.bashrc`) if `julia` is not found. The
  package depot and instantiated deps live in `~/.julia` and persist in the VM
  snapshot; the startup update script only refreshes them.
- **Do NOT run two Julia processes at once.** The ForwardDiff-based family
  fitters allocate several GB per fit; two concurrent `Pkg.test()` /
  `runtests.jl` / fit processes GC-thrash the 16 GB VM and *look* hung (minutes
  with no output at 100% CPU) when they are really just starved. Run tests and
  heavy fits **single-process** and be patient. If a run appears stuck, first
  check `ps -eo pid,pcpu,etimes,comm | grep julia` for a second Julia before
  assuming a real hang.
- **The full suite is slow (~50 min, single core).** Standard commands are in
  the "Standard commands" section above. `Pkg.test()` (canonical, incl.
  Aqua/JET) resolves a temp env and takes ~50 min here — first-run
  ForwardDiff/Optim compilation dominates. Long-pole files:
  `test_zero_inflated.jl`, `test_confint_family.jl`, `test_lv_ci.jl`,
  `test_missing_response_extra.jl`. Expect little interim stdout (the outer
  `Test Summary` prints at the end); this is normal, not a hang.
- **Core vs full run.** `julia --project=. test/runtests.jl` is the quick core
  run and **skips Aqua/JET** — `test_quality.jl` detects their absence and
  `@test_skip`s gracefully. Only `Pkg.test()` (which merges the parent package
  with `test/Project.toml`) exercises the Aqua hygiene + JET type-stability
  gate. Running `julia --project=test …` directly will fail because the test
  project does not list `GLLVM` as a dependency — that merge is Pkg.test-only.

<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->
