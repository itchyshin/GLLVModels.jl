# Session handover: true-parity resume (2026-09-24)

<!-- slop-ok: handover protocol requires OWED/DONE tables and exact maintainer paste strings -->

Meta: 2026-09-24 · from Cursor (Ada) · addressed to Claude Code

You are Claude, picking up the R–Julia true-parity programme on the renamed Julia package. Chat history is not authoritative; reconcile this doc with live `git` before acting.

Programme goal is not complete. Core070 ledger `FREE=0` is not true parity. Do not mark `/goal` done or bump `Project.toml` without maintainer signoff.

---

## Critical context

1. **Package rename is on `main`.** Julia package/module is **`GLLVModels`** ([PR #423](https://github.com/itchyshin/GLLVModels.jl/pull/423) merged 2026-09-17). GitHub repo slug is **`GLLVModels.jl`**; local Dropbox folder may still be `GLLVM.jl`. Documenter deploy slug aligned ([#425](https://github.com/itchyshin/GLLVModels.jl/pull/425)).
2. **Ungated true-parity engine work is exhausted.** Next parity motion requires **exact Shinichi paste strings** for DRAFT harness PRs **#399 / #409 / #410 / #411**. Do not merge those DRAFTs without paste.
3. **Multi-lane repo.** Lane preflight reported **foreign active lanes** (Claude reader arc **#444**, Codex doc cleanup **#433/#437**, Cursor latte **#463**, paste DRAFT heads, plus a **lease** on `src/grouped_nongaussian_fit.jl`). Pick **one lane** Shinichi assigns; do not bleed into protected files.
4. **gllvmTMB is read-only for engine surgery** (D-220). Tools/disposition PRs on the R twin are OK; no TMB/likelihood edits from the Julia repo.

---

## Reconciliation on pickup (Claude, 2026-09-24)

Measured against live git after this handover merged:

- `origin/main` is **`6ba1770ab`**, not `1703c54b4`: #464 (this file) and **#463** merged. #463 reverted #453, so the "#453 latte default ON" lines below no longer describe `main`.
- **#399 is CONFLICTING**, like #409/#410/#411. All four branched from `7a6fe4962`, before the #423 rename, so each needs a rename port as well as a conflict fix.
- Shinichi pasted all four gate strings in chat on 2026-09-24 and approved rebasing all four now. Execution state lives in `LOOP/lanes/true-parity-20260924/` and the pending board.

---

## Mission control (2026-09-24)

| Repo | Branch / tip | Recent merges | Next parity move |
|------|----------------|---------------|------------------|
| GLLVModels.jl | `origin/main` @ `1703c54b4` | Speed/Fir receipts (#457–#462); latte `diag_precision_kernel` default ON ([#453](https://github.com/itchyshin/GLLVModels.jl/pull/453)); rename landed (#423) | Wait for maintainer string; then rebase one DRAFT harness and run its runbook |
| gllvmTMB (twin) | `origin/main` @ `1d7e68da1` | Read-only reference | Frozen oracle `b4d5fee64def88bc768dda1f1f77c29b295edd86` unchanged |

Canonical programme docs (read before edits):

- Board: [`docs/dev-log/2026-09-14-true-parity-pending-board.md`](../2026-09-14-true-parity-pending-board.md) (tip lines inside may lag `main`; live git wins)
- Paste packet: [`docs/dev-log/owed/2026-09-16-post-402-paste-packet.md`](../owed/2026-09-16-post-402-paste-packet.md)
- Ultra-plan: [`docs/dev-log/plans/2026-09-05-true-parity-ultra-plan.md`](../plans/2026-09-05-true-parity-ultra-plan.md) (#291)
- Coordination (sibling lanes): [`docs/dev-log/coordination-board.md`](../coordination-board.md) (Active-Lane-Split; **stale** on tips; use for ownership, not as sole START HERE)

Prior true-parity handovers (historical): [`2026-09-15-mac-studio-true-parity-handover.md`](2026-09-15-mac-studio-true-parity-handover.md), [`2026-09-16-overnight-true-parity-handover.md`](2026-09-16-overnight-true-parity-handover.md).

---

## What was accomplished (since 2026-09-17 board tip)

| Item | State |
|------|--------|
| **GLLVM → GLLVModels rename** | **DONE** on `main` (#423); soft migration aid in module |
| **Documenter slug / deploy** | **DONE** (#425) |
| **#357** bridge lognormal + truncated-Poisson logLik | **DONE** (`5ee6dc596`; do not revert) |
| Ada defaults (#323 waive, matched-θ OUT, §2 Hessian A, arcG disposition) | **DONE** (decisions on board) |
| Cloud ungated SO / bridge tranche through **#391** | **DONE** on `main` (see overnight handover) |
| Speed programme (Fir DRAC, scoreboard, **#453** latte default ON) | **DONE** on `main` (≠ parity certificate) |

FINDINGS-OF-RECORD: none

---

## Classification ledger (rehydrate, then re-classify)

### DONE

- Rename to **GLLVModels** (#423, #425).
- Paste-gated **scaffold runbooks** merged via **#402** (Stage1 / S4 / Totoro checklists).
- **#357** bridge logLik receipts merged.
- Ledger gap **inventory** (not full bind).
- **#401** node24 CI bump.
- Maintainer **Ada defaults** on board (see pending board §Disposed).

### OWED (R–Julia parity only)

1. **Paste wait (Shinichi only):** no merge or execution until one exact string below is pasted in chat.
2. **DRAFT harness hygiene (mechanical, paste still required to merge):**
   - **#399** Delta dispersion A: DRAFT; head `5a4e8fec`; merge state UNKNOWN at handoff.
   - **#409** S4 probe: DRAFT; **CONFLICTING** with `main` (rebase onto `1703c54b4` when paste arrives).
   - **#410** Totoro #323 Track A: DRAFT; **CONFLICTING** (rebase when paste arrives).
   - **#411** D3 Stage 1: DRAFT; **CONFLICTING** (rebase when paste arrives).
3. **Docs tip refresh (optional, low-risk):** board + paste packet still cite tip **`8a751b55d`**; live **`origin/main`** is **`1703c54b4`**. Update tip lines and DRAFT head SHAs after rebase, without claiming new parity evidence.
4. **Programme honesty:** keep `Project.toml` at **`0.3.0`**; do not claim full 0.7 / §7 complete.

### RETRACTED / SKIP (do not reopen without explicit reverse paste)

- **#363** Cloud Agent env, **#314** old Codex handover: CONFLICTING DRAFT skip.
- **matched-θ** default cells for `beta_logit` / `nb2_log` (permanent OUT decision).
- **#384 / #391** dual Tweedie EOO: superseded; do not merge holds.

### PROTECTED (do not touch from this programme)

- **gllvmTMB** `src/` / TMB likelihood (read-only reference).
- Foreign lanes: #444 reader-documentation (Claude); #433/#437/#439 docstring cleanup (Codex/docs); #463 latte revert of #453 (Cursor speed/latte; outside parity scope).
- **Active lease** (preflight): `src/grouped_nongaussian_fit.jl` claimed by another Cursor lane.
- **`.unlazy/latte-kernel-after-448-20260923/gates/`** acceptance ledgers: **UNMET** in repo root (speed/latte slice; `handoff_gate.sh` fails until finished or declared abandoned). Not part of true-parity OWED.
- **Dropbox stale fork** row on coordination board; never write there.
- Never `git add -A`; never push without Shinichi instruction.

---

## Paste gates (exact strings → unlock)

| Paste (exact) | DRAFT | After paste (one line) |
|---------------|-------|-------------------------|
| `accept delta dispersion A` | [#399](https://github.com/itchyshin/GLLVModels.jl/pull/399) | ready+merge when green; ACCEPTED block → public `:species` → D1 remeasure |
| `G0 Stage 1` | [#411](https://github.com/itchyshin/GLLVModels.jl/pull/411) | merge harness; bounded Stage 1 slice per #402 runbook |
| `S4 probe yes` | [#409](https://github.com/itchyshin/GLLVModels.jl/pull/409) | merge harness; probe vs gllvmTMB [#1283](https://github.com/itchyshin/gllvmTMB/pull/1283) `97214679c` |
| `ack Totoro D-139 #323 Track A` | [#410](https://github.com/itchyshin/GLLVModels.jl/pull/410) | merge harness; Totoro Track A under D-139 |

Full table: [`owed/2026-09-16-post-402-paste-packet.md`](../owed/2026-09-16-post-402-paste-packet.md).

---

## Landing state (`handoff_gate.sh` 2026-09-24)

Gate **FAILED** (expected for a docs handover while other lanes carry WIP). Declared states:

| Artifact | Committed | Pushed | PR | State |
|----------|-----------|--------|-----|--------|
| `docs/claude-handover-20260924` @ (this commit) | y | pending push | open after push | **LANDING** |
| `origin/main` @ `1703c54b4` | y | y | n/a | **LANDED** |
| DRAFT **#399/#409/#410/#411** | y (remote branches) | y | open DRAFT | **CARRIED-OVER** paste-gated |
| Local checkout was `ci/documenter-gllvmodels-slug-20260918` (gone remote) | n/a | n/a | merged #425 | **DONE** (do not resume that branch) |
| Untracked `docs/dev-log/plans/2026-09-23-speed-then-20-report-sequence.md` | n | n | none | **CARRIED-OVER** optional plan; stash name `ada-handover-wip` on old branch |
| `.unlazy/latte-kernel-*` gates | y | y | n/a | **PROTECTED** foreign speed lane; gates UNMET |
| 270+ local unpushed branches | n/a | n | n/a | **IGNORE** unless Shinichi names one |

**Why gate failed:** uncommitted/unpushed noise on abandoned local branch + unmet `.unlazy` latte gates (not true-parity scope).

---

## Next immediate steps (parity only)

1. Run rehydration commands below; confirm `origin/main` still matches or update this doc’s tip line.
2. Run `~/shinichi-brain/tools/lane_preflight.sh .`; **state which lane you take**; refuse overlapping leased paths.
3. Re-read board + paste packet; classify items again (`OWED` / `DONE` / `RETRACTED` / `PROTECTED`).
4. **If Shinichi pastes** one of the four strings: rebase the matching DRAFT onto current `main`, wait for Julia CI green (Frozen R advisory fail OK), merge only that harness, then execute the merged runbook (still no R engine surgery).
5. **If no paste:** limit work to **docs tip refresh** (board/paste packet SHAs) or **stop** and report paste table to Shinichi. Do **not** invent engine slices, Totoro spend, S4 probe, Stage 1 implementation, or `Project.toml` bump.

---

## Blockers / open questions

- **Paste:** all four harness merges blocked on Shinichi exact strings (table above).
- **DRAFT conflicts:** #409/#410/#411 need rebase before merge even after paste.
- **Foreign lanes:** do not resolve #444 / #463 / reader cleanups as part of parity unless Shinichi reassigns ownership.

---

## Gotchas

- GitHub URLs use **`itchyshin/GLLVModels.jl`**; older docs say `GLLVM.jl` (redirects may work; prefer new slug in new edits).
- #453 turned latte `diag_precision_kernel` default ON; open #463 proposes revert (speed/latte lane).
- Coordination board Active-Lane-Split is historical for many rows; for true-parity rehydrate use this file plus the 2026-09-14 board, not the 2026-08-17 Cursor handover alone.
- `handoff_gate.sh` failure on `.unlazy` gates does **not** block a docs-only handover when those gates are declared PROTECTED foreign work.

---

## Environment and commands

Working directory (local):

```bash
cd "/Users/z3437171/Dropbox/Github Local/GLLVM.jl"
```

Twin (read-only):

```bash
cd "/Users/z3437171/Dropbox/Github Local/gllvmTMB"
```

Julia (CI primary 1.10):

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl          # quick core (no Aqua/JET)
julia --project=. -e 'using Pkg; Pkg.test()' # full CI parity (~50 min; single process)
```

R bridge parity tests (off by default in CI):

```bash
export GLLVM_PARITY_TESTS=1
```

Safe verification after docs-only edits:

```bash
julia --project=docs docs/make.jl
```

Do **not** stage: `.worktrees/`, unrelated untracked plans, foreign lane files, `intake/`, whole-repo sweeps.

---

## How to resume

```bash
cd "/Users/z3437171/Dropbox/Github Local/GLLVM.jl"
git fetch origin main && git checkout main && git pull origin main
git rev-parse --short origin/main
~/shinichi-brain/tools/lane_preflight.sh .
sed -n '1,200p' docs/dev-log/handover/2026-09-24-claude-handover.md
sed -n '1,100p' docs/dev-log/2026-09-14-true-parity-pending-board.md
sed -n '1,55p' docs/dev-log/owed/2026-09-16-post-402-paste-packet.md
gh pr list -R itchyshin/GLLVModels.jl --state open
for n in 399 409 410 411; do gh pr view $n -R itchyshin/GLLVModels.jl --json isDraft,mergeable,mergeStateStatus; done
```

Paste-ready prompt for a **fresh Claude session**:

```text
Read AGENTS.md and docs/dev-log/handover/2026-09-24-claude-handover.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```

---

## Files created / modified (this handover slice)

| Path | Action |
|------|--------|
| `docs/dev-log/handover/2026-09-24-claude-handover.md` | **created** (this file) |
| `AGENTS.md` | **modified** (Phase state snapshot pointer prepend) |
