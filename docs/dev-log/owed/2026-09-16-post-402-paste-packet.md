# True-parity paste packet (post-#402 paste-gated scaffolds)

<!-- slop-ok: paste-packet field labels (**STATE:** / Tip / Twin) match prior owed packets -->

STATE: **IN PROGRESS**. Do **not** mark the programme or `/goal` complete.

OUTCOME 2026-09-24 (Claude closeout): all four paste gates executed. #409, #410, #399 and #411 merged in that order; #469 carried the S4 wiring fixes. S4: probe ran with a probe-only shim; `pass=0 fail=2 oracle_defect=2` because the frozen recorder never attaches testthat; recorder fix requested on gllvmTMB#1283 (S4 option b). Track A (Totoro, 56 min): 13/16 required cells succeeded; NATIVE-12 `r_gradient_max` 5.90e-4 (R side), NATIVE-06 stopped at a seeded-data guard, NATIVE-10 parity cell passes. Delta A: default `:species`; D1 PASS on both cells (#470). Stage 1: #411 with the σ_eps pin fix; slice PR #471. START HERE: `docs/dev-log/handover/2026-09-24-claude-handover-closeout.md`. No parity claim beyond the receipts; `Project.toml` stays `0.3.0`.

UPDATE 2026-09-24 (Claude, lane `claude/lane-true-parity-20260924`): Shinichi pasted all four gate strings in chat (`S4 probe yes`, `G0 Stage 1`, `ack Totoro D-139 #323 Track A`, `accept delta dispersion A`) and approved rebasing all four DRAFTs now. Live `origin/main` @ **`6ba1770ab`** (#464 handover; **#463 reverted #453**, so latte `diag_precision_kernel` is back to default OFF). The 2026-09-17 "MERGEABLE + Julia green" status for **#399/#409/#410/#411** is **void**: all four branched from `7a6fe4962`, before the #423 rename (`69a69b0a0`), and all four conflicted with `main` on 2026-09-24. They are being rebased and ported (conflicts plus `GLLVM` to `GLLVModels`) with fresh CI. Order: #409, #410, #399, #411 harness, then a Stage 1 slice PR. Twin gllvmTMB `origin/main` @ `1d7e68da1`; frozen oracle `b4d5fee64def88bc768dda1f1f77c29b295edd86` unchanged; `Project.toml` stays `0.3.0`. Execution receipts will land here as each gate closes; none is claimed yet.

**Rehydrate:** `git fetch origin main && git rev-parse origin/main`  
Tip (2026-09-17): **`8a751b55d`** ([#419](https://github.com/itchyshin/GLLVM.jl/pull/419) Rose tip; #357 MERGED do not revert); paste DRAFT harnesses rebased on tip — **[#399](https://github.com/itchyshin/GLLVM.jl/pull/399)** @ `ee1f9a01e` (Delta A), **[#411](https://github.com/itchyshin/GLLVM.jl/pull/411)** @ `5ce6659d4` (Stage 1), **[#409](https://github.com/itchyshin/GLLVM.jl/pull/409)** @ `69edea4c4` (S4), **[#410](https://github.com/itchyshin/GLLVM.jl/pull/410)** @ `89816d468` (Totoro): **MERGEABLE + Julia green; still DRAFT until paste** (Frozen R advisory may fail).

**Twin:** gllvmTMB `origin/main` @ `1d7e68da1` (2026-09-24; was `02b46cfc8`); frozen oracle `b4d5fee64def88bc768dda1f1f77c29b295edd86`; **`Project.toml` stays `0.3.0`**.

**Board:** [`2026-09-14-true-parity-pending-board.md`](../2026-09-14-true-parity-pending-board.md)  
**Supersedes:** [`2026-09-16-post-399-paste-packet.md`](2026-09-16-post-399-paste-packet.md) (four paste strings unchanged).

---

## Adversarial scorecard (2026-09-17 @ `8a751b55d` + #402 MERGED / DRAFT harnesses tip-aligned)

| Named item | Verdict | Evidence |
|------------|---------|----------|
| Ledger gap | **DONE** (inventory) | [`after-task/2026-09-15-true-parity-ledger-gap-inventory.md`](../after-task/2026-09-15-true-parity-ledger-gap-inventory.md) |
| §2 A (delta dispersion) | **PASTE-GATED** | DRAFT **[#399](https://github.com/itchyshin/GLLVM.jl/pull/399)**; paste `accept delta dispersion A` |
| #402 runbooks (Stage1/S4/Totoro) | **DONE** | MERGED @ `08ca9e487` |
| D3 Stage 1 | **PASTE-GATED** | DRAFT **[#411](https://github.com/itchyshin/GLLVM.jl/pull/411)** + [`plans/2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md`](../plans/2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md); paste `G0 Stage 1` |
| S4 public-formula probe | **PASTE-GATED** | DRAFT **[#409](https://github.com/itchyshin/GLLVM.jl/pull/409)** + S4 checklist; paste `S4 probe yes` |
| Totoro #323 Track A | **PASTE-GATED** | DRAFT **[#410](https://github.com/itchyshin/GLLVM.jl/pull/410)** + Totoro runbook; paste `ack Totoro D-139 #323 Track A` |
| #357 bridge logLik | **DONE** | MERGED on `main` @ `5ee6dc596` |
| #363 / #314 | **SKIP** | CONFLICTING DRAFT |
| Ungated CI | **#401** node24 | **DONE** @ `c33745302` |

**Ungated implementable engine slice:** **none**. All four paste rows are DRAFT harnesses (**#399**, **#411**, **#409**, **#410**). Paste still required before merge or execution.

---

## Paste → unlocks → first action (Shinichi only)

| Paste (exact) | DRAFT / runbook | After paste |
|---------------|-----------------|-------------|
| `accept delta dispersion A` | DRAFT **[#399](https://github.com/itchyshin/GLLVM.jl/pull/399)** | ready+merge when green; ACCEPTED block + engine closeout + D1 remeasure |
| `G0 Stage 1` | DRAFT **[#411](https://github.com/itchyshin/GLLVM.jl/pull/411)** + **#402** runbook | Merge harness when green; implement bounded slice (`src/` export + fitter pin + confirmatory tests); no ledger bind without fixture evidence |
| `S4 probe yes` | DRAFT **[#409](https://github.com/itchyshin/GLLVM.jl/pull/409)** | Merge harness when green; run probe vs #1283 `97214679c`; no R `src/` edits |
| `ack Totoro D-139 #323 Track A` | DRAFT **[#410](https://github.com/itchyshin/GLLVM.jl/pull/410)** | Merge harness when green; Codex Track A on Totoro per launch pack |

---

## Quick refresh

```bash
cd "/Users/z3437171/Dropbox/Github Local/GLLVM.jl"
git fetch origin main && git rev-parse origin/main
gh pr view 399 409 410 411 --json isDraft,url
gh pr checks 401
```

Expected `origin/main` (2026-09-17): **`8a751b55d`**; live on 2026-09-24: **`6ba1770ab`**. DRAFT heads above after `gh pr update-branch --rebase` onto tip. Goal **not** complete.
