# True-parity programme: maintainer board (2026-09-14, updated 2026-09-16)

STATE: **IN PROGRESS**. (The live state is the OUTCOME 2026-09-24 line below; the rest of this line is pre-paste history.) **Mac Studio owns programme (STARTED 2026-09-15).** #323, matched-θ **(C)**, §2 Hessian **(A)** disposed (Ada defaults). Ledger gap inventory done. arcG Julia-only disposition ACCEPTED (#358 + gllvmTMB #1284). Still open: S4 probe, D3 Stage 1, `Project.toml` `0.3.0`, Delta dispersion paste.

OUTCOME 2026-09-24 (Claude closeout): all four paste gates executed. #409, #410, #399 and #411 merged in that order; #469 carried the S4 wiring fixes. S4: probe ran with a probe-only shim; `pass=0 fail=2 oracle_defect=2` because the frozen recorder never attaches testthat; recorder fix requested on gllvmTMB#1283 (S4 option b). Track A (Totoro, 56 min): 14/17 required cells succeeded; NATIVE-12 `r_gradient_max` 5.90e-4 (R side), NATIVE-06 stopped at a seeded-data guard, NATIVE-10 parity cell passes; in the Julia 1.13.0 CI job the pattern flips (NATIVE-10 fails, NATIVE-12 passes), so no holdout counts as a pass. Delta A: default `:species`; D1 PASS on both cells (#470; each-own-optimum, one seed per cell). Stage 1: #411 with the σ_eps pin fix; slice PR #471. START HERE: `docs/dev-log/handover/2026-09-24-claude-handover-closeout.md`. No parity claim beyond the receipts; `Project.toml` stays `0.3.0`.

UPDATE 2026-09-24 (Claude, lane `claude/lane-true-parity-20260924`): Shinichi pasted all four gate strings in chat (`S4 probe yes`, `G0 Stage 1`, `ack Totoro D-139 #323 Track A`, `accept delta dispersion A`) and approved rebasing all four DRAFTs now. Live `origin/main` @ **`6ba1770ab`** (#464 handover; **#463 reverted #453**, so latte `diag_precision_kernel` is back to default OFF). The 2026-09-17 "MERGEABLE + Julia green" status for **#399/#409/#410/#411** is **void**: all four branched from `7a6fe4962`, before the #423 rename (`69a69b0a0`), and all four conflicted with `main` on 2026-09-24. They are being rebased and ported (conflicts plus `GLLVM` to `GLLVModels`) with fresh CI. Order: #409, #410, #399, #411 harness, then a Stage 1 slice PR. Twin gllvmTMB `origin/main` @ `1d7e68da1`; frozen oracle `b4d5fee64def88bc768dda1f1f77c29b295edd86` unchanged; `Project.toml` stays `0.3.0`. Execution receipts will land here as each gate closes; none is claimed yet.

CLOUD STOP (2026-09-16): Ungated cloud queue **exhausted** after **#391** + docs through **#416** + **[#401](https://github.com/itchyshin/GLLVM.jl/pull/401)** node24 CI **MERGED** @ **`c33745302`**; **#357** bridge receipts **MERGED on `main`** @ **`5ee6dc596`** (Mac lane - do not revert). Tip @ **`8a751b55d`**. Four paste rows tip-aligned: DRAFT **[#399](https://github.com/itchyshin/GLLVM.jl/pull/399)** @ `ee1f9a01e`, **[#411](https://github.com/itchyshin/GLLVM.jl/pull/411)** @ `5ce6659d4`, **[#409](https://github.com/itchyshin/GLLVM.jl/pull/409)** @ `69edea4c4`, **[#410](https://github.com/itchyshin/GLLVM.jl/pull/410)** @ `89816d468` — **MERGEABLE + Julia green; still DRAFT until paste** (do not merge without Shinichi paste). No execution, GP-1, free-ν, Lambda raw, or `Project.toml` bump without paste. **#363/#314** CONFLICTING DRAFT skip. Goal **not** complete.

Merged SO tranche: #374/#376/#378 + **#391** @ **`c4dba35c4`**. **Canonical paste table:** [`owed/2026-09-16-post-402-paste-packet.md`](owed/2026-09-16-post-402-paste-packet.md) (post-**#402** merge).

Rehydrate: GLLVModels.jl `origin/main` @ **`6ba1770ab`** (2026-09-24; was `8a751b55d` on 2026-09-17) (paste packet [`owed/2026-09-16-post-402-paste-packet.md`](owed/2026-09-16-post-402-paste-packet.md); #402 runbooks @ `08ca9e487`; DRAFT **#399/#409/#410/#411** heads above); gllvmTMB `origin/main` @ `1d7e68da1` (2026-09-24; was `02b46cfc8`); frozen oracle `b4d5fee64def88bc768dda1f1f77c29b295edd86`; **`Project.toml` stays `0.3.0`**. Noon (real): [`handover/2026-09-16-noon-true-parity-wake-briefing.md`](handover/2026-09-16-noon-true-parity-wake-briefing.md).

---

## Named-item scorecard (adversarial, tip `8a751b55d` + DRAFT #399/#409/#410/#411; snapshot 2026-09-17, superseded by the 2026-09-24 UPDATE above)

| Item | Verdict |
|------|---------|
| Ledger gap | **DONE** (inventory); remaining ranks are paste / Totoro / foreign / large surface |
| §2 A (delta dispersion) | **PASTE-GATED**: DRAFT [#399](https://github.com/itchyshin/GLLVM.jl/pull/399) scaffold; paste `accept delta dispersion A` |
| #347 | **DONE** (shared-η Wald MERGED); species follow-on paste-gated |
| D3 Stage 1 / S4 / Totoro | **PASTE-GATED**: DRAFT [#411](https://github.com/itchyshin/GLLVM.jl/pull/411) / [#409](https://github.com/itchyshin/GLLVM.jl/pull/409) / [#410](https://github.com/itchyshin/GLLVM.jl/pull/410); runbooks **MERGED** #402 |
| #357 | **DONE** (MERGED `5ee6dc596`; bridge logLik receipts) |
| #363 / #314 | **SKIP** (CONFLICTING DRAFT) |

---

## Disposed (2026-09-15, Ada default)

| Item | Outcome | Decision |
|------|---------|----------|
| [#323](https://github.com/itchyshin/GLLVM.jl/issues/323) Frozen R smoke | Waived (B); advisory CI stays non-gating | [`2026-09-14-advisory-frozen-r-smoke-323-pending.md`](decisions/2026-09-14-advisory-frozen-r-smoke-323-pending.md) ACCEPTED |
| matched-θ beta_logit, nb2_log | Permanent OUT (C); each-own-optimum only | [`2026-09-14-matched-theta-beta-nb2-pending.md`](decisions/2026-09-14-matched-theta-beta-nb2-pending.md) ACCEPTED |
| §2 Hessian Binomial/cloglog + Tweedie grouped | Ratify `:observed` (A); receipts + defaults signed | [`2026-09-15-second-order-hessian-s2-pending.md`](decisions/2026-09-15-second-order-hessian-s2-pending.md) ACCEPTED |
| Julia-only arcG / DRAC (parity_ledger CLOSURE) | ACCOUNTED; Julia-beyond diagnostics; not owed to R | [`2026-09-15-julia-only-arcg-disposition.md`](decisions/2026-09-15-julia-only-arcg-disposition.md) ACCEPTED |

Reverse: Shinichi may paste `reopen #323`, `reject matched-θ C`, `reject §2 hessian A`, or `reject arcG disposition` in chat; agents must revert on explicit reverse only.

---

## Still open: paste-ready replies (Shinichi only) (historical: pre-paste state, superseded by the OUTCOME 2026-09-24 line at the top)

Cloud ungated work is done. **Do not** start these without the exact paste in chat.

### S4 public-formula probe (held)

Recorder on origin: [gllvmTMB PR #1283](https://github.com/itchyshin/gllvmTMB/pull/1283) (`97214679c`). Second explicit yes only (`LOOP/GOAL.md` QS4).

```
S4 probe yes
```

### D3 loading_profile (T5 row 8)

Scout: [PR #341](https://github.com/itchyshin/GLLVM.jl/pull/341). Stage 0 merged [PR #345](https://github.com/itchyshin/GLLVM.jl/pull/345).  
Stage 1: paste `G0 Stage 1` only.

### Ledger / capability gaps (item 3)

Inventory: [`2026-09-15-true-parity-ledger-gap-inventory.md`](after-task/2026-09-15-true-parity-ledger-gap-inventory.md). Rank 7 done (arcG; #358/#1284). Aliases landed (#355). [#357](https://github.com/itchyshin/GLLVM.jl/pull/357) bridge lognormal + truncated-Poisson logLik receipts **MERGED** @ `5ee6dc596`.

### Second-order follow-up

Receipts: Delta shared-η [#347](after-task/2026-09-15-second-order-delta-followup.md) **MERGED**; OrdinalPerTrait / TweedieGrouped / Lognormal+Trunc* on main (#362/#356/#361); **MultinomialFit** native Wald on main (#366); **Student-t fixed-ν** native Wald on main (#367); **BetaBinomial shared-φ** SO cell on main ([#374](after-task/2026-09-15-betabinomial-shared-phi-so.md), `eeb7e092`); **six holdout paired SO cells** on main ([#376](after-task/2026-09-16-six-holdout-so-cells.md), `47fcb23e`) → **PARTIAL (native Wald + toy Δ)**; **Tweedie shared-power SO cell** on main ([#378](after-task/2026-09-16-tweedie-shared-power-so.md), `67247f52`) → **PARTIAL (option A)**; **Tweedie estimated-power SO (shared+species)** on main ([#391](https://github.com/itchyshin/GLLVM.jl/pull/391), `c4dba35c4`) → **PARTIAL (option A; Rose winner over #384)**; **lognormal + truncated-Poisson bridge logLik** ([#357](after-task/2026-09-15-lognormal-truncpois-bridge-receipts.md), `5ee6dc596`) → live Δ receipt (**≠** full family parity).  
Delta SO species dispersion: DRAFT [#399](https://github.com/itchyshin/GLLVM.jl/pull/399) scaffold only; still OUT / not ACCEPTED until paste `accept delta dispersion A` (then ACCEPTED block → public `:species` default → D1 remeasure → ready+merge).  
Still OUT / open: GP-1 ruling / Student-t free ν / Λ raw / Tweedie **jointly-optimised** power / BB φ pairing / Delta species dispersion paste.

### Version

Project.toml remains 0.3.0 (D-183).

### Optional: re-run #323 on Totoro

```
ack Totoro D-139 #323 Track A
```

### T4 realistic-size second-order

Blocked on D-139 ack before Totoro spend (no separate paste string beyond an explicit D-139 ack in chat).

---

## Exact Shinichi pastes required (cloud STOP) (historical: pre-paste state, superseded by the OUTCOME 2026-09-24 line at the top)

| Paste (exact) | Unlocks (one line) |
|---------------|--------------------|
| `accept delta dispersion A` | DRAFT [#399](https://github.com/itchyshin/GLLVM.jl/pull/399): ACCEPTED block → public `:species` → D1 remeasure → ready+merge |
| `G0 Stage 1` | DRAFT [#411](https://github.com/itchyshin/GLLVM.jl/pull/411) + #402 runbook; Stage 1 bounded slice after harness merge |
| `S4 probe yes` | DRAFT [#409](https://github.com/itchyshin/GLLVM.jl/pull/409); probe vs gllvmTMB #1283 (no R engine edits) |
| `ack Totoro D-139 #323 Track A` | DRAFT [#410](https://github.com/itchyshin/GLLVM.jl/pull/410); Totoro Track A under D-139 |

Canonical packet: [`owed/2026-09-16-post-402-paste-packet.md`](owed/2026-09-16-post-402-paste-packet.md).

---

## Agent rules

- Mac owns true-parity (STARTED 2026-09-15). Tip @ **`8a751b55d`** (2026-09-17; live tip in the 2026-09-24 UPDATE). **#357** **MERGED** @ `5ee6dc596` (do not revert). DRAFT **#399/#409/#410/#411** wait for paste (do not merge DRAFT harness PRs without paste). Cloud ungated **exhausted**; Shinichi pastes only for next work. **#384** **CLOSED** superseded.
- No Totoro / D-139 without `ack Totoro D-139 #323 Track A|B` (or an explicit D-139 ack naming the T4 grid).
- Do not run S4 probe without `S4 probe yes`.
- Do not bump Project.toml or claim full 0.7 / §7 complete.
- Do not cite matched-θ for beta_logit / nb2_log default cells.
- Do not treat arcG/DRAC diagnostics as a coverage certificate or R-owed port.
- No gllvmTMB engine surgery (TMB/likelihood); tools disposition PRs OK.
- Skipped unrelated opens: #363 env, #314 handover. Cloud fence: no GP-1 / free-ν / Λ raw / Delta paste / Stage1 / S4 / Totoro / Project.toml from cloud.
