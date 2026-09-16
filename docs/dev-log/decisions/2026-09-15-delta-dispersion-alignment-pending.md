<!-- slop-ok: pending-decision field labels and option letters match prior Core070 decision docs -->
# Maintainer decision: Delta-lognormal / Delta-Gamma dispersion identity (shared Julia vs R per-trait)

**Date:** 2026-09-15  
**Status:** **PENDING_ACCEPTANCE** (unchanged; no ACCEPTED block)  
**Lane:** Cursor / Ada (true-parity programme)  
**Base:** `origin/main` @ `3091fe613` (post–PR #347 second-order Delta shared-η Wald wiring)  
**Trigger:** [`2026-09-15-second-order-delta-followup.md`](../after-task/2026-09-15-second-order-delta-followup.md) — paired Δ **D1 FAIL** (no tolerance widened)  
**Twin anchor (R):** `log_sigma_lognormal_delta` / `log_phi_gamma_delta` length **p** (`gllvmTMB.cpp` family ids 12–13)  
**Julia anchor:** `fit_delta_lognormal_gllvm` / `fit_delta_gamma_gllvm` — `disp_group ∈ {:shared, :species}` in `src/families/twopart.jl`; public fitter **default remains `:shared`** until paste; Wald `_family_ci` packs both modes; postfit `predict` / `getLV` / `residuals` handle vector σ/α (DRAFT #399).

**DRAFT scaffold (pre-paste, 2026-09-16):** DRAFT [#399](https://github.com/itchyshin/GLLVM.jl/pull/399) wires species CI, SO cells at `:species`, postfit vector dispersion, and `fit_gllvm(..., disp_group=)` for Delta. Not an ACCEPTED disposition. Not a D1 pass. Not a public `fit_gllvm` default coerce to `:species`. Merge only after paste `accept delta dispersion A`.

**Does not:** accept any option; edit R `gllvmTMB`; promote D1 pass; claim covered / Stage 1 / S4 / Totoro clearance; bump `Project.toml`; close programme §7 or true-parity destination.

---

## Question

How should the twin programme treat **continuous-part dispersion** for **`delta_lognormal`** and **`delta_gamma`** when Julia’s default paired cells use **one shared** log-scale (σ or α) and R **`gllvmTMB`** estimates **per-trait** dispersion?

Until this is signed, **second-order D1** at contract §4 tolerances on the default Delta cells is **not honest** (different models ⇒ different θ̂ ⇒ SE/vcov/CI Δ unrelated to Wald wiring alone).

---

## Measured state (2026-09-15 — PR #347 follow-up; not a disposition)

Each-own-optimum, local R + frozen-style oracle, `predictor = :shared`, Julia **shared** dispersion vs R **per-trait** (from after-task):

| Cell | logLik Δ (J−R) | SE max rel Δ | vcov Fro rel Δ | CI endpoint max Δ | D1 @ §4 |
|------|----------------:|-------------:|---------------:|------------------:|---------|
| `delta_lognormal` (seed 61) | **−1.923** | **0.145** | **0.221** | **0.041** | **FAIL** |
| `delta_gamma` (seed 62) | (same class as fid 13) | **0.212** | **0.255** | **0.082** | **FAIL** |

**Cause (proven):** R fits **p** dispersion parameters; Julia default cell fits **1**. R logLik is generically higher; second-order quantities compare unlike estimands. PR #347 **wired** shared-η Wald packing — it did **not** resolve this first-order gap.

**Prior framing:** [`2026-08-28-per-trait-dispersion-synthesis.md`](2026-08-28-per-trait-dispersion-synthesis.md) (cells 12–13); [`2026-08-28-delta-shared-predictor-identity.md`](2026-08-28-delta-shared-predictor-identity.md) (η mode only). **Out of scope here:** one-part `Lognormal` shared-σ identity ([`2026-08-15-lognormal-identity.md`](2026-08-15-lognormal-identity.md)) — do not conflate with two-part delta.

---

## Options

### (A) Align Julia default twin path to R **per-trait** dispersion

Lock the **parity-comparable estimand** to R’s default: **`disp_group = :species`** on Delta fitters for default bridge / `core070_second_order` cells (and align public `fit_gllvm` / bridge routing policy with NB2/Beta “match twin by default” — see [`2026-08-02-nb2-beta-x-dispersion-identity.md`](2026-08-02-nb2-beta-x-dispersion-identity.md)).

**Implementation follow-on (separate arc, not this doc):**

1. Extend `_family_ci` for `DeltaLogNormalFit` / `DeltaGammaFit` when `disp_group === :species` (today throws — shared-η block already exists for `:shared`).
2. Point paired cells + smoke drivers at `:species`; regenerate receipts; **rtol unchanged** (no silent widen).
3. Keep **`disp_group = :shared`** as explicit opt-in (laboratory / parsimony), not the default parity cell.

**Unlocks:** Honest first-order logLik Δ and a **path** to second-order D1 on default Delta cells after identity + FD checks.  
**Cost:** Engine-adjacent CI + confint work; p extra parameters (weak ID at small n — parity comparability, not accuracy claim).  
**Does not unlock:** D1 pass until remeasured green; programme §7; Totoro/S4; “covered” promotion.

### (B) Keep Julia **shared** dispersion as default; **permanently fence** default-cell D1

Retain one shared σ / α on the public default and paired oracle cells. Document that R default **per-trait** dispersion is a **different model**.

**Rose fence (required if (B) is accepted):**

- **≠** first-order point parity or logLik Δ promotion for default `delta_lognormal` / `delta_gamma` cells.  
- **≠** second-order **D1 pass** or §4-signed receipts on those default cells — **ever**, unless a **later** decision reverses to **(A)**.  
- **=** PR #347 Wald wiring remains valid for **shared-dispersion** Julia fits only.  
- **=** Optional **shared-φ laboratory** cell (both sides forced shared) is a **separate** fenced fixture — not a substitute for default-bridge claims.

**Unlocks:** Close the disposition without Julia dispersion expansion.  
**Cost:** Permanent default-bridge gap vs R `gllvmTMB`; inventory stays “partial / blocked” for Delta D1.

### (C) Second-order **each-own-optimum only**; matched-θ **OUT** for Delta (dispersion choice deferred or paired with A/B)

Sign that **matched-coordinates** second-order tier **never** applies to `delta_lognormal` / `delta_gamma`. Receipts stay **each-own-optimum** only (as today). **Do not** cite Delta SO smokes as D1 pass/fail for programme gates until **(A)** is implemented and remeasured, or **(B)** is accepted with the permanent D1 fence.

**If combined with (A):** EOO-only until per-trait wiring lands; then re-run D1 — matched-θ still OUT unless a future decision opens it.  
**If combined with (B):** EOO receipts record Δ for engineering; **D1 promotion stays forbidden** on default cells.

**Unlocks:** Honest tier labelling while #323 / engine cadence pauses; separates “Wald wired” from “D1 signed.”  
**Cost:** Does **not** alone fix logLik Δ; **(A)** or **(B)** still required for dispersion policy.

---

## Recommendation (Ada)

**Default letter if Shinichi wants twin-default honesty before the next engine slice:** **(A)**.

Rationale: R already uses per-trait delta dispersion; Julia **fitters** already accept `disp_group = :species` (packing + NLL); the gap is **default routing**, **`core070_second_order` cells**, and **`_family_ci` for `:species`** — the same class as NB2/Beta grouped dispersion, not a new likelihood design ([`2026-08-28-per-trait-dispersion-synthesis.md`](2026-08-28-per-trait-dispersion-synthesis.md)). **(B)** is honest but encodes a deliberate twin break on default cells. **(C)** should be **accepted together with (A) or (B)** for second-order tier wording; **(C) alone** does not resolve the dispersion mismatch.

**If the programme must stay docs-only until #323 / maintainer scale:** accept **(C) + (B)** via maintainer message (see phrases) — fence D1 and EOO receipts; reopen **(A)** post-#323.

**Avoid:** comparing shared-φ Julia to per-trait R while calling it D1 pass; pinning R to shared dispersion from this repo (read-only R twin).

---

## Maintainer reply contract (exact phrases)

| Action | Required reply |
|--------|----------------|
| Accept **(A)** — per-trait default + path to remeasure D1 | **`accept delta dispersion A`** |
| Accept **(B)** — keep shared; permanent default-cell D1 fence | **`accept delta dispersion B`** |
| Accept **(C)** — Delta SO EOO-only; matched-θ OUT | **`accept delta dispersion C`** |

For **(C) + (B)** in one beat: **`accept delta dispersion C+B`**.  
For **(C) then (A)** intent: **`accept delta dispersion A`** (C tier fence can be recorded in the acceptance block).

Until one phrase above appears in maintainer chat, **this file is PENDING** and must not be cited as **ACCEPTED** disposition.

---

## Rose fence (all options until acceptance)

- **≠** Delta second-order D1 pass at signed §4 tolerances (measured **FAIL** today).  
- **≠** “covered” / Core070 Stage 1 / S4 / Totoro campaign clearance for Delta.  
- **≠** programme §7 or true-parity destination complete.  
- **=** PR #347 claim: shared-η **Wald wiring** for **shared-dispersion** fits only.

---

## Record on acceptance

When Shinichi chooses, append a dated **ACCEPTED** block below with option letter(s) and exact reply phrase. Follow with a **bounded implementation arc** for **(A)** only (confint + cells + receipts); **(B)/(C)** are docs + contract fence updates only.

```text
(pending — no ACCEPTED block yet; paste-gate draft below is NOT live disposition)
```

<!--
=== PASTE-GATE: on `accept delta dispersion A`, replace the pending block above with: ===

## ACCEPTED — 2026-09-__ (Option A)

**Maintainer phrase:** `accept delta dispersion A`  
**Option:** **(A)** — Align Julia default twin path to R **per-trait** dispersion (`disp_group = :species` on default bridge / `fit_gllvm` / `core070_second_order` Delta cells).

**Disposition:**
- Coerce `disp_group = :species` for `DeltaLogNormal` / `DeltaGamma` when `fit_gllvm` omits `disp_group` (same pattern as NB2/Beta).
- Remeasure **D1** on default Delta SO cells at unchanged [`second-order-parity-contract.md`](../core070/second-order-parity-contract.md) §4 tolerances (no rtol widen).
- Keep `disp_group = :shared` as explicit opt-in (laboratory / parsimony).

**Rose fence (until D1 receipts green):**
- ≠ second-order D1 pass / “covered” promotion for default Delta cells.
- ≠ programme §7 or true-parity destination complete.

**Post-paste closeout:** flip coerce in `fit_gllvm.jl`, run D1 remeasure runbook, mark DRAFT #399 ready + merge on Julia 8/8.

=== END PASTE-GATE ===
-->

---

## Links (read order)

1. Follow-up measurements: [`2026-09-15-second-order-delta-followup.md`](../after-task/2026-09-15-second-order-delta-followup.md)  
2. Dispersion class synthesis: [`2026-08-28-per-trait-dispersion-synthesis.md`](2026-08-28-per-trait-dispersion-synthesis.md)  
3. Shared-η mode (orthogonal): [`2026-08-28-delta-shared-predictor-identity.md`](2026-08-28-delta-shared-predictor-identity.md)  
4. NB2/Beta twin precedent: [`2026-08-02-nb2-beta-x-dispersion-identity.md`](2026-08-02-nb2-beta-x-dispersion-identity.md)  
5. Second-order contract: [`second-order-parity-contract.md`](../core070/second-order-parity-contract.md)  
6. Matched-θ disposition pattern: [`2026-09-14-matched-theta-beta-nb2-pending.md`](2026-09-14-matched-theta-beta-nb2-pending.md)
