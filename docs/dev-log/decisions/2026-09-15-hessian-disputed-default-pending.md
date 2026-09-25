# Maintainer decision — §2 disputed Laplace Hessian defaults (cloglog / Tweedie grouped)

**Status 2026-09-25:** superseded, still PENDING_ACCEPTANCE as written, and
not the draft that was accepted. The same §2 question (cloglog / Tweedie
grouped Hessian default) was decided through a separate, differently-named
draft, `docs/dev-log/decisions/2026-09-15-second-order-hessian-s2-pending.md`,
which was ACCEPTED (option A) and applied 2026-09-15 (PR #347, merged
`3091fe61`). Do not cite this file as the operative decision.

**Date:** 2026-09-15  
**Status:** **PENDING_ACCEPTANCE** — do not cite as ACCEPTED until Shinichi pastes a reply phrase below  
**Lane:** Cursor / Ada (true-parity programme; ledger-gap inventory rank **3**)  
**Base:** `origin/main` @ `0da63860` (post–ledger gap inventory PR #346)  
**Contract anchor:** [`second-order-parity-contract.md`](../core070/second-order-parity-contract.md) §2 (Hessian convention)  
**Inventory source:** [`2026-09-15-true-parity-ledger-gap-inventory.md`](../after-task/2026-09-15-true-parity-ledger-gap-inventory.md) (rank 3)  
**Prior evidence (read-only, not acceptance):**

- Binomial **cloglog:** [`cloglog-leaf-notes.md`](../core070/cloglog-leaf-notes.md); code comment `src/families/binomial.jl:74-95` (`_default_hessian(::Binomial, ::CLogLogLink) = :observed`)
- Tweedie **grouped:** maintainer batch gate 1 (**FLIP to `:observed`**, [`2026-08-28-arc-decision-batch.md`](2026-08-28-arc-decision-batch.md)); `TweedieGroupedFit` default `hessian = :observed` (`src/families/grouped_dispersion.jl:1901-1902`, docstring `fit_tweedie_gllvm_grouped` §2018-2020)
- **Stale prose:** 2026-08-28 batch row 2 still says *"cloglog stays Fisher"*; contract §2 flags **STALE IN THE BRIEF** while HEAD already flips both routes

**Does not:** change `src/`; run Totoro; close programme §7; promote second-order parity; resolve **#323**; accept D3 Stage 1 or S4 probe; edit gllvmTMB.

---

## Question

The second-order contract treats **Binomial/cloglog** and **Tweedie grouped** (`fit_tweedie_gllvm_grouped`) as **`hessian_selector_disputed`** until maintainer signs which Laplace log-det curvature is the **programme default** for parity receipts — even though **`origin/main` already ships `:observed`** on both.

Which option locks the default for:

1. **`_default_hessian(::Binomial, ::CLogLogLink)`** (marginal likelihood + Wald path inherits via `fit.hessian`), and  
2. **`fit_tweedie_gllvm_grouped` / `TweedieGroupedFit`** default `hessian` (LogLink only for `:observed` weight)?

**Structural backdrop (frozen, not re-litigated here):** TMB/gllvmTMB Laplace log-dets use the **observed** joint curvature by construction (`second-order-parity-contract.md` §2; `gllvmtmb-parity.md` §Laplace curvature). Julia’s generic Laplace default remains `:fisher` except per-family overrides. **GP-1 stays Fisher-retained** under separate standing decision — **out of scope** for this ticket.

**Orthogonal issue:** cloglog **saturation / runaway loadings** (`LaplaceSaturationHealth`, 2026-08-28 Arc) was measured under **both** selectors; it is **not** a reason to keep `:fisher` as the **likelihood** default (`cloglog-leaf-notes.md` §Confirmed defect).

---

## Measured state @ `0da63860` (not a maintainer disposition)

| Cell | HEAD default | First-order cloglog receipt | Second-order holdout |
|------|--------------|----------------------------|----------------------|
| Binomial + **cloglog** | `:observed` | Seed-81012 fixture: Δ logLik **2.099 → 7.4e-12** vs R when `:observed` (`cloglog-leaf-notes.md`) | Contract §6: each-own allowed only with **`hessian_selector_disputed=true`** — **no promotion claim** |
| **Tweedie grouped** (LogLink) | `:observed` on fitter + struct default | No dedicated paired toy cell in the 20-cell arc (`second-order-parity-inventory.md`) | **NOT ATTEMPTED** for SO receipts; disputed flag blocks promotion |

Docs drift: `docs/src/gllvmtmb-parity.md:336-346` already describes both as **resolved on main**; contract §2 and holdout tables still say **disputed** — receipts and Rose cannot treat the user-facing pages as signed until this decision lands.

---

## Options

### (A) Ratify `:observed` as the signed programme default (both cells)

Sign that **current HEAD is correct** and the 2026-08-28 *"cloglog stays Fisher"* line is **superseded** by the 2026-09-01 cloglog diagnosis (Julia defect, not R deviation).

**Unlocks (follow-on slices, not this PR):**

- Set `hessian_selector_disputed=false` in second-order receipt schema for these cells once cascade PR lands  
- Update contract §2 net paragraph + §6 holdout row for cloglog; add Tweedie-grouped to attempted receipt class when a paired toy cell exists  
- Doc cascade only where still stale (`2026-08-28-arc-decision-batch.md` row 2 annotation, contract §2 stale flags)

**Does not unlock:** programme §7 / true-parity destination; Tweedie grouped **second-order** claim until a paired cell is run; saturation guard removal.

**Cost:** Requires a small **engine/docs cascade** commit after acceptance (explicitly **not** bundled here per lane isolation).

### (B) Revert **Binomial/cloglog** default to `:fisher`** (Tweedie grouped unchanged or ratified separately)

Restore the 2026-08-28 policy: cloglog marginal uses **Fisher** weight; rely on **`LaplaceSaturationHealth`** + warnings for unreliable optima, not observed curvature in the log-det.

**Unlocks:** Aligns short-form prose with the August batch without touching Tweedie.

**Cost / risk:** Re-opens the **~2.1 nats** first-order gap vs R on the retained fixture; contradicts quadrature + TMB structural parity rationale documented in `cloglog-leaf-notes.md`. Second-order receipts at `:fisher` would **not** match R `sdreport()` for cloglog unless R were shown to use Fisher (it does not).

**Ada view:** Only coherent if maintainer explicitly rejects the 2026-09-01 diagnosis as insufficient — not recommended.

### (C) Permanent **disputed** fence (defaults frozen in code, promotion forever blocked)

Leave HEAD defaults as-is but sign that **`hessian_selector_disputed` never clears** for cloglog and Tweedie grouped in parity receipts; no second-order **promotion** for these routes regardless of code defaults.

**Unlocks:** Closes the *decision ticket* without cascade work; honest “we will not claim SO parity here.”

**Cost:** Permanent gap vs `gllvmtmb-parity.md` “resolved” wording (would need revert or fence edit in a later docs slice); blocks rank-3 inventory intent (“unblocks promotion for those cells only”).

---

## Recommendation (Ada)

**Default if Shinichi says “use judgment”:** **(A)** — ratify `:observed` on **both** cells.

Rationale in one line: **TMB is observed by construction; cloglog `:fisher` was measured wrong on our side; Tweedie grouped flip was already maintainer-gate 1 on 2026-08-28.** The open work is **governance** (sign HEAD + clear disputed flags), not re-adjudicating curvature. Saturation stays a **post-fit health** layer, not a reason to keep Fisher on cloglog.

**Do not choose (C)** unless the programme explicitly abandons second-order claims for these two routes. **Avoid (B)** unless new evidence overturns `cloglog-leaf-notes.md`.

---

## Maintainer reply contract (exact phrases)

| Action | Required reply |
|--------|----------------|
| Accept **(A)** — ratify `:observed` both | **`accept §2 hessian A`** |
| Accept **(B)** — revert cloglog to `:fisher` | **`accept §2 hessian B`** |
| Accept **(C)** — permanent disputed fence | **`accept §2 hessian C`** |

Until one phrase above appears in maintainer chat, **this file is PENDING** and must not be cited as disposition.

**#323 / D3 Stage 1 / S4:** Unchanged — no waiver or stage authorisation in this slice.

---

## After acceptance (explicitly deferred)

1. Engine/docs cascade PR: contract §2, holdouts §6, receipt driver defaults, any stale batch row — **one concern per commit** where possible.  
2. Second-order: add paired toy cells **`binomial_cloglog`** (each-own D1) and **Tweedie grouped** when harness ready; record `hessian_selector` per contract §5.  
3. **No** `Project.toml` bump from this decision alone.

---

## Sign-off

| Role | Verdict |
|------|---------|
| Fisher | Options framed; evidence pointers to leaf + batch |
| Rose | Pending only — no ACCEPTED claim in after-task |
| Maintainer | **Awaiting paste** |
